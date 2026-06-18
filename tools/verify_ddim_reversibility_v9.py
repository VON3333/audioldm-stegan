import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from audioldm2.pipeline import build_model, make_batch_for_text_to_audio, seed_everything
from audioldm2.latent_diffusion.models.ddim import DDIMSampler
from audioldm2.utils import default_audioldm_config, save_wave
from audioldm2.utilities.audio.stft import TacotronSTFT
from audioldm2.utilities.audio.tools import wav_to_fbank


def clone_cpu(x):
    return x.detach().float().cpu().clone()


def diff_metrics(a, b):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    d = a - b
    return {
        "shape": list(a.shape),
        "max_abs": float(d.abs().max().item()),
        "mean_abs": float(d.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(d * d)).item()),
    }


def standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def bits_to_symbols(bits):
    bits = bits.to(torch.long)
    bits_per_symbol = bits.shape[1]
    weights = 2 ** torch.arange(
        bits_per_symbol - 1, -1, -1, device=bits.device, dtype=torch.long
    )
    return torch.sum(bits * weights[None, :], dim=1)


def symbols_to_bits(symbols, bits_per_symbol):
    weights = 2 ** torch.arange(
        bits_per_symbol - 1, -1, -1, device=symbols.device, dtype=torch.long
    )
    return ((symbols[:, None].long() & weights[None, :]) > 0).to(torch.long)


@torch.no_grad()
def random_bits_to_gaussian_midpoints(shape, bits_per_z, seed, device):
    num_symbols = int(np.prod(shape))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    bits = torch.randint(
        0,
        2,
        (num_symbols, bits_per_z),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    symbols = bits_to_symbols(bits)
    num_bins = 2 ** bits_per_z
    u = (symbols.to(torch.float32) + 0.5) / float(num_bins)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
    )
    z = normal.icdf(u).reshape(shape)
    return bits, symbols, z


@torch.no_grad()
def random_interleaved_repetition_bits_to_gaussian_midpoints(
    num_symbols, bits_per_z, repeat_k, seed, device
):
    if bits_per_z != 1:
        raise ValueError("Interleaved repetition currently supports bits_per_z=1 only.")
    if repeat_k < 1:
        raise ValueError("repeat_k must be >= 1.")

    num_message_bits = num_symbols // repeat_k
    used_symbols = num_message_bits * repeat_k
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    message_bits = torch.randint(
        0,
        2,
        (num_message_bits, 1),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    permutation = torch.randperm(num_symbols, generator=generator, device=device)
    used_positions = permutation[:used_symbols]
    grouped_positions = used_positions.reshape(num_message_bits, repeat_k)

    flat_bits = torch.randint(
        0,
        2,
        (num_symbols, 1),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    flat_bits[grouped_positions.reshape(-1), 0] = (
        message_bits.repeat_interleave(repeat_k, dim=0).reshape(-1)
    )

    symbols = flat_bits[:, 0].long()
    u = (symbols.to(torch.float32) + 0.5) / 2.0
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
    )
    z = normal.icdf(u)
    return {
        "bits": flat_bits,
        "symbols": symbols,
        "embedded_zs": z,
        "message_bits": message_bits,
        "grouped_positions": grouped_positions,
        "permutation": permutation,
        "used_symbols": used_symbols,
        "num_message_bits": num_message_bits,
    }


def parse_channel_list(text):
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def seconds_to_time_range(start_sec, end_sec, duration, time_steps):
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}.")
    start = 0.0 if start_sec is None else float(start_sec)
    end = float(duration) if end_sec is None else float(end_sec)
    start = max(0.0, min(start, float(duration)))
    end = max(0.0, min(end, float(duration)))
    if not start < end:
        raise ValueError(
            f"Invalid embedding time window [{start}, {end}) for duration {duration}."
        )
    start_idx = int(torch.floor(torch.tensor(start / duration * time_steps)).item())
    end_idx = int(torch.ceil(torch.tensor(end / duration * time_steps)).item())
    start_idx = max(0, min(start_idx, time_steps))
    end_idx = max(0, min(end_idx, time_steps))
    if not start_idx < end_idx:
        raise ValueError(
            f"Embedding time window [{start}, {end}) maps to empty latent range "
            f"[{start_idx}, {end_idx}) with {time_steps} time steps."
        )
    return start_idx, end_idx, start, end


def make_selected_positions(
    shape,
    exclude_channels,
    embed_start_sec,
    embed_end_sec,
    duration,
    device,
):
    _, channels, time_steps, freq_bins = shape
    excluded = set(exclude_channels)
    time_start_idx, time_end_idx, start_sec, end_sec = seconds_to_time_range(
        embed_start_sec, embed_end_sec, duration, time_steps
    )
    positions = []
    for c in range(channels):
        if c in excluded:
            continue
        channel_start = c * time_steps * freq_bins
        for t in range(time_start_idx, time_end_idx):
            start = channel_start + t * freq_bins
            end = start + freq_bins
            positions.extend(range(start, end))
    metadata = {
        "embed_start_sec": start_sec,
        "embed_end_sec": end_sec,
        "time_start_idx": time_start_idx,
        "time_end_idx": time_end_idx,
    }
    return torch.tensor(positions, device=device, dtype=torch.long), metadata


@torch.no_grad()
def random_selected_bits_to_gaussian_midpoints(num_symbols, bits_per_z, seed, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    bits = torch.randint(
        0,
        2,
        (num_symbols, bits_per_z),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    symbols = bits_to_symbols(bits)
    num_bins = 2 ** bits_per_z
    u = (symbols.to(torch.float32) + 0.5) / float(num_bins)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
    )
    return bits, symbols, normal.icdf(u)


@torch.no_grad()
def decode_gaussian_bins(z, bits_per_z):
    num_bins = 2 ** bits_per_z
    u = standard_normal_cdf(z.reshape(-1).float())
    symbols = torch.floor(torch.clamp(u * num_bins, 0, num_bins - 1e-6)).long()
    bits = symbols_to_bits(symbols, bits_per_z)
    return bits, symbols


def bit_accuracy_metrics(bits, decoded_bits, symbols, decoded_symbols):
    bit_matches = bits == decoded_bits
    symbol_matches = symbols == decoded_symbols
    return {
        "num_symbols": int(symbols.numel()),
        "bits_per_z": int(bits.shape[1]),
        "num_bits": int(bits.numel()),
        "bit_accuracy": float(bit_matches.float().mean().item()),
        "bit_error_rate": float(1.0 - bit_matches.float().mean().item()),
        "symbol_accuracy": float(symbol_matches.float().mean().item()),
        "symbol_error_rate": float(1.0 - symbol_matches.float().mean().item()),
        "bit_errors": int((~bit_matches).sum().item()),
        "symbol_errors": int((~symbol_matches).sum().item()),
    }


def repetition_vote_metrics(message_bits, decoded_bits, grouped_positions):
    votes = decoded_bits[grouped_positions.reshape(-1), 0].reshape(
        grouped_positions.shape
    )
    vote_sum = votes.sum(dim=1)
    threshold = (grouped_positions.shape[1] // 2) + 1
    decoded_message_bits = (vote_sum >= threshold).long().reshape(-1, 1)
    matches = decoded_message_bits == message_bits
    return decoded_message_bits, {
        "repeat_k": int(grouped_positions.shape[1]),
        "message_bits": int(message_bits.numel()),
        "effective_capacity_bits": int(message_bits.numel()),
        "effective_capacity_bytes": float(message_bits.numel() / 8.0),
        "voted_bit_accuracy": float(matches.float().mean().item()),
        "voted_bit_error_rate": float(1.0 - matches.float().mean().item()),
        "voted_bit_errors": int((~matches).sum().item()),
    }


@torch.no_grad()
def cfg_model_output(
    sampler,
    x,
    t,
    cond,
    guidance_scale,
    unconditional_conditioning,
):
    if unconditional_conditioning is None or guidance_scale == 1.0:
        model_output = sampler.model.apply_model(x, t, cond)
    else:
        model_uncond = sampler.model.apply_model(x, t, unconditional_conditioning)
        model_cond = sampler.model.apply_model(x, t, cond)
        model_output = model_uncond + guidance_scale * (model_cond - model_uncond)

    if sampler.model.parameterization == "v":
        return sampler.model.predict_eps_from_z_and_v(x, t, model_output), model_output
    return model_output, model_output


@torch.no_grad()
def ddim_step_with_trace(
    sampler,
    x,
    t,
    index,
    cond,
    guidance_scale,
    unconditional_conditioning,
):
    b = x.shape[0]
    device = x.device
    e_t, model_output = cfg_model_output(
        sampler,
        x,
        t,
        cond,
        guidance_scale,
        unconditional_conditioning,
    )

    a_t = torch.ones((b, 1, 1, 1), device=device) * sampler.ddim_alphas[index]
    a_prev = torch.ones((b, 1, 1, 1), device=device) * sampler.ddim_alphas_prev[index]
    sigma_t = torch.ones((b, 1, 1, 1), device=device) * sampler.ddim_sigmas[index]
    sqrt_one_minus_at = (
        torch.ones((b, 1, 1, 1), device=device)
        * sampler.ddim_sqrt_one_minus_alphas[index]
    )

    if sampler.model.parameterization != "v":
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    else:
        pred_x0 = sampler.model.predict_start_from_z_and_v(x, t, model_output)

    dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
    noise = sigma_t * torch.randn_like(x)
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

    denom = (1.0 - a_prev - sigma_t**2).sqrt()
    z_from_update = (x_prev - a_prev.sqrt() * pred_x0 - noise) / denom

    return {
        "x_in": x,
        "x_prev": x_prev,
        "pred_x0": pred_x0,
        "e_t": e_t,
        "z_from_update": z_from_update,
        "noise": noise,
        "index": index,
        "step": int(t[0].item()),
    }


@torch.no_grad()
def run_ddim_trace(
    sampler,
    cond,
    shape,
    x_T,
    guidance_scale,
    unconditional_conditioning,
):
    timesteps = sampler.ddim_timesteps
    total_steps = timesteps.shape[0]
    states = [x_T]
    records = []
    x = x_T

    for i, step in enumerate(np.flip(timesteps)):
        index = total_steps - i - 1
        t = torch.full((shape[0],), int(step), device=x.device, dtype=torch.long)
        rec = ddim_step_with_trace(
            sampler,
            x,
            t,
            index,
            cond,
            guidance_scale,
            unconditional_conditioning,
        )
        x = rec["x_prev"]
        records.append(rec)
        states.append(x)

    return states, records


@torch.no_grad()
def ddim_invert_two_steps(
    sampler,
    x0,
    cond,
    guidance_scale,
    unconditional_conditioning,
):
    x = x0
    states = [x]
    records = []

    for index in range(2):
        step = int(sampler.ddim_timesteps[index])
        t = torch.full((x0.shape[0],), step, device=x0.device, dtype=torch.long)
        e_t, _ = cfg_model_output(
            sampler,
            x,
            t,
            cond,
            guidance_scale,
            unconditional_conditioning,
        )

        a_next = torch.as_tensor(
            sampler.ddim_alphas[index], device=x.device, dtype=x.dtype
        )
        a_prev = torch.as_tensor(
            sampler.ddim_alphas_prev[index], device=x.device, dtype=x.dtype
        )
        xt_weighted = (a_next / a_prev).sqrt() * x
        weighted_noise = (
            a_next.sqrt()
            * ((1.0 / a_next - 1.0).sqrt() - (1.0 / a_prev - 1.0).sqrt())
            * e_t
        )
        x_next = xt_weighted + weighted_noise
        records.append({"x_in": x, "x_next": x_next, "e_t": e_t, "index": index, "step": step})
        x = x_next
        states.append(x)

    return states, records


@torch.no_grad()
def build_conditioning(model, text, transcription, batchsize, guidance_scale):
    batch = make_batch_for_text_to_audio(
        text,
        transcription=transcription,
        waveform=None,
        batchsize=batchsize,
    )
    z, cond = model.get_input(
        batch,
        model.first_stage_key,
        unconditional_prob_cfg=0.0,
    )
    cond = model.filter_useful_cond_dict(cond)
    batch_size = z.shape[0]

    unconditional_conditioning = None
    if guidance_scale != 1.0:
        unconditional_conditioning = {}
        for key in model.cond_stage_model_metadata:
            model_idx = model.cond_stage_model_metadata[key]["model_idx"]
            unconditional_conditioning[key] = model.cond_stage_models[
                model_idx
            ].get_unconditional_condition(batch_size)

    return cond, unconditional_conditioning


def print_metrics(title, metrics):
    print(title)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


@torch.no_grad()
def latent_to_waveform(model, latent):
    mel = model.decode_first_stage(latent)
    waveform = model.mel_spectrogram_to_waveform(
        mel,
        savepath="",
        bs=None,
        name=["generated"],
        save=False,
    )
    return waveform


@torch.no_grad()
def latent_to_mel(model, latent):
    return model.decode_first_stage(latent)


def save_waveform_file(model, latent, output_path):
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    basename = os.path.basename(output_path)
    if basename.lower().endswith(".wav"):
        basename = basename[:-4]
    os.makedirs(output_dir, exist_ok=True)
    waveform = latent_to_waveform(model, latent)
    save_wave(
        waveform,
        output_dir,
        name=basename,
        samplerate=model.sampling_rate,
    )
    return os.path.join(output_dir, basename + ".wav")


def build_stft_from_config(model_name):
    config = default_audioldm_config(model_name)
    return TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )


@torch.no_grad()
def wav_path_to_x0(model, wav_path, model_name, duration, vae_encode_mode):
    fn_stft = build_stft_from_config(model_name)
    fbank, _, _ = wav_to_fbank(
        wav_path,
        target_length=int(duration * 102.4),
        fn_STFT=fn_stft,
    )
    x = fbank[None, None, ...].to(model.device).float()
    posterior = model.encode_first_stage(x)
    if vae_encode_mode == "sample":
        z = posterior.sample()
    elif vae_encode_mode == "mode":
        z = posterior.mode()
    else:
        raise ValueError(f"Unknown VAE encode mode: {vae_encode_mode}")
    return model.scale_factor * z.detach(), fbank


@torch.no_grad()
def run_bit_embedding_experiment(
    sampler,
    cond,
    shape,
    receiver_x2_replay,
    sender_x2_to_x1,
    bits_per_z,
    message_seed,
    repeat_k,
    interleave_repetition,
    exclude_channels,
    embed_start_sec,
    embed_end_sec,
    duration,
    guidance_scale,
    unconditional_conditioning,
):
    device = receiver_x2_replay.device
    selected_positions, position_metadata = make_selected_positions(
        shape,
        exclude_channels,
        embed_start_sec,
        embed_end_sec,
        duration,
        device,
    )
    if selected_positions.numel() == 0:
        raise ValueError("No latent positions left after channel/time selection.")

    embedded_zs = sender_x2_to_x1["z_from_update"].clone()
    flat_embedded = embedded_zs.reshape(-1)
    repetition = None
    if interleave_repetition or repeat_k > 1:
        repetition = random_interleaved_repetition_bits_to_gaussian_midpoints(
            int(selected_positions.numel()), bits_per_z, repeat_k, message_seed, device
        )
        local_used = repetition["permutation"][: repetition["used_symbols"]]
        bits = repetition["bits"][local_used]
        symbols = repetition["symbols"][local_used]
        used_positions = selected_positions[local_used]
        grouped_positions = used_positions.reshape(repetition["num_message_bits"], repeat_k)
        flat_embedded[used_positions] = repetition["embedded_zs"][: repetition["used_symbols"]]
    else:
        bits, symbols, selected_zs = random_selected_bits_to_gaussian_midpoints(
            int(selected_positions.numel()), bits_per_z, message_seed, device
        )
        used_positions = selected_positions
        grouped_positions = None
        flat_embedded[used_positions] = selected_zs

    a_prev_for_x1 = torch.as_tensor(
        sampler.ddim_alphas_prev[1], device=device, dtype=receiver_x2_replay.dtype
    )
    stego_x1 = (
        a_prev_for_x1.sqrt() * sender_x2_to_x1["pred_x0"]
        + (1.0 - a_prev_for_x1).sqrt() * embedded_zs
    )

    stego_x1_to_x0 = ddim_step_with_trace(
        sampler,
        stego_x1,
        torch.full(
            (shape[0],),
            int(sampler.ddim_timesteps[0]),
            device=device,
            dtype=torch.long,
        ),
        0,
        cond,
        guidance_scale,
        unconditional_conditioning,
    )
    stego_x0 = stego_x1_to_x0["x_prev"]

    inv_states, _ = ddim_invert_two_steps(
        sampler,
        stego_x0,
        cond,
        guidance_scale,
        unconditional_conditioning,
    )
    recovered_x1 = inv_states[1]

    replay_x2_to_recovered_x1 = ddim_step_with_trace(
        sampler,
        receiver_x2_replay,
        torch.full(
            (shape[0],),
            int(sampler.ddim_timesteps[1]),
            device=device,
            dtype=torch.long,
        ),
        1,
        cond,
        guidance_scale,
        unconditional_conditioning,
    )
    recovered_zs = (
        recovered_x1
        - replay_x2_to_recovered_x1["pred_x0"] * a_prev_for_x1.sqrt()
    ) / (1.0 - a_prev_for_x1).sqrt()

    all_decoded_bits, all_decoded_symbols = decode_gaussian_bins(recovered_zs, bits_per_z)
    decoded_bits = all_decoded_bits[used_positions]
    decoded_symbols = all_decoded_symbols[used_positions]
    voted_message_bits = None
    vote_metrics = None
    if repetition is not None:
        voted_message_bits, vote_metrics = repetition_vote_metrics(
            repetition["message_bits"], all_decoded_bits, grouped_positions
        )

    result = {
        "bits": bits,
        "symbols": symbols,
        "selected_positions": selected_positions,
        "used_positions": used_positions,
        "exclude_channels": torch.tensor(exclude_channels, dtype=torch.long),
        "position_metadata": position_metadata,
        "embedded_zs": embedded_zs,
        "stego_x1": stego_x1,
        "stego_x0": stego_x0,
        "recovered_x1": recovered_x1,
        "recovered_zs": recovered_zs,
        "decoded_bits": decoded_bits,
        "decoded_symbols": decoded_symbols,
        "metrics": bit_accuracy_metrics(bits, decoded_bits, symbols, decoded_symbols),
    }
    if repetition is not None:
        result.update(
            {
                "message_bits": repetition["message_bits"],
                "decoded_message_bits": voted_message_bits,
                "grouped_positions": grouped_positions,
                "permutation": repetition["permutation"],
                "vote_metrics": vote_metrics,
            }
        )
    return result


@torch.no_grad()
def run_audio_roundtrip_extraction(
    model,
    sampler,
    cond,
    shape,
    receiver_x2_replay,
    embedded_zs,
    stego_x0,
    used_positions,
    bits,
    symbols,
    bits_per_z,
    stego_wav_path,
    model_name,
    duration,
    vae_encode_mode,
    guidance_scale,
    unconditional_conditioning,
):
    audio_x0_prime, receiver_fbank = wav_path_to_x0(
        model,
        stego_wav_path,
        model_name,
        duration,
        vae_encode_mode,
    )
    sender_mel = latent_to_mel(model, stego_x0)
    receiver_mel = receiver_fbank[None, None, ...].to(sender_mel.device).float()
    inv_states, _ = ddim_invert_two_steps(
        sampler,
        audio_x0_prime,
        cond,
        guidance_scale,
        unconditional_conditioning,
    )
    audio_x1_prime = inv_states[1]

    device = receiver_x2_replay.device
    a_prev_for_x1 = torch.as_tensor(
        sampler.ddim_alphas_prev[1], device=device, dtype=receiver_x2_replay.dtype
    )
    replay_x2_to_audio_x1 = ddim_step_with_trace(
        sampler,
        receiver_x2_replay,
        torch.full(
            (shape[0],),
            int(sampler.ddim_timesteps[1]),
            device=device,
            dtype=torch.long,
        ),
        1,
        cond,
        guidance_scale,
        unconditional_conditioning,
    )
    audio_recovered_zs = (
        audio_x1_prime
        - replay_x2_to_audio_x1["pred_x0"] * a_prev_for_x1.sqrt()
    ) / (1.0 - a_prev_for_x1).sqrt()

    all_decoded_bits, all_decoded_symbols = decode_gaussian_bins(audio_recovered_zs, bits_per_z)
    decoded_bits = all_decoded_bits[used_positions]
    decoded_symbols = all_decoded_symbols[used_positions]
    return {
        "audio_x0_prime": audio_x0_prime,
        "audio_x1_prime": audio_x1_prime,
        "audio_recovered_zs": audio_recovered_zs,
        "decoded_bits": decoded_bits,
        "decoded_symbols": decoded_symbols,
        "sender_mel": sender_mel,
        "receiver_mel": receiver_mel,
        "receiver_fbank": receiver_fbank,
        "metrics": bit_accuracy_metrics(bits, decoded_bits, symbols, decoded_symbols),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify DDIM last-step reversibility for AudioLDM2 speech latent diffusion."
    )
    parser.add_argument(
        "-t",
        "--text",
        default="A female reporter is speaking full of emotion",
        help="Speaker/style prompt.",
    )
    parser.add_argument(
        "--transcription",
        default="Wish you have a good day",
        help="TTS transcription.",
    )
    parser.add_argument(
        "--model_name",
        default="audioldm2-speech-ljspeech",
        help="AudioLDM2 checkpoint name.",
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Optional local .pth checkpoint path. If set, skip Hugging Face download.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--latent_t_per_second", type=float, default=25.6)
    parser.add_argument(
        "--bits_per_z",
        type=int,
        default=1,
        help="Random payload bits embedded per latent element for the real bit experiment.",
    )
    parser.add_argument(
        "--message_seed",
        type=int,
        default=1234,
        help="Seed for the random payload bits.",
    )
    parser.add_argument(
        "--repeat_k",
        type=int,
        default=1,
        help="Repeat each message bit across k randomly interleaved latent positions.",
    )
    parser.add_argument(
        "--interleave_repetition",
        action="store_true",
        help="Use message_seed permutation to scatter repeated bit copies across latent positions.",
    )
    parser.add_argument(
        "--exclude_channels",
        default="",
        help="Comma-separated latent channels to leave unmodified, e.g. '1,2'.",
    )
    parser.add_argument(
        "--embed_start_sec",
        type=float,
        default=None,
        help="Start time in seconds for payload embedding. Defaults to 0.",
    )
    parser.add_argument(
        "--embed_end_sec",
        type=float,
        default=None,
        help="End time in seconds for payload embedding. Defaults to duration.",
    )
    parser.add_argument(
        "--save_stego_wav",
        default="",
        help="Optional .wav path for audio generated from embedded stego x0.",
    )
    parser.add_argument(
        "--save_cover_wav",
        default="",
        help="Optional .wav path for audio generated from the original cover x0.",
    )
    parser.add_argument(
        "--audio_roundtrip",
        action="store_true",
        help="Run full wav -> fbank/mel -> VAE encoder -> x0' extraction for stego audio.",
    )
    parser.add_argument(
        "--vae_encode_mode",
        choices=["mode", "sample"],
        default="mode",
        help="Use VAE posterior mean or sample when encoding received wav to x0'.",
    )
    parser.add_argument("--save_pt", default="", help="Optional path for traced tensors.")
    args = parser.parse_args()
    exclude_channels = parse_channel_list(args.exclude_channels)

    if args.ddim_eta != 0.0:
        print(
            "Warning: ddim_eta is not 0. DDIM will inject fresh Gaussian noise, "
            "so strict inversion should not be expected."
        )

    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    model = build_model(
        ckpt_path=args.ckpt_path,
        model_name=args.model_name,
        device=args.device,
    )
    model.latent_t_size = int(args.duration * args.latent_t_per_second)
    model.eval()

    cond, unconditional_conditioning = build_conditioning(
        model,
        args.text,
        args.transcription,
        batchsize=1,
        guidance_scale=args.guidance_scale,
    )

    sampler = DDIMSampler(model, device=model.device)
    sampler.make_schedule(
        ddim_num_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        verbose=False,
    )

    shape = (
        1,
        model.channels,
        model.latent_t_size,
        model.latent_f_size,
    )
    device = model.betas.device

    seed_everything(args.seed)
    x_T = torch.randn(shape, device=device)
    sender_states, sender_records = run_ddim_trace(
        sampler,
        cond,
        shape,
        x_T,
        args.guidance_scale,
        unconditional_conditioning,
    )

    # x2 -> x1 -> x0 are the last two denoising transitions.
    sender_x2 = sender_states[-3]
    sender_x1 = sender_states[-2]
    sender_x0 = sender_states[-1]
    sender_x2_to_x1 = sender_records[-2]
    sender_zs = sender_x2_to_x1["z_from_update"]

    receiver_states, receiver_records = run_ddim_trace(
        sampler,
        cond,
        shape,
        x_T.clone(),
        args.guidance_scale,
        unconditional_conditioning,
    )
    receiver_x2_replay = receiver_states[-3]

    inv_states, inv_records = ddim_invert_two_steps(
        sampler,
        sender_x0,
        cond,
        args.guidance_scale,
        unconditional_conditioning,
    )
    inv_x1 = inv_states[1]
    inv_x2 = inv_states[2]

    inv_x2_to_x1 = ddim_step_with_trace(
        sampler,
        inv_x2,
        torch.full(
            (shape[0],),
            int(sampler.ddim_timesteps[1]),
            device=device,
            dtype=torch.long,
        ),
        1,
        cond,
        args.guidance_scale,
        unconditional_conditioning,
    )
    a_prev_for_zs = torch.as_tensor(
        sampler.ddim_alphas_prev[1], device=device, dtype=inv_x1.dtype
    )
    inv_zs = (inv_x1 - inv_x2_to_x1["pred_x0"] * a_prev_for_zs.sqrt()) / (
        1.0 - a_prev_for_zs
    ).sqrt()

    replay_x2_to_inv_x1 = ddim_step_with_trace(
        sampler,
        receiver_x2_replay,
        torch.full(
            (shape[0],),
            int(sampler.ddim_timesteps[1]),
            device=device,
            dtype=torch.long,
        ),
        1,
        cond,
        args.guidance_scale,
        unconditional_conditioning,
    )
    replay_x2_inv_x1_zs = (
        inv_x1 - replay_x2_to_inv_x1["pred_x0"] * a_prev_for_zs.sqrt()
    ) / (1.0 - a_prev_for_zs).sqrt()

    report = {
        "settings": {
            "model_name": args.model_name,
            "seed": args.seed,
            "ddim_steps": args.ddim_steps,
            "ddim_eta": args.ddim_eta,
            "guidance_scale": args.guidance_scale,
            "duration": args.duration,
            "text": args.text,
            "transcription": args.transcription,
            "bits_per_z": args.bits_per_z,
            "message_seed": args.message_seed,
            "repeat_k": args.repeat_k,
            "interleave_repetition": bool(args.interleave_repetition or args.repeat_k > 1),
            "exclude_channels": exclude_channels,
            "embed_start_sec": 0.0 if args.embed_start_sec is None else args.embed_start_sec,
            "embed_end_sec": args.duration if args.embed_end_sec is None else args.embed_end_sec,
            "last_three_ddpm_timesteps": [
                int(sampler.ddim_timesteps[1]),
                int(sampler.ddim_timesteps[0]),
                0,
            ],
        },
        "normal_replay_receiver_x2_prime_vs_sender_x2": diff_metrics(
            receiver_x2_replay, sender_x2
        ),
        "ddim_inversion_x1_prime_vs_sender_x1": diff_metrics(inv_x1, sender_x1),
        "ddim_inversion_x2_prime_vs_sender_x2": diff_metrics(inv_x2, sender_x2),
        "zs_prime_vs_sender_zs": diff_metrics(inv_zs, sender_zs),
        "zs_from_replayed_x2_and_inverted_x1_vs_sender_zs": diff_metrics(
            replay_x2_inv_x1_zs, sender_zs
        ),
        "sender_zs_vs_sender_e_t": diff_metrics(sender_zs, sender_x2_to_x1["e_t"]),
        "x0_self_check": diff_metrics(sender_x0, receiver_states[-1]),
    }

    bit_experiment = None
    if args.bits_per_z > 0:
        bit_experiment = run_bit_embedding_experiment(
            sampler,
            cond,
            shape,
            receiver_x2_replay,
            sender_x2_to_x1,
            args.bits_per_z,
            args.message_seed,
            args.repeat_k,
            bool(args.interleave_repetition or args.repeat_k > 1),
            exclude_channels,
            args.embed_start_sec,
            args.embed_end_sec,
            args.duration,
            args.guidance_scale,
            unconditional_conditioning,
        )
        vote_metrics = bit_experiment.get("vote_metrics")
        report["bit_embedding_experiment"] = {
            **bit_experiment["metrics"],
            **({} if vote_metrics is None else vote_metrics),
            "selected_positions": int(bit_experiment["selected_positions"].numel()),
            "used_positions": int(bit_experiment["used_positions"].numel()),
            **bit_experiment["position_metadata"],
            "embedded_zs_recovered_zs_diff": diff_metrics(
                bit_experiment["recovered_zs"], bit_experiment["embedded_zs"]
            ),
            "stego_x1_recovered_x1_diff": diff_metrics(
                bit_experiment["recovered_x1"], bit_experiment["stego_x1"]
            ),
        }

    if args.save_cover_wav:
        save_waveform_file(model, sender_x0, args.save_cover_wav)

    stego_wav_path = args.save_stego_wav
    if args.save_stego_wav:
        if bit_experiment is None:
            raise ValueError("--save_stego_wav requires --bits_per_z > 0")
        stego_wav_path = save_waveform_file(
            model, bit_experiment["stego_x0"], args.save_stego_wav
        )

    audio_roundtrip = None
    if args.audio_roundtrip:
        if bit_experiment is None:
            raise ValueError("--audio_roundtrip requires --bits_per_z > 0")
        if not stego_wav_path:
            stego_wav_path = os.path.abspath("outputs/stego_roundtrip_tmp.wav")
            stego_wav_path = save_waveform_file(
                model, bit_experiment["stego_x0"], stego_wav_path
            )
        audio_roundtrip = run_audio_roundtrip_extraction(
            model,
            sampler,
            cond,
            shape,
            receiver_x2_replay,
            bit_experiment["embedded_zs"],
            bit_experiment["stego_x0"],
            bit_experiment["used_positions"],
            bit_experiment["bits"],
            bit_experiment["symbols"],
            args.bits_per_z,
            stego_wav_path,
            args.model_name,
            args.duration,
            args.vae_encode_mode,
            args.guidance_scale,
            unconditional_conditioning,
        )
        report["audio_roundtrip_extraction"] = {
            **audio_roundtrip["metrics"],
            "stego_wav_path": stego_wav_path,
            "vae_encode_mode": args.vae_encode_mode,
            "receiver_fbank_vs_sender_decoded_mel": diff_metrics(
                audio_roundtrip["receiver_mel"], audio_roundtrip["sender_mel"]
            ),
            "audio_x0_prime_vs_stego_x0": diff_metrics(
                audio_roundtrip["audio_x0_prime"], bit_experiment["stego_x0"]
            ),
            "audio_x1_prime_vs_stego_x1": diff_metrics(
                audio_roundtrip["audio_x1_prime"], bit_experiment["stego_x1"]
            ),
            "audio_recovered_zs_vs_embedded_zs": diff_metrics(
                audio_roundtrip["audio_recovered_zs"], bit_experiment["embedded_zs"]
            ),
        }

    print_metrics("AudioLDM2 DDIM reversibility report", report)

    if args.save_pt:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_pt)), exist_ok=True)
        torch.save(
            {
                "report": report,
                "sender": {
                    "x2": clone_cpu(sender_x2),
                    "x1": clone_cpu(sender_x1),
                    "x0": clone_cpu(sender_x0),
                    "zs": clone_cpu(sender_zs),
                },
                "receiver": {
                    "x2_replay": clone_cpu(receiver_x2_replay),
                    "x1_inverted": clone_cpu(inv_x1),
                    "x2_inverted": clone_cpu(inv_x2),
                    "zs_inverted": clone_cpu(inv_zs),
                    "zs_from_replayed_x2_and_inverted_x1": clone_cpu(
                        replay_x2_inv_x1_zs
                    ),
                },
                "bit_embedding_experiment": None
                if bit_experiment is None
                else {
                    "bits": bit_experiment["bits"].cpu(),
                    "decoded_bits": bit_experiment["decoded_bits"].cpu(),
                    "symbols": bit_experiment["symbols"].cpu(),
                    "decoded_symbols": bit_experiment["decoded_symbols"].cpu(),
                    "selected_positions": bit_experiment["selected_positions"].cpu(),
                    "used_positions": bit_experiment["used_positions"].cpu(),
                    "exclude_channels": bit_experiment["exclude_channels"].cpu(),
                    "position_metadata": bit_experiment["position_metadata"],
                    "message_bits": None
                    if "message_bits" not in bit_experiment
                    else bit_experiment["message_bits"].cpu(),
                    "decoded_message_bits": None
                    if "decoded_message_bits" not in bit_experiment
                    else bit_experiment["decoded_message_bits"].cpu(),
                    "grouped_positions": None
                    if "grouped_positions" not in bit_experiment
                    else bit_experiment["grouped_positions"].cpu(),
                    "permutation": None
                    if "permutation" not in bit_experiment
                    else bit_experiment["permutation"].cpu(),
                    "embedded_zs": clone_cpu(bit_experiment["embedded_zs"]),
                    "recovered_zs": clone_cpu(bit_experiment["recovered_zs"]),
                    "stego_x1": clone_cpu(bit_experiment["stego_x1"]),
                    "stego_x0": clone_cpu(bit_experiment["stego_x0"]),
                    "recovered_x1": clone_cpu(bit_experiment["recovered_x1"]),
                },
                "audio_roundtrip_extraction": None
                if audio_roundtrip is None
                else {
                    "audio_x0_prime": clone_cpu(audio_roundtrip["audio_x0_prime"]),
                    "audio_x1_prime": clone_cpu(audio_roundtrip["audio_x1_prime"]),
                    "audio_recovered_zs": clone_cpu(
                        audio_roundtrip["audio_recovered_zs"]
                    ),
                    "decoded_bits": audio_roundtrip["decoded_bits"].cpu(),
                    "decoded_symbols": audio_roundtrip["decoded_symbols"].cpu(),
                    "sender_mel": clone_cpu(audio_roundtrip["sender_mel"]),
                    "receiver_mel": clone_cpu(audio_roundtrip["receiver_mel"]),
                    "receiver_fbank": audio_roundtrip["receiver_fbank"].cpu(),
                },
            },
            args.save_pt,
        )
        print(f"Saved traced tensors to {args.save_pt}")


if __name__ == "__main__":
    main()
