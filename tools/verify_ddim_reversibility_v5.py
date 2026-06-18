""" audio_roundtrip_extraction 会新增一项：

"receiver_fbank_vs_sender_decoded_mel": {
  "shape": [1, 1, 1024, 64],
  "max_abs": ...,
  "mean_abs": ...,
  "rmse": ...
}
它对比的是：

发送端：
stego_x0 -> VAE decoder -> sender_mel

接收端：
stego wav -> wav_to_fbank() -> receiver_fbank / receiver_mel
也就是你要看的：

接收端由 wav 得到的 mel/fbank
vs
发送端正常得到的 mel/fbank

AudioLDM2 DDIM reversibility report
{
  "settings": {
    "model_name": "audioldm2-speech-ljspeech",
    "seed": 0,
    "ddim_steps": 100,
    "ddim_eta": 0.0,
    "guidance_scale": 3.5,
    "duration": 10.0,
    "text": "A female reporter is speaking full of emotion",
    "transcription": "Wish you have a good day",
    "bits_per_z": 1,
    "message_seed": 1234,
    "last_three_ddpm_timesteps": [
      11,
      1,
      0
    ]
  },
  "normal_replay_receiver_x2_prime_vs_sender_x2": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.0,
    "mean_abs": 0.0,
    "rmse": 0.0
  },
  "ddim_inversion_x1_prime_vs_sender_x1": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.038596127182245255,
    "mean_abs": 0.0035498596262186766,
    "rmse": 0.004566266667097807
  },
  "ddim_inversion_x2_prime_vs_sender_x2": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.4128532111644745,
    "mean_abs": 0.05031488090753555,
    "rmse": 0.06380932033061981
  },
  "zs_prime_vs_sender_zs": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 23.49013328552246,
    "mean_abs": 0.4608341455459595,
    "rmse": 1.1474034786224365
  },
  "zs_from_replayed_x2_and_inverted_x1_vs_sender_zs": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.7040129899978638,
    "mean_abs": 0.06475125253200531,
    "rmse": 0.08329103887081146
  },
  "sender_zs_vs_sender_e_t": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 4.5299530029296875e-06,
    "mean_abs": 5.175033948034979e-07,
    "rmse": 7.978398457453295e-07
  },
  "x0_self_check": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.0,
    "mean_abs": 0.0,
    "rmse": 0.0
  },
  "bit_embedding_experiment": {
    "num_symbols": 32768,
    "bits_per_z": 1,
    "num_bits": 32768,
    "bit_accuracy": 0.99981689453125,
    "bit_error_rate": 0.00018310546875,
    "symbol_accuracy": 0.99981689453125,
    "symbol_error_rate": 0.00018310546875,
    "bit_errors": 6,
    "symbol_errors": 6,
    "embedded_zs_recovered_zs_diff": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 0.8624199032783508,
      "mean_abs": 0.036335960030555725,
      "rmse": 0.054536618292331696
    },
    "stego_x1_recovered_x1_diff": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 0.04728043079376221,
      "mean_abs": 0.001992038218304515,
      "rmse": 0.0029898560605943203
    }
  },
  "audio_roundtrip_extraction": {
    "num_symbols": 32768,
    "bits_per_z": 1,
    "num_bits": 32768,
    "bit_accuracy": 0.509521484375,
    "bit_error_rate": 0.490478515625,
    "symbol_accuracy": 0.509521484375,
    "symbol_error_rate": 0.490478515625,
    "bit_errors": 16072,
    "symbol_errors": 16072,
    "stego_wav_path": "/home/ZhangYifan/ldm-stegan/AudioLDM2/tools/outputs_v5/stego_bits_1b.wav",
    "vae_encode_mode": "mode",
    "receiver_fbank_vs_sender_decoded_mel": {
      "shape": [
        1,
        1,
        1024,
        64
      ],
      "max_abs": 3.280848503112793,
      "mean_abs": 0.21013730764389038,
      "rmse": 0.34846898913383484
    },
    "audio_x0_prime_vs_stego_x0": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 1.5172107219696045,
      "mean_abs": 0.13896778225898743,
      "rmse": 0.193498432636261
    },
    "audio_x1_prime_vs_stego_x1": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 1.5549938678741455,
      "mean_abs": 0.14956115186214447,
      "rmse": 0.2042532116174698
    },
    "audio_recovered_zs_vs_embedded_zs": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 28.36388397216797,
      "mean_abs": 2.728071928024292,
      "rmse": 3.7256827354431152
    }
  }
}
Saved traced tensors to outputs_v5/ddim_bits_1b.pt
 """
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
    guidance_scale,
    unconditional_conditioning,
):
    device = receiver_x2_replay.device
    bits, symbols, embedded_zs = random_bits_to_gaussian_midpoints(
        shape, bits_per_z, message_seed, device
    )

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

    decoded_bits, decoded_symbols = decode_gaussian_bins(recovered_zs, bits_per_z)

    return {
        "bits": bits,
        "symbols": symbols,
        "embedded_zs": embedded_zs,
        "stego_x1": stego_x1,
        "stego_x0": stego_x0,
        "recovered_x1": recovered_x1,
        "recovered_zs": recovered_zs,
        "decoded_bits": decoded_bits,
        "decoded_symbols": decoded_symbols,
        "metrics": bit_accuracy_metrics(bits, decoded_bits, symbols, decoded_symbols),
    }


@torch.no_grad()
def run_audio_roundtrip_extraction(
    model,
    sampler,
    cond,
    shape,
    receiver_x2_replay,
    embedded_zs,
    stego_x0,
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

    decoded_bits, decoded_symbols = decode_gaussian_bins(audio_recovered_zs, bits_per_z)
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
            args.guidance_scale,
            unconditional_conditioning,
        )
        report["bit_embedding_experiment"] = {
            **bit_experiment["metrics"],
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
