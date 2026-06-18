""" 实现不只重复相邻 latent，而是随机交织：
用 message_seed 生成随机 permutation
把同一个 bit 分散到不同 channel/time/freq 位置
改判决方式，不用固定 0 边界
现在 1 bit 解码是：
zs' > 0 -> 1
zs' < 0 -> 0
但 VAE 通道可能有整体偏移或缩放。可以估计一个校准阈值：
threshold = median(zs')
或者按同步导频 bits 估计：
z' = a * z + b + noise
然后先校正再解码。
我建议下一步不要盲目堆 repeat_k，先做 阈值校准/线性校准。如果校准后 raw accuracy 从 59% 提到比如 70%+，重复投票才会真正有效。

含义：

median_threshold：用 median(zs') 替代固定 0 阈值，属于无监督粗校准。
oracle_best_threshold：用已知原始 bits 找最优阈值，这是上限测试，用来判断“只要调阈值是否能救回来”。
class_stats：看 bit0/bit1 的 zs' 分布是否还能分开

CUDA_VISIBLE_DEVICES=7 python ablate_vae_roundtrip_full.py --model_name audioldm2-speech-ljspeech --input_pt outputs_v6/ddim_bits_1b_repeat31.pt --vae_encode_mode mode --save_pt outputs_v6/ablate_vae_repeat31.pt

VAE-only roundtrip ablation report
{
  "settings": {
    "input_pt": "outputs_v6/ddim_bits_1b_repeat31.pt",
    "model_name": "audioldm2-speech-ljspeech",
    "vae_encode_mode": "mode",
    "seed": 0,
    "ddim_steps": 100,
    "ddim_eta": 0.0,
    "guidance_scale": 3.5,
    "duration": 10.0,
    "text": "A female reporter is speaking full of emotion",
    "transcription": "Wish you have a good day",
    "bits_per_z": 1
  },
  "x0_from_sender_mel_vs_stego_x0": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.14683043956756592,
    "mean_abs": 0.029166024178266525,
    "rmse": 0.0363653264939785
  },
  "mel_redecoded_vs_sender_mel": {
    "shape": [
      1,
      1,
      1024,
      64
    ],
    "max_abs": 0.1996140480041504,
    "mean_abs": 0.009756402112543583,
    "rmse": 0.014848017133772373
  },
  "vae_channel_extraction": {
    "num_symbols": 32768,
    "bits_per_z": 1,
    "num_bits": 32768,
    "bit_accuracy": 0.59246826171875,
    "bit_error_rate": 0.40753173828125,
    "symbol_accuracy": 0.59246826171875,
    "symbol_error_rate": 0.40753173828125,
    "bit_errors": 13354,
    "symbol_errors": 13354,
    "repeat_k": 31,
    "message_bits": 1057,
    "effective_capacity_bits": 1057,
    "effective_capacity_bytes": 132.125,
    "voted_bit_accuracy": 0.7521286606788635,
    "voted_bit_error_rate": 0.24787133932113647,
    "voted_bit_errors": 262,
    "median_threshold": -0.15338923037052155,
    "median_threshold_decode": {
      "num_symbols": 32768,
      "bits_per_z": 1,
      "num_bits": 32768,
      "bit_accuracy": 0.5985107421875,
      "bit_error_rate": 0.4014892578125,
      "symbol_accuracy": 0.5985107421875,
      "symbol_error_rate": 0.4014892578125,
      "bit_errors": 13156,
      "symbol_errors": 13156
    },
    "median_threshold_vote": {
      "repeat_k": 31,
      "message_bits": 1057,
      "effective_capacity_bits": 1057,
      "effective_capacity_bytes": 132.125,
      "voted_bit_accuracy": 0.8675496578216553,
      "voted_bit_error_rate": 0.13245034217834473,
      "voted_bit_errors": 140
    },
    "oracle_best_threshold": -0.29576873779296875,
    "oracle_threshold_decode": {
      "num_symbols": 32768,
      "bits_per_z": 1,
      "num_bits": 32768,
      "bit_accuracy": 0.599639892578125,
      "bit_error_rate": 0.400360107421875,
      "symbol_accuracy": 0.599639892578125,
      "symbol_error_rate": 0.400360107421875,
      "bit_errors": 13119,
      "symbol_errors": 13119
    },
    "oracle_threshold_vote": {
      "repeat_k": 31,
      "message_bits": 1057,
      "effective_capacity_bits": 1057,
      "effective_capacity_bytes": 132.125,
      "voted_bit_accuracy": 0.7975401878356934,
      "voted_bit_error_rate": 0.20245981216430664,
      "voted_bit_errors": 214
    },
    "class_stats": {
      "z_mean_for_bit0": -0.32906875014305115,
      "z_std_for_bit0": 0.6699572801589966,
      "z_mean_for_bit1": 0.004428111016750336,
      "z_std_for_bit1": 0.6914398670196533
    },
    "vae_x1_prime_vs_stego_x1": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 0.19283318519592285,
      "mean_abs": 0.038455914705991745,
      "rmse": 0.04782741889357567
    },
    "vae_recovered_zs_vs_embedded_zs": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 3.516505002975464,
      "mean_abs": 0.7014643549919128,
      "rmse": 0.8723718523979187
    }
  }
}
Saved tensors to outputs_v6/ablate_vae_repeat31.pt

CUDA_VISIBLE_DEVICES=7 python ablate_vae_roundtrip_full.py --model_name audioldm2-speech-ljspeech --input_pt outputs_v6/ddim_bits_1b_repeat63.pt --vae_encode_mode mode --save_pt outputs_v6/ablate_vae_repeat63.pt

VAE-only roundtrip ablation report
{
  "settings": {
    "input_pt": "outputs_v6/ddim_bits_1b_repeat63.pt",
    "model_name": "audioldm2-speech-ljspeech",
    "vae_encode_mode": "mode",
    "seed": 0,
    "ddim_steps": 100,
    "ddim_eta": 0.0,
    "guidance_scale": 3.5,
    "duration": 10.0,
    "text": "A female reporter is speaking full of emotion",
    "transcription": "Wish you have a good day",
    "bits_per_z": 1
  },
  "x0_from_sender_mel_vs_stego_x0": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.15241843461990356,
    "mean_abs": 0.02934458665549755,
    "rmse": 0.0365963876247406
  },
  "mel_redecoded_vs_sender_mel": {
    "shape": [
      1,
      1,
      1024,
      64
    ],
    "max_abs": 0.19037866592407227,
    "mean_abs": 0.00965157337486744,
    "rmse": 0.014780949801206589
  },
  "vae_channel_extraction": {
    "num_symbols": 32768,
    "bits_per_z": 1,
    "num_bits": 32768,
    "bit_accuracy": 0.589263916015625,
    "bit_error_rate": 0.410736083984375,
    "symbol_accuracy": 0.589263916015625,
    "symbol_error_rate": 0.410736083984375,
    "bit_errors": 13459,
    "symbol_errors": 13459,
    "repeat_k": 63,
    "message_bits": 520,
    "effective_capacity_bits": 520,
    "effective_capacity_bytes": 65.0,
    "voted_bit_accuracy": 0.7615384459495544,
    "voted_bit_error_rate": 0.23846155405044556,
    "voted_bit_errors": 124,
    "median_threshold": -0.14793138206005096,
    "median_threshold_decode": {
      "num_symbols": 32768,
      "bits_per_z": 1,
      "num_bits": 32768,
      "bit_accuracy": 0.59881591796875,
      "bit_error_rate": 0.40118408203125,
      "symbol_accuracy": 0.59881591796875,
      "symbol_error_rate": 0.40118408203125,
      "bit_errors": 13146,
      "symbol_errors": 13146
    },
    "median_threshold_vote": {
      "repeat_k": 63,
      "message_bits": 520,
      "effective_capacity_bits": 520,
      "effective_capacity_bytes": 65.0,
      "voted_bit_accuracy": 0.932692289352417,
      "voted_bit_error_rate": 0.06730771064758301,
      "voted_bit_errors": 35
    },
    "oracle_best_threshold": -0.5566964149475098,
    "oracle_threshold_decode": {
      "num_symbols": 32768,
      "bits_per_z": 1,
      "num_bits": 32768,
      "bit_accuracy": 0.603973388671875,
      "bit_error_rate": 0.396026611328125,
      "symbol_accuracy": 0.603973388671875,
      "symbol_error_rate": 0.396026611328125,
      "bit_errors": 12977,
      "symbol_errors": 12977
    },
    "oracle_threshold_vote": {
      "repeat_k": 63,
      "message_bits": 520,
      "effective_capacity_bits": 520,
      "effective_capacity_bytes": 65.0,
      "voted_bit_accuracy": 0.5653846263885498,
      "voted_bit_error_rate": 0.4346153736114502,
      "voted_bit_errors": 226
    },
    "class_stats": {
      "z_mean_for_bit0": -0.33002084493637085,
      "z_std_for_bit0": 0.6690358519554138,
      "z_mean_for_bit1": 0.004116804338991642,
      "z_std_for_bit1": 0.6917293071746826
    },
    "vae_x1_prime_vs_stego_x1": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 0.1837519109249115,
      "mean_abs": 0.03853686526417732,
      "rmse": 0.04795510321855545
    },
    "vae_recovered_zs_vs_embedded_zs": {
      "shape": [
        1,
        8,
        256,
        16
      ],
      "max_abs": 3.354274034500122,
      "mean_abs": 0.7029349207878113,
      "rmse": 0.8747377991676331
    }
  }
}
Saved tensors to outputs_v6/ablate_vae_repeat63.pt

"""
import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from audioldm2.pipeline import build_model, make_batch_for_text_to_audio, seed_everything
from audioldm2.latent_diffusion.models.ddim import DDIMSampler


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


def symbols_to_bits(symbols, bits_per_symbol):
    weights = 2 ** torch.arange(
        bits_per_symbol - 1, -1, -1, device=symbols.device, dtype=torch.long
    )
    return ((symbols[:, None].long() & weights[None, :]) > 0).to(torch.long)


@torch.no_grad()
def decode_gaussian_bins(z, bits_per_z):
    num_bins = 2 ** bits_per_z
    u = standard_normal_cdf(z.reshape(-1).float())
    symbols = torch.floor(torch.clamp(u * num_bins, 0, num_bins - 1e-6)).long()
    bits = symbols_to_bits(symbols, bits_per_z)
    return bits, symbols


@torch.no_grad()
def decode_one_bit_with_threshold(z, threshold):
    decoded_bits = (z.reshape(-1).float() >= threshold).long().reshape(-1, 1)
    decoded_symbols = decoded_bits[:, 0].long()
    return decoded_bits, decoded_symbols


@torch.no_grad()
def best_threshold_for_one_bit(z, bits):
    scores = z.reshape(-1).float()
    labels = bits.reshape(-1).long()
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    total_ones = int(sorted_labels.sum().item())
    total_zeros = int(sorted_labels.numel() - total_ones)

    # threshold below all scores predicts all ones.
    best_errors = total_zeros
    best_threshold = sorted_scores[0] - 1e-6

    ones_left = 0
    zeros_left = 0
    n = sorted_labels.numel()
    for i in range(n):
        if int(sorted_labels[i].item()) == 1:
            ones_left += 1
        else:
            zeros_left += 1
        if i < n - 1 and sorted_scores[i].item() == sorted_scores[i + 1].item():
            continue
        # threshold after i: left predicts 0, right predicts 1.
        errors = ones_left + (total_zeros - zeros_left)
        if errors < best_errors:
            best_errors = errors
            if i < n - 1:
                best_threshold = (sorted_scores[i] + sorted_scores[i + 1]) / 2.0
            else:
                best_threshold = sorted_scores[i] + 1e-6
    return best_threshold


def one_bit_calibrated_metrics(z, bits, symbols, grouped_positions=None, message_bits=None):
    if bits.shape[1] != 1:
        return {}

    metrics = {}
    median_threshold = torch.median(z.reshape(-1).float())
    median_bits, median_symbols = decode_one_bit_with_threshold(z, median_threshold)
    metrics["median_threshold"] = float(median_threshold.item())
    metrics["median_threshold_decode"] = bit_accuracy_metrics(
        bits, median_bits, symbols, median_symbols
    )
    if grouped_positions is not None and message_bits is not None:
        _, vote_metrics = repetition_vote_metrics(
            message_bits, median_bits, grouped_positions
        )
        metrics["median_threshold_vote"] = vote_metrics

    oracle_threshold = best_threshold_for_one_bit(z, bits)
    oracle_bits, oracle_symbols = decode_one_bit_with_threshold(z, oracle_threshold)
    metrics["oracle_best_threshold"] = float(oracle_threshold.item())
    metrics["oracle_threshold_decode"] = bit_accuracy_metrics(
        bits, oracle_bits, symbols, oracle_symbols
    )
    if grouped_positions is not None and message_bits is not None:
        _, vote_metrics = repetition_vote_metrics(
            message_bits, oracle_bits, grouped_positions
        )
        metrics["oracle_threshold_vote"] = vote_metrics

    pos = z.reshape(-1).float()[bits.reshape(-1).long() == 1]
    neg = z.reshape(-1).float()[bits.reshape(-1).long() == 0]
    metrics["class_stats"] = {
        "z_mean_for_bit0": float(neg.mean().item()),
        "z_std_for_bit0": float(neg.std(unbiased=False).item()),
        "z_mean_for_bit1": float(pos.mean().item()),
        "z_std_for_bit1": float(pos.std(unbiased=False).item()),
    }
    return metrics


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


def load_experiment_data(path):
    data = torch.load(path, map_location="cpu")
    if "bit_embedding_experiment" not in data or data["bit_embedding_experiment"] is None:
        raise ValueError("Input .pt does not contain bit_embedding_experiment.")
    bit_data = data["bit_embedding_experiment"]
    required = ["stego_x0", "stego_x1", "embedded_zs", "bits", "symbols"]
    missing = [key for key in required if key not in bit_data]
    if missing:
        raise ValueError(f"Input .pt missing bit_embedding_experiment keys: {missing}")
    return data, bit_data


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
        sampler, x, t, cond, guidance_scale, unconditional_conditioning
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

    x_prev = (
        a_prev.sqrt() * pred_x0
        + (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        + sigma_t * torch.randn_like(x)
    )
    return {"x_prev": x_prev, "pred_x0": pred_x0, "e_t": e_t}


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
        x = (
            (a_next / a_prev).sqrt() * x
            + a_next.sqrt()
            * ((1.0 / a_next - 1.0).sqrt() - (1.0 / a_prev - 1.0).sqrt())
            * e_t
        )
        states.append(x)
    return states


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


@torch.no_grad()
def encode_mel_to_x0(model, mel, mode):
    posterior = model.encode_first_stage(mel)
    if mode == "mode":
        z = posterior.mode()
    elif mode == "sample":
        z = posterior.sample()
    else:
        raise ValueError(f"Unknown encode mode: {mode}")
    return model.scale_factor * z.detach()


def main():
    parser = argparse.ArgumentParser(
        description="Ablation 1: test VAE-only roundtrip x0 -> mel -> VAE encoder -> x0'."
    )
    parser.add_argument("--input_pt", required=True, help="Output .pt from verify_ddim_reversibility.py.")
    parser.add_argument("--model_name", default="audioldm2-speech-ljspeech")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("-t", "--text", default=None)
    parser.add_argument("--transcription", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ddim_steps", type=int, default=None)
    parser.add_argument("--ddim_eta", type=float, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--latent_t_per_second", type=float, default=25.6)
    parser.add_argument("--vae_encode_mode", choices=["mode", "sample"], default="mode")
    parser.add_argument("--save_pt", default="", help="Optional path to save ablation tensors.")
    args = parser.parse_args()

    source_data, bit_data = load_experiment_data(args.input_pt)
    source_settings = source_data.get("report", {}).get("settings", {})
    text = args.text if args.text is not None else source_settings.get(
        "text", "A female reporter is speaking full of emotion"
    )
    transcription = (
        args.transcription
        if args.transcription is not None
        else source_settings.get("transcription", "Wish you have a good day")
    )
    seed = args.seed if args.seed is not None else int(source_settings.get("seed", 0))
    ddim_steps = (
        args.ddim_steps
        if args.ddim_steps is not None
        else int(source_settings.get("ddim_steps", 100))
    )
    ddim_eta = (
        args.ddim_eta
        if args.ddim_eta is not None
        else float(source_settings.get("ddim_eta", 0.0))
    )
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else float(source_settings.get("guidance_scale", 3.5))
    )
    duration = (
        args.duration
        if args.duration is not None
        else float(source_settings.get("duration", 10.0))
    )
    bits_per_z = int(source_settings.get("bits_per_z", bit_data["bits"].shape[1]))

    torch.set_float32_matmul_precision("high")
    seed_everything(seed)
    model = build_model(
        ckpt_path=args.ckpt_path,
        model_name=args.model_name,
        device=args.device,
    )
    model.eval()
    model.latent_t_size = int(duration * args.latent_t_per_second)

    stego_x0 = bit_data["stego_x0"].to(model.device)
    stego_x1 = bit_data["stego_x1"].to(model.device)
    embedded_zs = bit_data["embedded_zs"].to(model.device)
    bits = bit_data["bits"].to(model.device)
    symbols = bit_data["symbols"].to(model.device)

    with torch.no_grad():
        sender_mel = model.decode_first_stage(stego_x0)
        x0_from_sender_mel = encode_mel_to_x0(model, sender_mel, args.vae_encode_mode)
        mel_redecoded = model.decode_first_stage(x0_from_sender_mel)

        cond, unconditional_conditioning = build_conditioning(
            model,
            text,
            transcription,
            batchsize=1,
            guidance_scale=guidance_scale,
        )
        sampler = DDIMSampler(model, device=model.device)
        sampler.make_schedule(
            ddim_num_steps=ddim_steps,
            ddim_eta=ddim_eta,
            verbose=False,
        )
        shape = (
            1,
            model.channels,
            model.latent_t_size,
            model.latent_f_size,
        )
        seed_everything(seed)
        x_T = torch.randn(shape, device=model.betas.device)
        receiver_states, _ = run_ddim_trace(
            sampler,
            cond,
            shape,
            x_T,
            guidance_scale,
            unconditional_conditioning,
        )
        receiver_x2_replay = receiver_states[-3]

        inv_states = ddim_invert_two_steps(
            sampler,
            x0_from_sender_mel,
            cond,
            guidance_scale,
            unconditional_conditioning,
        )
        vae_x1_prime = inv_states[1]
        a_prev_for_x1 = torch.as_tensor(
            sampler.ddim_alphas_prev[1],
            device=model.betas.device,
            dtype=vae_x1_prime.dtype,
        )
        replay_x2_to_vae_x1 = ddim_step_with_trace(
            sampler,
            receiver_x2_replay,
            torch.full(
                (shape[0],),
                int(sampler.ddim_timesteps[1]),
                device=model.betas.device,
                dtype=torch.long,
            ),
            1,
            cond,
            guidance_scale,
            unconditional_conditioning,
        )
        vae_recovered_zs = (
            vae_x1_prime
            - replay_x2_to_vae_x1["pred_x0"] * a_prev_for_x1.sqrt()
        ) / (1.0 - a_prev_for_x1).sqrt()
        decoded_bits, decoded_symbols = decode_gaussian_bins(vae_recovered_zs, bits_per_z)
        decoded_message_bits = None
        vote_metrics = None
        grouped_positions = None
        message_bits = None
        if (
            bit_data.get("message_bits") is not None
            and bit_data.get("grouped_positions") is not None
        ):
            message_bits = bit_data["message_bits"].to(model.device)
            grouped_positions = bit_data["grouped_positions"].to(model.device)
            decoded_message_bits, vote_metrics = repetition_vote_metrics(
                message_bits,
                decoded_bits,
                grouped_positions,
            )
        calibration_metrics = one_bit_calibrated_metrics(
            vae_recovered_zs,
            bits,
            symbols,
            grouped_positions=grouped_positions,
            message_bits=message_bits,
        )

    report = {
        "settings": {
            "input_pt": args.input_pt,
            "model_name": args.model_name,
            "vae_encode_mode": args.vae_encode_mode,
            "seed": seed,
            "ddim_steps": ddim_steps,
            "ddim_eta": ddim_eta,
            "guidance_scale": guidance_scale,
            "duration": duration,
            "text": text,
            "transcription": transcription,
            "bits_per_z": bits_per_z,
        },
        "x0_from_sender_mel_vs_stego_x0": diff_metrics(x0_from_sender_mel, stego_x0),
        "mel_redecoded_vs_sender_mel": diff_metrics(mel_redecoded, sender_mel),
        "vae_channel_extraction": {
            **bit_accuracy_metrics(bits, decoded_bits, symbols, decoded_symbols),
            **({} if vote_metrics is None else vote_metrics),
            **calibration_metrics,
            "vae_x1_prime_vs_stego_x1": diff_metrics(vae_x1_prime, stego_x1),
            "vae_recovered_zs_vs_embedded_zs": diff_metrics(
                vae_recovered_zs, embedded_zs
            ),
        },
    }
    print("VAE-only roundtrip ablation report")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.save_pt:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_pt)), exist_ok=True)
        torch.save(
            {
                "report": report,
                "stego_x0": stego_x0.detach().float().cpu(),
                "sender_mel": sender_mel.detach().float().cpu(),
                "x0_from_sender_mel": x0_from_sender_mel.detach().float().cpu(),
                "mel_redecoded": mel_redecoded.detach().float().cpu(),
                "vae_x1_prime": vae_x1_prime.detach().float().cpu(),
                "vae_recovered_zs": vae_recovered_zs.detach().float().cpu(),
                "decoded_bits": decoded_bits.cpu(),
                "decoded_symbols": decoded_symbols.cpu(),
                "decoded_message_bits": None
                if decoded_message_bits is None
                else decoded_message_bits.cpu(),
            },
            args.save_pt,
        )
        print(f"Saved tensors to {args.save_pt}")


if __name__ == "__main__":
    main()
