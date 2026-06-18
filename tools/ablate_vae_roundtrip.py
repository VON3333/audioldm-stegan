# CUDA_VISIBLE_DEVICES=7 python ablate_vae_roundtrip.py --model_name audioldm2-speech-ljspeech --input_pt outputs_v5/ddim_bits_1b.pt --vae_encode_mode mode --save_pt outputs_v5/ablate_vae_roundtrip.pt
""" VAE-only roundtrip ablation report
{
  "settings": {
    "input_pt": "outputs_v5/ddim_bits_1b.pt",
    "model_name": "audioldm2-speech-ljspeech",
    "vae_encode_mode": "mode"
  },
  "x0_from_sender_mel_vs_stego_x0": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.1504276990890503,
    "mean_abs": 0.028552649542689323,
    "rmse": 0.03561815991997719
  },
  "mel_redecoded_vs_sender_mel": {
    "shape": [
      1,
      1,
      1024,
      64
    ],
    "max_abs": 0.20094609260559082,
    "mean_abs": 0.009448152966797352,
    "rmse": 0.014453847892582417
  }
}
Saved tensors to outputs_v5/ablate_vae_roundtrip.pt """

import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from audioldm2.pipeline import build_model


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


def load_stego_x0(path):
    data = torch.load(path, map_location="cpu")
    if "bit_embedding_experiment" not in data or data["bit_embedding_experiment"] is None:
        raise ValueError("Input .pt does not contain bit_embedding_experiment.")
    bit_data = data["bit_embedding_experiment"]
    if "stego_x0" not in bit_data:
        raise ValueError("Input .pt does not contain bit_embedding_experiment['stego_x0'].")
    return bit_data["stego_x0"]


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
    parser.add_argument("--vae_encode_mode", choices=["mode", "sample"], default="mode")
    parser.add_argument("--save_pt", default="", help="Optional path to save ablation tensors.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    model = build_model(
        ckpt_path=args.ckpt_path,
        model_name=args.model_name,
        device=args.device,
    )
    model.eval()

    stego_x0 = load_stego_x0(args.input_pt).to(model.device)

    with torch.no_grad():
        sender_mel = model.decode_first_stage(stego_x0)
        x0_from_sender_mel = encode_mel_to_x0(model, sender_mel, args.vae_encode_mode)
        mel_redecoded = model.decode_first_stage(x0_from_sender_mel)

    report = {
        "settings": {
            "input_pt": args.input_pt,
            "model_name": args.model_name,
            "vae_encode_mode": args.vae_encode_mode,
        },
        "x0_from_sender_mel_vs_stego_x0": diff_metrics(x0_from_sender_mel, stego_x0),
        "mel_redecoded_vs_sender_mel": diff_metrics(mel_redecoded, sender_mel),
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
            },
            args.save_pt,
        )
        print(f"Saved tensors to {args.save_pt}")


if __name__ == "__main__":
    main()
