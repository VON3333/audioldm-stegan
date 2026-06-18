import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from audioldm2.pipeline import build_model
from audioldm2.utils import default_audioldm_config, save_wave
from audioldm2.utilities.audio.stft import TacotronSTFT
from audioldm2.utilities.audio.tools import wav_to_fbank


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
def latent_to_wav(model, latent):
    mel = model.decode_first_stage(latent)
    wav = model.mel_spectrogram_to_waveform(
        mel,
        savepath="",
        bs=None,
        name=["generated"],
        save=False,
    )
    return mel, wav


def save_wav_file(model, waveform, output_path):
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    basename = os.path.basename(output_path)
    if basename.lower().endswith(".wav"):
        basename = basename[:-4]
    os.makedirs(output_dir, exist_ok=True)
    save_wave(waveform, output_dir, name=basename, samplerate=model.sampling_rate)
    return os.path.join(output_dir, basename + ".wav")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation 2: test vocoder/fbank roundtrip mel -> wav -> fbank/mel'."
    )
    parser.add_argument("--input_pt", required=True, help="Output .pt from verify_ddim_reversibility.py.")
    parser.add_argument("--model_name", default="audioldm2-speech-ljspeech")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--wav_path", default="outputs/ablate_vocoder_fbank.wav")
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
    sender_mel, waveform = latent_to_wav(model, stego_x0)
    wav_path = save_wav_file(model, waveform, args.wav_path)

    fn_stft = build_stft_from_config(args.model_name)
    receiver_fbank, _, _ = wav_to_fbank(
        wav_path,
        target_length=int(args.duration * 102.4),
        fn_STFT=fn_stft,
    )
    receiver_mel = receiver_fbank[None, None, ...].to(sender_mel.device).float()

    report = {
        "settings": {
            "input_pt": args.input_pt,
            "model_name": args.model_name,
            "duration": args.duration,
            "wav_path": wav_path,
        },
        "receiver_fbank_vs_sender_decoded_mel": diff_metrics(receiver_mel, sender_mel),
    }
    print("Vocoder/fbank roundtrip ablation report")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.save_pt:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_pt)), exist_ok=True)
        torch.save(
            {
                "report": report,
                "stego_x0": stego_x0.detach().float().cpu(),
                "sender_mel": sender_mel.detach().float().cpu(),
                "receiver_mel": receiver_mel.detach().float().cpu(),
                "receiver_fbank": receiver_fbank.detach().float().cpu(),
            },
            args.save_pt,
        )
        print(f"Saved tensors to {args.save_pt}")


if __name__ == "__main__":
    main()
