""" 只验证zs是否可恢复，没有嵌入秘密信息 """
""" 
AudioLDM2 DDIM reversibility report -50
{
  "settings": {
    "model_name": "audioldm2-speech-ljspeech",
    "seed": 0,
    "ddim_steps": 50,
    "ddim_eta": 0.0,
    "guidance_scale": 3.5,
    "duration": 10.0,
    "text": "A female reporter is speaking full of emotion",
    "transcription": "Wish you have a good day",
    "last_three_ddpm_timesteps": [
      21,
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
    "max_abs": 0.06901095062494278,
    "mean_abs": 0.0036780002992600203,
    "rmse": 0.005155459977686405
  },
  "ddim_inversion_x2_prime_vs_sender_x2": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.8878087997436523,
    "mean_abs": 0.08994927257299423,
    "rmse": 0.11510290205478668
  },
  "zs_prime_vs_sender_zs": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 46.940460205078125,
    "mean_abs": 0.8173935413360596,
    "rmse": 2.3577866554260254
  },
  "zs_from_replayed_x2_and_inverted_x1_vs_sender_zs": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 1.2587952613830566,
    "mean_abs": 0.06708861142396927,
    "rmse": 0.094038225710392
  },
  "sender_zs_vs_sender_e_t": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 4.291534423828125e-06,
    "mean_abs": 5.194357299842522e-07,
    "rmse": 8.0040058492159e-07
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
  }
}
Saved traced tensors to outputsddim_reversibility.pt

AudioLDM2 DDIM reversibility report -100
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
    "max_abs": 0.03838180750608444,
    "mean_abs": 0.0035484852269291878,
    "rmse": 0.004564377013593912
  },
  "ddim_inversion_x2_prime_vs_sender_x2": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.414145290851593,
    "mean_abs": 0.050311993807554245,
    "rmse": 0.06380337476730347
  },
  "zs_prime_vs_sender_zs": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 23.56169891357422,
    "mean_abs": 0.4605868458747864,
    "rmse": 1.1473493576049805
  },
  "zs_from_replayed_x2_and_inverted_x1_vs_sender_zs": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 0.700103759765625,
    "mean_abs": 0.06472618132829666,
    "rmse": 0.08325657248497009
  },
  "sender_zs_vs_sender_e_t": {
    "shape": [
      1,
      8,
      256,
      16
    ],
    "max_abs": 4.410743713378906e-06,
    "mean_abs": 5.142720738149364e-07,
    "rmse": 7.921976816760434e-07
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
  }
} """

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
            },
            args.save_pt,
        )
        print(f"Saved traced tensors to {args.save_pt}")


if __name__ == "__main__":
    main()
