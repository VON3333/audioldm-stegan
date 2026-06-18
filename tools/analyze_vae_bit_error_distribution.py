""" 
python analyze_vae_bit_error_distribution.py \
  --vae_pt outputs_v7/ablate_vae_10s.pt \
  --input_pt outputs_v7/ddim_bits_1b_10s.pt \
  --out_dir outputs_v7/error_distribution_no_repeat \
  --export_bits \
  --save_heatmaps \
  --bit_export_limit 1000
 """
import argparse
import csv
import json
import os
import sys

import torch


def save_heatmap(matrix, path, title, xlabel, ylabel, vmin=0.0, vmax=1.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.figure(figsize=(10, 4.8))
    im = plt.imshow(matrix, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="BER")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def build_heatmap_matrices(error_3d, time_block, freq_block):
    c, t, f = error_3d.shape
    channel_time = error_3d.float().mean(dim=2).numpy()
    channel_freq = error_3d.float().mean(dim=1).numpy()
    time_freq = error_3d.float().mean(dim=0).numpy().T

    tb = (t + time_block - 1) // time_block
    fb = (f + freq_block - 1) // freq_block
    block_map = torch.zeros((c * fb, tb), dtype=torch.float32)
    for ci in range(c):
        for ti, ts in enumerate(range(0, t, time_block)):
            te = min(ts + time_block, t)
            for fi, fs in enumerate(range(0, f, freq_block)):
                fe = min(fs + freq_block, f)
                block_map[ci * fb + fi, ti] = error_3d[ci, ts:te, fs:fe].float().mean()

    return {
        "channel_time": channel_time,
        "channel_freq": channel_freq,
        "time_freq": time_freq,
        "block_map": block_map.numpy(),
    }


def load_pair(vae_pt, input_pt=None):
    vae_data = torch.load(vae_pt, map_location="cpu")
    if input_pt is None:
        input_pt = vae_data.get("report", {}).get("settings", {}).get("input_pt")
    if not input_pt:
        raise ValueError("input_pt not provided and not found in VAE report settings.")

    input_data = torch.load(input_pt, map_location="cpu")
    bit_data = input_data.get("bit_embedding_experiment")
    if bit_data is None:
        raise ValueError("Input .pt does not contain bit_embedding_experiment.")
    if "bits" not in bit_data:
        raise ValueError("Input .pt does not contain original bits.")
    if "decoded_bits" not in vae_data:
        raise ValueError("VAE .pt does not contain decoded_bits.")
    if "vae_recovered_zs" not in vae_data:
        raise ValueError("VAE .pt does not contain vae_recovered_zs.")

    bits = bit_data["bits"].long()
    decoded_bits = vae_data["decoded_bits"].long()
    z = vae_data["vae_recovered_zs"].float()
    embedded_z = bit_data.get("embedded_zs")
    if embedded_z is not None:
        embedded_z = embedded_z.float()
    return input_pt, bits, decoded_bits, z, embedded_z


def basic_metrics(error_mask):
    total = int(error_mask.numel())
    errors = int(error_mask.sum().item())
    return {
        "total_bits": total,
        "bit_errors": errors,
        "bit_error_rate": errors / total,
        "bit_accuracy": 1.0 - errors / total,
    }


def summarize_axis(error_3d, axis_name, axis):
    # error_3d shape: [C, T, F]
    dims = [0, 1, 2]
    reduce_dims = [d for d in dims if d != axis]
    counts = error_3d.sum(dim=reduce_dims)
    totals = torch.ones_like(error_3d).sum(dim=reduce_dims)
    rates = counts.float() / totals.float()
    rows = []
    for i in range(counts.numel()):
        rows.append(
            {
                axis_name: int(i),
                "total": int(totals[i].item()),
                "errors": int(counts[i].item()),
                "ber": float(rates[i].item()),
            }
        )
    return rows


def block_summary(error_3d, time_block, freq_block):
    c, t, f = error_3d.shape
    rows = []
    for ci in range(c):
        for ts in range(0, t, time_block):
            te = min(ts + time_block, t)
            for fs in range(0, f, freq_block):
                fe = min(fs + freq_block, f)
                block = error_3d[ci, ts:te, fs:fe]
                total = int(block.numel())
                errors = int(block.sum().item())
                rows.append(
                    {
                        "channel": ci,
                        "time_start": ts,
                        "time_end": te,
                        "freq_start": fs,
                        "freq_end": fe,
                        "total": total,
                        "errors": errors,
                        "ber": errors / total,
                    }
                )
    rows.sort(key=lambda x: x["ber"], reverse=True)
    return rows


def concentration_metrics(error_3d, group_rows):
    total_errors = sum(row["errors"] for row in group_rows)
    sorted_errors = sorted([row["errors"] for row in group_rows], reverse=True)
    out = {
        "num_groups": len(group_rows),
        "total_errors": total_errors,
    }
    if total_errors == 0:
        out.update(
            {
                "top_1pct_error_share": 0.0,
                "top_5pct_error_share": 0.0,
                "top_10pct_error_share": 0.0,
                "groups_for_50pct_errors": 0,
                "groups_for_80pct_errors": 0,
            }
        )
        return out

    for pct in [0.01, 0.05, 0.10]:
        k = max(1, int(round(len(sorted_errors) * pct)))
        out[f"top_{int(pct*100)}pct_error_share"] = sum(sorted_errors[:k]) / total_errors

    cumulative = 0
    groups_50 = 0
    groups_80 = 0
    for i, val in enumerate(sorted_errors, 1):
        cumulative += val
        if groups_50 == 0 and cumulative >= 0.5 * total_errors:
            groups_50 = i
        if groups_80 == 0 and cumulative >= 0.8 * total_errors:
            groups_80 = i
            break
    out["groups_for_50pct_errors"] = groups_50
    out["groups_for_80pct_errors"] = groups_80
    return out


def bit_value_metrics(bits, decoded_bits):
    b = bits.reshape(-1)
    d = decoded_bits.reshape(-1)
    rows = {}
    for bit_value in [0, 1]:
        mask = b == bit_value
        total = int(mask.sum().item())
        errors = int((d[mask] != b[mask]).sum().item())
        rows[f"bit{bit_value}"] = {
            "total": total,
            "errors": errors,
            "ber": errors / total if total else 0.0,
        }
    return rows


def z_stats_by_error(z, embedded_z, error_mask):
    z_flat = z.reshape(-1)
    err = error_mask.reshape(-1).bool()
    ok = ~err
    out = {
        "recovered_z_correct_mean": float(z_flat[ok].mean().item()),
        "recovered_z_correct_std": float(z_flat[ok].std(unbiased=False).item()),
        "recovered_z_error_mean": float(z_flat[err].mean().item()),
        "recovered_z_error_std": float(z_flat[err].std(unbiased=False).item()),
        "abs_recovered_z_correct_mean": float(z_flat[ok].abs().mean().item()),
        "abs_recovered_z_error_mean": float(z_flat[err].abs().mean().item()),
    }
    if embedded_z is not None:
        diff = (z.reshape(-1) - embedded_z.reshape(-1)).abs()
        out.update(
            {
                "abs_z_diff_correct_mean": float(diff[ok].mean().item()),
                "abs_z_diff_error_mean": float(diff[err].mean().item()),
            }
        )
    return out


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_bit_comparison_rows(bits, decoded_bits, z, embedded_z, limit=None):
    flat_bits = bits.reshape(-1).long()
    flat_decoded = decoded_bits.reshape(-1).long()
    flat_z = z.reshape(-1).float()
    flat_embedded = embedded_z.reshape(-1).float() if embedded_z is not None else None

    _, c, t, f = z.shape
    total = flat_bits.numel()
    n = total if limit is None or limit <= 0 else min(total, limit)
    rows = []
    for idx in range(n):
        channel = idx // (t * f)
        rem = idx % (t * f)
        time = rem // f
        freq = rem % f
        bit = int(flat_bits[idx].item())
        decoded = int(flat_decoded[idx].item())
        row = {
            "flat_index": idx,
            "channel": channel,
            "time": time,
            "freq": freq,
            "embedded_bit": bit,
            "decoded_bit": decoded,
            "error": int(bit != decoded),
            "recovered_z": float(flat_z[idx].item()),
        }
        if flat_embedded is not None:
            row["embedded_z"] = float(flat_embedded[idx].item())
            row["abs_z_diff"] = abs(row["recovered_z"] - row["embedded_z"])
        rows.append(row)
    return rows


def make_error_only_rows(bits, decoded_bits, z, embedded_z, limit=None):
    rows = make_bit_comparison_rows(bits, decoded_bits, z, embedded_z, limit=None)
    errors = [row for row in rows if row["error"] == 1]
    if limit is not None and limit > 0:
        errors = errors[:limit]
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Analyze whether VAE-channel bit errors are concentrated or dispersed."
    )
    parser.add_argument("--vae_pt", required=True, help="VAE ablation .pt with decoded_bits.")
    parser.add_argument("--input_pt", default=None, help="Original embedding .pt. Defaults to report settings input_pt.")
    parser.add_argument("--time_block", type=int, default=16)
    parser.add_argument("--freq_block", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--out_dir", default="outputs/error_distribution")
    parser.add_argument(
        "--save_heatmaps",
        action="store_true",
        help="Save PNG heatmaps for error distribution.",
    )
    parser.add_argument(
        "--export_bits",
        action="store_true",
        help="Export per-position embedded/decoded bit comparison CSVs.",
    )
    parser.add_argument(
        "--bit_export_limit",
        type=int,
        default=0,
        help="Limit rows in bit_comparison.csv; 0 means export all positions.",
    )
    args = parser.parse_args()

    input_pt, bits, decoded_bits, z, embedded_z = load_pair(args.vae_pt, args.input_pt)
    if bits.shape[1] != 1:
        raise ValueError("This analyzer currently expects bits_per_z=1.")
    if list(z.shape) != [1, 8, 256, 16]:
        print(f"Warning: expected z shape [1,8,256,16], got {list(z.shape)}")

    error_flat = (bits.reshape(-1) != decoded_bits.reshape(-1)).long()
    error_3d = error_flat.reshape(z.shape[1], z.shape[2], z.shape[3])

    by_channel = summarize_axis(error_3d, "channel", 0)
    by_time = summarize_axis(error_3d, "time", 1)
    by_freq = summarize_axis(error_3d, "freq", 2)
    by_block = block_summary(error_3d, args.time_block, args.freq_block)

    report = {
        "settings": {
            "vae_pt": args.vae_pt,
            "input_pt": input_pt,
            "time_block": args.time_block,
            "freq_block": args.freq_block,
            "shape": list(z.shape),
        },
        "overall": basic_metrics(error_flat),
        "bit_value_metrics": bit_value_metrics(bits, decoded_bits),
        "concentration": concentration_metrics(error_3d, by_block),
        "z_stats_by_error": z_stats_by_error(z, embedded_z, error_flat),
        "by_channel": by_channel,
        "by_freq": by_freq,
        "top_error_blocks": by_block[: args.top_k],
        "lowest_error_blocks": list(reversed(by_block[-args.top_k :])),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "vae_bit_error_distribution_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    write_csv(os.path.join(args.out_dir, "by_channel.csv"), by_channel)
    write_csv(os.path.join(args.out_dir, "by_time.csv"), by_time)
    write_csv(os.path.join(args.out_dir, "by_freq.csv"), by_freq)
    write_csv(os.path.join(args.out_dir, "by_block.csv"), by_block)
    if args.export_bits:
        write_csv(
            os.path.join(args.out_dir, "bit_comparison.csv"),
            make_bit_comparison_rows(
                bits, decoded_bits, z, embedded_z, limit=args.bit_export_limit
            ),
        )
        write_csv(
            os.path.join(args.out_dir, "bit_errors_only.csv"),
            make_error_only_rows(
                bits, decoded_bits, z, embedded_z, limit=args.bit_export_limit
            ),
        )

    if args.save_heatmaps:
        heatmaps = build_heatmap_matrices(error_3d, args.time_block, args.freq_block)
        save_heatmap(
            heatmaps["channel_time"],
            os.path.join(args.out_dir, "heatmap_channel_time.png"),
            "Bit Error Rate by Channel and Time",
            "time index",
            "channel",
            vmin=0.0,
            vmax=1.0,
        )
        save_heatmap(
            heatmaps["channel_freq"],
            os.path.join(args.out_dir, "heatmap_channel_freq.png"),
            "Bit Error Rate by Channel and Frequency",
            "frequency index",
            "channel",
            vmin=0.0,
            vmax=1.0,
        )
        save_heatmap(
            heatmaps["time_freq"],
            os.path.join(args.out_dir, "heatmap_time_freq.png"),
            "Bit Error Rate by Time and Frequency",
            "time index",
            "frequency index",
            vmin=0.0,
            vmax=1.0,
        )
        save_heatmap(
            heatmaps["block_map"],
            os.path.join(args.out_dir, "heatmap_group_blocks.png"),
            f"Group BER: channel x freq-block rows, time-block columns",
            "time block",
            "channel/freq-block",
            vmin=0.0,
            vmax=1.0,
        )

    print("VAE bit error distribution report")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report and CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
