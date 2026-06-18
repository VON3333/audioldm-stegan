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


def masked_mean(values, mask, dims):
    values = values.float()
    mask = mask.float()
    counts = mask.sum(dim=dims)
    sums = (values * mask).sum(dim=dims)
    out = sums / counts.clamp_min(1.0)
    return out.masked_fill(counts == 0, float("nan"))


def build_heatmap_matrices(error_3d, valid_3d, time_block, freq_block):
    c, t, f = error_3d.shape
    channel_time = masked_mean(error_3d, valid_3d, dims=2).numpy()
    channel_freq = masked_mean(error_3d, valid_3d, dims=1).numpy()
    time_freq = masked_mean(error_3d, valid_3d, dims=0).numpy().T

    tb = (t + time_block - 1) // time_block
    fb = (f + freq_block - 1) // freq_block
    block_map = torch.full((c * fb, tb), float("nan"), dtype=torch.float32)
    for ci in range(c):
        for ti, ts in enumerate(range(0, t, time_block)):
            te = min(ts + time_block, t)
            for fi, fs in enumerate(range(0, f, freq_block)):
                fe = min(fs + freq_block, f)
                block_errors = error_3d[ci, ts:te, fs:fe]
                block_valid = valid_3d[ci, ts:te, fs:fe]
                total = int(block_valid.sum().item())
                if total:
                    block_map[ci * fb + fi, ti] = block_errors[block_valid].float().mean()

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

    bits = bit_data["bits"].long().reshape(-1)
    decoded_bits = vae_data["decoded_bits"].long().reshape(-1)
    z = vae_data["vae_recovered_zs"].float()
    embedded_z = bit_data.get("embedded_zs")
    if embedded_z is not None:
        embedded_z = embedded_z.float()
    used_positions = bit_data.get("used_positions")
    if used_positions is None:
        if bits.numel() != z.numel():
            raise ValueError(
                "Input .pt does not contain used_positions, and bits length does not match full latent size."
            )
        used_positions = torch.arange(bits.numel(), dtype=torch.long)
    else:
        used_positions = used_positions.long().reshape(-1)
    if bits.numel() != used_positions.numel():
        raise ValueError(
            f"bits length ({bits.numel()}) does not match used_positions length ({used_positions.numel()})."
        )
    if decoded_bits.numel() != bits.numel():
        raise ValueError(
            f"decoded_bits length ({decoded_bits.numel()}) does not match bits length ({bits.numel()})."
        )
    exclude_channels = bit_data.get("exclude_channels")
    if exclude_channels is not None:
        exclude_channels = [int(x) for x in exclude_channels.reshape(-1).tolist()]
    position_metadata = bit_data.get("position_metadata")
    if not isinstance(position_metadata, dict):
        position_metadata = vae_data.get("position_metadata", {})
    source_settings = input_data.get("report", {}).get("settings", {})
    return (
        input_pt,
        bits,
        decoded_bits,
        z,
        embedded_z,
        used_positions,
        exclude_channels,
        position_metadata,
        source_settings,
    )


def basic_metrics(error_mask):
    total = int(error_mask.numel())
    errors = int(error_mask.sum().item())
    return {
        "total_bits": total,
        "bit_errors": errors,
        "bit_error_rate": errors / total,
        "bit_accuracy": 1.0 - errors / total,
    }


def summarize_axis(error_3d, valid_3d, axis_name, axis):
    # error_3d shape: [C, T, F]
    dims = [0, 1, 2]
    reduce_dims = [d for d in dims if d != axis]
    counts = (error_3d.long() * valid_3d.long()).sum(dim=reduce_dims)
    totals = valid_3d.long().sum(dim=reduce_dims)
    rates = counts.float() / totals.float().clamp_min(1.0)
    rows = []
    for i in range(counts.numel()):
        total = int(totals[i].item())
        rows.append(
            {
                axis_name: int(i),
                "total": total,
                "errors": int(counts[i].item()),
                "ber": float(rates[i].item()) if total else None,
            }
        )
    return rows


def block_summary(error_3d, valid_3d, time_block, freq_block):
    c, t, f = error_3d.shape
    rows = []
    for ci in range(c):
        for ts in range(0, t, time_block):
            te = min(ts + time_block, t)
            for fs in range(0, f, freq_block):
                fe = min(fs + freq_block, f)
                block = error_3d[ci, ts:te, fs:fe]
                valid = valid_3d[ci, ts:te, fs:fe]
                total = int(valid.sum().item())
                errors = int(block[valid].sum().item()) if total else 0
                rows.append(
                    {
                        "channel": ci,
                        "time_start": ts,
                        "time_end": te,
                        "freq_start": fs,
                        "freq_end": fe,
                        "total": total,
                        "errors": errors,
                        "ber": errors / total if total else None,
                    }
                )
    rows.sort(key=lambda x: -1.0 if x["ber"] is None else x["ber"], reverse=True)
    return rows


def add_time_seconds_to_rows(rows, duration, time_steps):
    if duration is None:
        return rows
    for row in rows:
        if "time" in row:
            row["time_sec_start"] = float(row["time"] / time_steps * duration)
            row["time_sec_end"] = float((row["time"] + 1) / time_steps * duration)
        if "time_start" in row and "time_end" in row:
            row["time_sec_start"] = float(row["time_start"] / time_steps * duration)
            row["time_sec_end"] = float(row["time_end"] / time_steps * duration)
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


def z_stats_by_error(z, embedded_z, error_mask, used_positions):
    z_flat = z.reshape(-1)[used_positions]
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
        diff = (z.reshape(-1)[used_positions] - embedded_z.reshape(-1)[used_positions]).abs()
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


def metadata_to_plain_dict(value):
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, item in value.items():
        if torch.is_tensor(item):
            out[key] = item.item() if item.numel() == 1 else item.detach().cpu().tolist()
        else:
            out[key] = item
    return out


def make_bit_comparison_rows(
    bits,
    decoded_bits,
    z,
    embedded_z,
    used_positions,
    duration=None,
    limit=None,
):
    flat_bits = bits.reshape(-1).long()
    flat_decoded = decoded_bits.reshape(-1).long()
    flat_z = z.reshape(-1).float()
    flat_embedded = embedded_z.reshape(-1).float() if embedded_z is not None else None

    _, c, t, f = z.shape
    total = flat_bits.numel()
    n = total if limit is None or limit <= 0 else min(total, limit)
    rows = []
    for message_index in range(n):
        idx = int(used_positions[message_index].item())
        channel = idx // (t * f)
        rem = idx % (t * f)
        time = rem // f
        freq = rem % f
        bit = int(flat_bits[message_index].item())
        decoded = int(flat_decoded[message_index].item())
        row = {
            "flat_index": idx,
            "message_index": message_index,
            "channel": channel,
            "time": time,
            "freq": freq,
            "embedded_bit": bit,
            "decoded_bit": decoded,
            "error": int(bit != decoded),
            "recovered_z": float(flat_z[idx].item()),
        }
        if duration is not None:
            row["time_sec_start"] = float(time / t * duration)
            row["time_sec_end"] = float((time + 1) / t * duration)
        if flat_embedded is not None:
            row["embedded_z"] = float(flat_embedded[idx].item())
            row["abs_z_diff"] = abs(row["recovered_z"] - row["embedded_z"])
        rows.append(row)
    return rows


def make_error_only_rows(
    bits,
    decoded_bits,
    z,
    embedded_z,
    used_positions,
    duration=None,
    limit=None,
):
    rows = make_bit_comparison_rows(
        bits,
        decoded_bits,
        z,
        embedded_z,
        used_positions,
        duration=duration,
        limit=None,
    )
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

    (
        input_pt,
        bits,
        decoded_bits,
        z,
        embedded_z,
        used_positions,
        exclude_channels,
        position_metadata,
        source_settings,
    ) = load_pair(args.vae_pt, args.input_pt)
    position_metadata = metadata_to_plain_dict(position_metadata)
    duration = source_settings.get("duration")
    duration = float(duration) if duration is not None else None
    if list(z.shape) != [1, 8, 256, 16]:
        print(f"Warning: expected z shape [1,8,256,16], got {list(z.shape)}")

    error_flat = (bits.reshape(-1) != decoded_bits.reshape(-1)).long()
    full_error_flat = torch.zeros(z.numel(), dtype=torch.long)
    full_valid_flat = torch.zeros(z.numel(), dtype=torch.bool)
    full_error_flat[used_positions] = error_flat
    full_valid_flat[used_positions] = True
    error_3d = full_error_flat.reshape(z.shape[1], z.shape[2], z.shape[3])
    valid_3d = full_valid_flat.reshape(z.shape[1], z.shape[2], z.shape[3])

    by_channel = summarize_axis(error_3d, valid_3d, "channel", 0)
    by_time = summarize_axis(error_3d, valid_3d, "time", 1)
    by_freq = summarize_axis(error_3d, valid_3d, "freq", 2)
    by_block = block_summary(error_3d, valid_3d, args.time_block, args.freq_block)
    by_time = add_time_seconds_to_rows(by_time, duration, z.shape[2])
    by_block = add_time_seconds_to_rows(by_block, duration, z.shape[2])
    valid_blocks = [row for row in by_block if row["total"] > 0]

    report = {
        "settings": {
            "vae_pt": args.vae_pt,
            "input_pt": input_pt,
            "time_block": args.time_block,
            "freq_block": args.freq_block,
            "shape": list(z.shape),
            "analyzed_positions": int(used_positions.numel()),
            "exclude_channels": exclude_channels or [],
            "duration": duration,
            **position_metadata,
        },
        "overall": basic_metrics(error_flat),
        "bit_value_metrics": bit_value_metrics(bits, decoded_bits),
        "concentration": concentration_metrics(error_3d, valid_blocks),
        "z_stats_by_error": z_stats_by_error(z, embedded_z, error_flat, used_positions),
        "by_channel": by_channel,
        "by_freq": by_freq,
        "top_error_blocks": valid_blocks[: args.top_k],
        "lowest_error_blocks": sorted(valid_blocks, key=lambda x: x["ber"])[: args.top_k],
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
                bits,
                decoded_bits,
                z,
                embedded_z,
                used_positions,
                duration=duration,
                limit=args.bit_export_limit,
            ),
        )
        write_csv(
            os.path.join(args.out_dir, "bit_errors_only.csv"),
            make_error_only_rows(
                bits,
                decoded_bits,
                z,
                embedded_z,
                used_positions,
                duration=duration,
                limit=args.bit_export_limit,
            ),
        )

    if args.save_heatmaps:
        heatmaps = build_heatmap_matrices(error_3d, valid_3d, args.time_block, args.freq_block)
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
