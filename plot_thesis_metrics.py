"""
Thesis Evaluation Graphs
Reads thesis_monitor_log.csv and generates publication-quality figures.

Usage:
    python plot_thesis_metrics.py                  # saves all figures to ./thesis_plots/
    python plot_thesis_metrics.py --show           # also opens interactive windows
"""

import argparse
import glob
import os
import struct
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Georgia"],
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

SCENE_COLORS = {
    "ficus":  "#2E86AB",   # blue
    "hotdog": "#E84855",   # red
    "lego":   "#F4A261",   # orange
}

STAGE_STYLES = {
    "vanilla": dict(linestyle="-",  linewidth=2.0, marker="o", markersize=3),
    "stage1":  dict(linestyle="--", linewidth=2.0, marker="s", markersize=3),
    "stage2":  dict(linestyle=":",  linewidth=2.0, marker="^", markersize=3),
    "unknown": dict(linestyle="-",  linewidth=1.5, alpha=0.5),
}

OUT_DIR = "./thesis_plots"
os.makedirs(OUT_DIR, exist_ok=True)


def load(csv_path="thesis_monitor_log.csv"):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["elapsed_hr"] = df["elapsed_min"] / 60

    # Derive instantaneous network & disk rates (GB/min)
    for col in ["net_sent_gb", "net_recv_gb", "disk_read_gb", "disk_write_gb"]:
        df[f"{col}_rate"] = df[col].diff() / df["elapsed_min"].diff()
        df[f"{col}_rate"] = df[f"{col}_rate"].clip(lower=0)

    return df


def mark_stage_transitions(ax, df, scene):
    """Draw vertical lines where the stage changes for a scene."""
    stage_col = f"{scene}_stage"
    if stage_col not in df.columns:
        return
    prev = None
    for _, row in df.iterrows():
        s = row[stage_col]
        if s != prev and prev is not None and s in ("stage1", "stage2"):
            ax.axvline(row["elapsed_hr"], color="grey", linestyle=":", linewidth=1, alpha=0.6)
            ax.text(row["elapsed_hr"] + 0.02, ax.get_ylim()[1] * 0.97,
                    s.replace("stage", "S"), fontsize=7, color="grey", va="top")
        prev = s


# ── Figure 1: GPU Overview (2×2) ──────────────────────────────────────────────
def fig_gpu_overview(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    fig.suptitle("GPU Performance — NVIDIA H200 (143 GB HBM3)", fontsize=13, fontweight="bold")

    x = df["elapsed_hr"]

    # Utilisation
    ax = axes[0, 0]
    ax.fill_between(x, df["gpu_util_pct"], alpha=0.2, color="#2E86AB")
    ax.plot(x, df["gpu_util_pct"], color="#2E86AB", linewidth=1.5)
    ax.set_ylabel("GPU Utilisation (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))

    # VRAM
    ax = axes[0, 1]
    ax.fill_between(x, df["gpu_mem_used_mb"] / 1024, alpha=0.2, color="#E84855")
    ax.plot(x, df["gpu_mem_used_mb"] / 1024, color="#E84855", linewidth=1.5, label="Used")
    ax.axhline(df["gpu_mem_total_mb"].iloc[0] / 1024, color="grey",
               linestyle="--", linewidth=1, label=f"Total ({df['gpu_mem_total_mb'].iloc[0]/1024:.0f} GB)")
    ax.set_ylabel("VRAM (GB)")
    ax.legend(loc="upper right")

    # Temperature
    ax = axes[1, 0]
    ax.plot(x, df["gpu_temp_c"], color="#F4A261", linewidth=1.5)
    ax.fill_between(x, df["gpu_temp_c"], alpha=0.15, color="#F4A261")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xlabel("Elapsed Time (hours)")
    ax.axhline(80, color="red", linestyle=":", linewidth=1, alpha=0.6, label="80°C caution")
    ax.legend(loc="upper right")

    # Power Draw
    ax = axes[1, 1]
    ax.plot(x, df["gpu_power_w"], color="#6A0572", linewidth=1.5, label="Draw")
    ax.axhline(df["gpu_power_limit_w"].iloc[0], color="red",
               linestyle="--", linewidth=1, label=f"Limit ({df['gpu_power_limit_w'].iloc[0]:.0f} W)")
    ax.fill_between(x, df["gpu_power_w"], alpha=0.15, color="#6A0572")
    ax.set_ylabel("Power Draw (W)")
    ax.set_xlabel("Elapsed Time (hours)")
    ax.legend(loc="upper right")

    for ax in axes.flat:
        ax.set_xlim(left=0)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig1_gpu_overview.pdf"
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"  Saved: {path}")
    return fig


# ── Figure 2: CPU & RAM ───────────────────────────────────────────────────────
def fig_cpu_ram(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    fig.suptitle("CPU and RAM Utilisation During Training", fontsize=13, fontweight="bold")

    x = df["elapsed_hr"]

    ax1.fill_between(x, df["cpu_util_pct"], alpha=0.2, color="#2E86AB")
    ax1.plot(x, df["cpu_util_pct"], color="#2E86AB", linewidth=1.5)
    ax1.set_ylabel("CPU Utilisation (%)")
    ax1.set_xlabel("Elapsed Time (hours)")
    ax1.set_ylim(0, 105)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))

    ax2.fill_between(x, df["ram_used_gb"], alpha=0.2, color="#E84855")
    ax2.plot(x, df["ram_used_gb"], color="#E84855", linewidth=1.5, label="Used")
    ax2.axhline(df["ram_total_gb"].iloc[0], color="grey", linestyle="--",
                linewidth=1, label=f"Total ({df['ram_total_gb'].iloc[0]:.0f} GB)")
    ax2.set_ylabel("RAM (GB)")
    ax2.set_xlabel("Elapsed Time (hours)")
    ax2.legend()

    for ax in (ax1, ax2):
        ax.set_xlim(left=0)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig2_cpu_ram.pdf"
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"  Saved: {path}")
    return fig


# ── Figure 3: Network & Disk I/O rates ───────────────────────────────────────
def fig_io(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    fig.suptitle("I/O Throughput During Training", fontsize=13, fontweight="bold")

    x = df["elapsed_hr"]

    ax = axes[0]
    ax.plot(x, df["net_recv_gb_rate"], color="#2E86AB", linewidth=1.5, label="Download (↓)")
    ax.plot(x, df["net_sent_gb_rate"], color="#E84855", linewidth=1.5, label="Upload (↑)")
    ax.fill_between(x, df["net_recv_gb_rate"], alpha=0.15, color="#2E86AB")
    ax.set_ylabel("Network Rate (GB/min)")
    ax.set_xlabel("Elapsed Time (hours)")
    ax.set_title("Network I/O")
    ax.legend()

    ax = axes[1]
    ax.plot(x, df["disk_read_gb_rate"],  color="#F4A261", linewidth=1.5, label="Read")
    ax.plot(x, df["disk_write_gb_rate"], color="#6A0572", linewidth=1.5, label="Write")
    ax.fill_between(x, df["disk_write_gb_rate"], alpha=0.15, color="#6A0572")
    ax.set_ylabel("Disk Rate (GB/min)")
    ax.set_xlabel("Elapsed Time (hours)")
    ax.set_title("Disk I/O")
    ax.legend()

    for ax in axes:
        ax.set_xlim(left=0)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig3_io_throughput.pdf"
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"  Saved: {path}")
    return fig


# ── Figure 4: Training PSNR convergence per scene ────────────────────────────
def fig_psnr_convergence(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle("Training PSNR Convergence by Scene and Stage", fontsize=13, fontweight="bold")

    for ax, scene in zip(axes, ["ficus", "hotdog", "lego"]):
        color = SCENE_COLORS[scene]
        stage_col  = f"{scene}_stage"
        iter_col   = f"{scene}_iter"
        psnr_col   = f"{scene}_psnr_train"

        sub = df[(df[iter_col] > 0) & (df[psnr_col] > 0)].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(scene.capitalize())
            continue

        for stage, grp in sub.groupby(stage_col):
            style = STAGE_STYLES.get(stage, STAGE_STYLES["unknown"])
            ax.plot(grp[iter_col], grp[psnr_col],
                    color=color, label=stage, **style,
                    markevery=max(1, len(grp)//10))

        ax.set_title(scene.capitalize(), fontweight="bold")
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Train PSNR (dB)")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
        ax.legend(title="Stage")

    fig.tight_layout()
    path = f"{OUT_DIR}/fig4_psnr_convergence.pdf"
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"  Saved: {path}")
    return fig


# ── Figure 5: VRAM usage coloured by active stage ────────────────────────────
def fig_vram_stages(df):
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.suptitle("VRAM Usage Across All Training Stages", fontsize=13, fontweight="bold")

    x  = df["elapsed_hr"]
    y  = df["gpu_mem_used_mb"] / 1024

    ax.fill_between(x, y, alpha=0.15, color="steelblue")
    ax.plot(x, y, color="steelblue", linewidth=1.5)
    ax.axhline(df["gpu_mem_total_mb"].iloc[0] / 1024, color="red",
               linestyle="--", linewidth=1, alpha=0.7,
               label=f"H200 total: {df['gpu_mem_total_mb'].iloc[0]/1024:.0f} GB")

    # Annotate active run count transitions
    prev_n = None
    for _, row in df.iterrows():
        n = row["active_runs"]
        if n != prev_n and prev_n is not None:
            ax.axvline(row["elapsed_hr"], color="grey", linestyle=":", linewidth=1, alpha=0.5)
            ax.text(row["elapsed_hr"] + 0.02,
                    df["gpu_mem_total_mb"].iloc[0] / 1024 * 0.95,
                    f"{int(n)} proc", fontsize=7, color="grey", va="top")
        prev_n = n

    ax.set_ylabel("VRAM Used (GB)")
    ax.set_xlabel("Elapsed Time (hours)")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    path = f"{OUT_DIR}/fig5_vram_stages.pdf"
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"  Saved: {path}")
    return fig


# ── Figure 6: Summary dashboard ───────────────────────────────────────────────
def fig_dashboard(df):
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Training Run Dashboard — H200 GPU, TensoIR NeP Experiments",
                 fontsize=13, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    x = df["elapsed_hr"]

    panels = [
        (gs[0, 0], "GPU Util (%)",    df["gpu_util_pct"],         "#2E86AB", (0, 105)),
        (gs[0, 1], "VRAM Used (GB)",  df["gpu_mem_used_mb"]/1024, "#E84855", None),
        (gs[0, 2], "Power Draw (W)",  df["gpu_power_w"],           "#6A0572", None),
        (gs[1, 0], "GPU Temp (°C)",   df["gpu_temp_c"],            "#F4A261", None),
        (gs[1, 1], "CPU Util (%)",    df["cpu_util_pct"],          "#2E86AB", (0, 105)),
        (gs[1, 2], "RAM Used (GB)",   df["ram_used_gb"],           "#E84855", None),
        (gs[2, 0], "Net Recv (GB/min)", df["net_recv_gb_rate"],    "#2E86AB", None),
        (gs[2, 1], "Disk Write (GB/min)", df["disk_write_gb_rate"],"#6A0572", None),
    ]

    for spec, label, series, color, ylim in panels:
        ax = fig.add_subplot(spec)
        ax.fill_between(x, series, alpha=0.2, color=color)
        ax.plot(x, series, color=color, linewidth=1.2)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("hrs", fontsize=8)
        ax.set_xlim(left=0)
        if ylim:
            ax.set_ylim(*ylim)

    # Bottom right: PSNR all scenes
    ax = fig.add_subplot(gs[2, 2])
    for scene in ["ficus", "hotdog", "lego"]:
        sub = df[(df[f"{scene}_iter"] > 0) & (df[f"{scene}_psnr_train"] > 0)]
        if not sub.empty:
            ax.plot(sub[f"{scene}_iter"], sub[f"{scene}_psnr_train"],
                    color=SCENE_COLORS[scene], linewidth=1.2, label=scene.capitalize())
    ax.set_title("Train PSNR", fontsize=9)
    ax.set_xlabel("Iteration", fontsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.legend(fontsize=7)

    path = f"{OUT_DIR}/fig6_dashboard.pdf"
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"  Saved: {path}")
    return fig


# ── TFEvents helpers ──────────────────────────────────────────────────────────

def _parse_varint(buf, pos):
    result, shift = 0, 0
    while pos < len(buf):
        b = buf[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _extract_scalars(data):
    """Yield (step, tag, value) tuples from a single TFRecord event blob."""
    pos = 0
    step = 0
    while pos < len(data):
        if pos >= len(data):
            break
        b = data[pos]; pos += 1
        fn = b >> 3; wt = b & 7
        if fn == 2 and wt == 0:                     # step (varint)
            step, pos = _parse_varint(data, pos)
        elif fn == 5 and wt == 2:                   # summary message
            length, pos = _parse_varint(data, pos)
            summary = data[pos:pos + length]; pos += length
            spos = 0
            while spos < len(summary):
                sb = summary[spos]; spos += 1
                sf = sb >> 3; swt = sb & 7
                if sf == 1 and swt == 2:            # Value sub-message
                    vlen, spos = _parse_varint(summary, spos)
                    vd = summary[spos:spos + vlen]; spos += vlen
                    tag, value = b"", None
                    vpos = 0
                    while vpos < len(vd):
                        vb = vd[vpos]; vpos += 1
                        vf = vb >> 3; vwt = vb & 7
                        if vf == 1 and vwt == 2:    # tag string
                            tlen, vpos = _parse_varint(vd, vpos)
                            tag = vd[vpos:vpos + tlen]; vpos += tlen
                        elif vf == 2 and vwt == 5:  # simple_value float
                            value = struct.unpack("<f", vd[vpos:vpos + 4])[0]
                            vpos += 4
                        else:
                            break
                    if tag and value is not None:
                        yield step, tag.decode("utf-8", errors="replace"), value
                else:
                    break
        elif wt == 0:
            _, pos = _parse_varint(data, pos)
        elif wt == 2:
            length, pos = _parse_varint(data, pos); pos += length
        elif wt == 5:
            pos += 4
        elif wt == 1:
            pos += 8
        else:
            break


def load_tfevents(path, tags=None, every_n=100):
    """
    Parse a TFEvents file and return {tag: [(step, value), ...]} for
    the requested tags (all tags if tags=None).  Dense series are
    downsampled to every every_n-th point to keep plots readable.
    """
    data = {}
    with open(path, "rb") as f:
        while True:
            hdr = f.read(12)
            if len(hdr) < 12:
                break
            length = struct.unpack("<Q", hdr[:8])[0]
            blob = f.read(length)
            f.read(4)
            if len(blob) < length:
                break
            for step, tag, val in _extract_scalars(blob):
                if tags is None or tag in tags:
                    data.setdefault(tag, []).append((step, val))

    # Downsample dense series
    out = {}
    for tag, pts in data.items():
        pts.sort(key=lambda x: x[0])
        if len(pts) > every_n * 2:
            pts = pts[::every_n]
        out[tag] = pts
    return out


LOG_DIR = "/root/TensoIR/log/log_rotated_multi_lights"
RUNS    = ["vanilla_v2", "unified_v6"]
SCENES  = ["ficus", "hotdog", "lego"]

RUN_STYLES = {
    "vanilla_v2": dict(linestyle="-",  linewidth=1.8, alpha=0.9),
    "unified_v6": dict(linestyle="--", linewidth=1.8, alpha=0.9),
}
RUN_LABELS = {
    "vanilla_v2": "Vanilla v2",
    "unified_v6": "Unified v6",
}


def load_all_runs(wanted_tags, every_n=100):
    """
    Returns data[run][scene] = {tag: [(step, val), ...]}
    Skips silently when a log directory is missing.
    """
    data = {r: {s: {} for s in SCENES} for r in RUNS}
    for run in RUNS:
        for scene in SCENES:
            pattern = os.path.join(LOG_DIR, f"{scene}_{run}-*", "events.out.*")
            matches = sorted(glob.glob(pattern))
            if not matches:
                continue
            tf_path = matches[-1]  # take the latest if multiple
            data[run][scene] = load_tfevents(tf_path, tags=wanted_tags, every_n=every_n)
    return data


def _savefig(fig, path):
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    print(f"  Saved: {path}")


def _kfmt(v, _):
    return f"{v/1000:.0f}k"


# ── Figure 7: PSNR convergence v2 vs v6 (3 scenes × 2 metrics) ───────────────
def fig_psnr_v2_vs_v6(data):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=False)
    fig.suptitle("Training PSNR Convergence — Vanilla v2 vs Unified v6",
                 fontsize=13, fontweight="bold")

    metrics = [
        ("train/PSNRs_rgb",      "Train PSNR RGB (dB)"),
        ("train/PSNRs_rgb_brdf", "Train PSNR BRDF-RGB (dB)"),
    ]

    for row, scene in enumerate(SCENES):
        color = SCENE_COLORS[scene]
        for col, (tag, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for run in RUNS:
                pts = data[run][scene].get(tag, [])
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, color=color, label=RUN_LABELS[run], **RUN_STYLES[run])

            if col == 0:
                ax.set_ylabel(f"{scene.capitalize()}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("Training Iteration")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(_kfmt))
            ax.legend(fontsize=8)

    fig.tight_layout()
    _savefig(fig, f"{OUT_DIR}/fig7_psnr_v2_vs_v6.pdf")
    return fig


# ── Figure 8: Normal loss curves v2 vs v6 ────────────────────────────────────
def fig_normal_loss(data):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=False)
    fig.suptitle("Normal Loss Curves — Vanilla v2 vs Unified v6",
                 fontsize=13, fontweight="bold")

    metrics = [
        ("train/normals_diff_loss",        "Normals Diff Loss"),
        ("train/normals_orientation_loss",  "Normals Orientation Loss"),
    ]
    # BRDF-stage onset iterations per run
    brdf_onset = {"vanilla_v2": 10000, "unified_v6": 15000}

    for row, scene in enumerate(SCENES):
        color = SCENE_COLORS[scene]
        for col, (tag, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for run in RUNS:
                pts = data[run][scene].get(tag, [])
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, color=color, label=RUN_LABELS[run], **RUN_STYLES[run])

            # Mark BRDF-stage onset for each run
            for run in RUNS:
                if data[run][scene].get(tag):
                    ax.axvline(brdf_onset[run], color="grey",
                               linestyle=":" if run == "unified_v6" else "-.",
                               linewidth=1, alpha=0.6,
                               label=f"BRDF on ({RUN_LABELS[run]})")

            if col == 0:
                ax.set_ylabel(f"{scene.capitalize()}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("Training Iteration")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(_kfmt))
            ax.legend(fontsize=7)

    fig.tight_layout()
    _savefig(fig, f"{OUT_DIR}/fig8_normal_loss.pdf")
    return fig


# ── Figure 9: S3IM loss + curriculum schedules (unified_v6 only) ─────────────
def fig_s3im_curriculum(data):
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("S3IM Loss and Curriculum Schedules — Unified v6",
                 fontsize=13, fontweight="bold")
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Top row: s3im_loss per scene (RGB + BRDF overlaid)
    for col, scene in enumerate(SCENES):
        ax = fig.add_subplot(gs[0, col])
        color = SCENE_COLORS[scene]
        d = data["unified_v6"][scene]

        rgb_pts  = d.get("train/s3im_loss", [])
        brdf_pts = d.get("train/s3im_loss_brdf", [])

        if rgb_pts:
            ax.plot([p[0] for p in rgb_pts], [p[1] for p in rgb_pts],
                    color=color, linewidth=1.8, label="RGB branch")
        if brdf_pts:
            ax.plot([p[0] for p in brdf_pts], [p[1] for p in brdf_pts],
                    color=color, linewidth=1.8, linestyle="--", label="BRDF branch")

        ax.set_title(scene.capitalize(), fontweight="bold")
        ax.set_xlabel("Iteration")
        if col == 0:
            ax.set_ylabel("S3IM Loss")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_kfmt))
        ax.legend(fontsize=7)

    # Bottom row: curriculum schedules from ficus (representative; same for all scenes)
    sched_tags = [
        ("train/curriculum_brdf_ramp",   "BRDF Ramp",        "#E84855"),
        ("train/curriculum_alpha_nep",   "NeP α curriculum", "#2E86AB"),
        ("train/curriculum_s3im_rgb_w",  "S3IM w (RGB)",     "#F4A261"),
        ("train/curriculum_s3im_brdf_w", "S3IM w (BRDF)",    "#6A0572"),
    ]
    ax_sched = fig.add_subplot(gs[1, :])
    d_ficus = data["unified_v6"]["ficus"]
    for tag, label, color in sched_tags:
        pts = d_ficus.get(tag, [])
        if pts:
            ax_sched.plot([p[0] for p in pts], [p[1] for p in pts],
                          color=color, linewidth=1.8, label=label)
    ax_sched.set_title("Curriculum Weight Schedules (ficus representative)", fontsize=10)
    ax_sched.set_xlabel("Training Iteration")
    ax_sched.set_ylabel("Weight / Factor")
    ax_sched.xaxis.set_major_formatter(ticker.FuncFormatter(_kfmt))
    ax_sched.legend(fontsize=8, ncol=4)

    _savefig(fig, f"{OUT_DIR}/fig9_s3im_curriculum.pdf")
    return fig


# ── Figure 10: Normal MAE at test checkpoints — v2 vs v6 ─────────────────────
def fig_normal_mae(data):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    fig.suptitle("Normal MAE at Test Checkpoints — Vanilla v2 vs Unified v6",
                 fontsize=13, fontweight="bold")

    for ax, scene in zip(axes, SCENES):
        color = SCENE_COLORS[scene]
        for run in RUNS:
            pts = data[run][scene].get("test/mae", [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            style = RUN_STYLES[run].copy()
            style["marker"] = "o" if run == "vanilla_v2" else "s"
            style["markersize"] = 5
            ax.plot(xs, ys, color=color, label=RUN_LABELS[run], **style)

        ax.set_title(scene.capitalize(), fontweight="bold")
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Normal MAE (°)" if scene == "ficus" else "")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_kfmt))
        ax.legend(fontsize=8)

    fig.tight_layout()
    _savefig(fig, f"{OUT_DIR}/fig10_normal_mae.pdf")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  default="/root/TensoIR/thesis_monitor_log.csv")
    parser.add_argument("--show", action="store_true", help="Open interactive plot windows")
    args = parser.parse_args()

    print(f"Loading: {args.csv}")
    df = load(args.csv)
    print(f"  {len(df)} samples | "
          f"{df['elapsed_hr'].max():.1f} hrs logged | "
          f"from {df['timestamp'].min()} to {df['timestamp'].max()}")

    print("\nGenerating figures (monitor CSV):")
    fig_gpu_overview(df)
    fig_cpu_ram(df)
    fig_io(df)
    fig_psnr_convergence(df)
    fig_vram_stages(df)
    fig_dashboard(df)

    print("\nLoading TFEvents for training-dynamics figures…")
    wanted = {
        "train/PSNRs_rgb", "train/PSNRs_rgb_brdf",
        "train/normals_diff_loss", "train/normals_orientation_loss",
        "train/s3im_loss", "train/s3im_loss_brdf",
        "train/curriculum_brdf_ramp", "train/curriculum_alpha_nep",
        "train/curriculum_s3im_rgb_w", "train/curriculum_s3im_brdf_w",
        "test/mae",
    }
    tf_data = load_all_runs(wanted, every_n=100)

    print("\nGenerating figures (TFEvents):")
    fig_psnr_v2_vs_v6(tf_data)
    fig_normal_loss(tf_data)
    fig_s3im_curriculum(tf_data)
    fig_normal_mae(tf_data)

    print(f"\nAll figures saved to: {OUT_DIR}/")
    print("Formats: PDF (vector, for LaTeX) + PNG (300 DPI, for Word/slides)")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()


if __name__ == "__main__":
    main()
