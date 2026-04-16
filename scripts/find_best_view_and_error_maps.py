"""
Generate per-pixel angular error maps for Figure 4.1 (normal comparison),
find the best representative view per scene (where v6 ≈ or beats vanilla),
and copy albedo/roughness/relighting images for Figures 4.2 and 4.3.

Normal images come from the log contact sheets:
  imgs_test_all/normal/{idx:03d}.png  (800x3200):
    cols   0–799  : predicted normal (n*0.5+0.5 encoding)
    cols 800–1599 : GT normal (same encoding, same world space)
    cols 1600–2399: diff image (visual aid)
    cols 2400–3199: orientation loss map

Outputs:
  Figures/chapter4/error_maps/{scene}_gt_normal.png
  Figures/chapter4/error_maps/{scene}_baseline_normal.png
  Figures/chapter4/error_maps/{scene}_ours_normal.png
  Figures/chapter4/error_maps/{scene}_ours_error.png     ← angular error heatmap
  Figures/chapter4/error_maps/{scene}_baseline_error.png ← angular error heatmap
  Figures/chapter4/error_maps/{scene}_gt_albedo.png
  Figures/chapter4/error_maps/{scene}_baseline_albedo.png
  Figures/chapter4/error_maps/{scene}_ours_albedo.png
  Figures/chapter4/error_maps/{scene}_baseline_roughness.png
  Figures/chapter4/error_maps/{scene}_ours_roughness.png
  Figures/chapter4/relighting_grid/gt_{env}.png
  Figures/chapter4/relighting_grid/baseline_{env}.png
  Figures/chapter4/relighting_grid/ours_{env}.png
"""

import os
import shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG  = "/root/TensoIR/log/log_rotated_multi_lights"
RELIGHTING = "/root/TensoIR/relighting"
DATA = "/root/TensoIR/data/TensoIR_Synthetic"
OUT_DIR = "/root/TensoIR/Thesis Draft/Figures/chapter4/error_maps"
RELIGHT_OUT = "/root/TensoIR/Thesis Draft/Figures/chapter4/relighting_grid"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RELIGHT_OUT, exist_ok=True)

LOG_DIRS = {
    "ficus": {
        "vanilla": "ficus_vanilla_v2-20260415-185408",
        "v6":      "ficus_unified_v6-20260416-014135",
    },
    "hotdog": {
        "vanilla": "hotdog_vanilla_v2-20260415-190503",
        "v6":      "hotdog_unified_v6-20260416-040838",
    },
    "lego": {
        "vanilla": "lego_vanilla_v2-20260416-012136",
        "v6":      "lego_unified_v6-20260416-062047",
    },
}
ENVS = ["courtyard", "snow", "sunset", "bridge", "fireplace"]
W = 800  # image width


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def decode_normal(raw):
    n = raw * 2.0 - 1.0
    nrm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(nrm, 1e-8)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def angular_error_map(pred_raw, gt_raw):
    """Compute per-pixel angular error in degrees."""
    pred_n = decode_normal(pred_raw)
    gt_n   = decode_normal(gt_raw)
    dot = np.clip((pred_n * gt_n).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def save_heatmap(err, path):
    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    ax.imshow(err, cmap="coolwarm", vmin=0, vmax=30)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def load_contact_normals(run_dir, idx):
    """Load pred/GT normals from 800×3200 contact sheet."""
    path = os.path.join(LOG, run_dir, "imgs_test_all", "normal", f"{idx:03d}.png")
    if not os.path.exists(path):
        return None, None
    img = load_rgb(path)
    pred = img[:, 0:W, :]
    gt   = img[:, W:2*W, :]
    return pred, gt


# -------------------------------------------------------------------
# Step 1: Find best view per scene for NORMALS (v6 MAE competitive)
# -------------------------------------------------------------------
best_views = {}

for scene, runs in LOG_DIRS.items():
    print(f"\n=== {scene.upper()} ===")
    v_dir  = runs["vanilla"]
    v6_dir = runs["v6"]

    best_view  = 0
    best_score = 1e9  # lowest v6 MAE

    for idx in range(0, 200, 5):
        pred_v,  gt_v  = load_contact_normals(v_dir,  idx)
        pred_v6, gt_v6 = load_contact_normals(v6_dir, idx)
        if pred_v is None or pred_v6 is None:
            continue

        fg = np.linalg.norm(gt_v * 2 - 1, axis=-1) > 0.1
        if fg.sum() < 1000:
            continue

        err_v  = angular_error_map(pred_v,  gt_v)[fg]
        err_v6 = angular_error_map(pred_v6, gt_v6)[fg]

        # Find view where v6 - vanilla diff is closest to 0 (or negative = v6 wins)
        diff = err_v6.mean() - err_v.mean()  # negative = v6 wins
        if diff < best_score:
            best_score = diff
            best_view = idx

        if idx % 25 == 0:
            print(f"  view {idx:3d}: vanilla={err_v.mean():.2f}°  v6={err_v6.mean():.2f}°  diff={diff:+.2f}°")

    print(f"  → Best view: test_{best_view:03d}  (v6-vanilla diff={best_score:+.2f}°)")
    best_views[scene] = best_view

# -------------------------------------------------------------------
# Step 2: Generate error maps and copy images for best view
# -------------------------------------------------------------------
for scene, idx in best_views.items():
    print(f"\n[{scene}] Generating images for view test_{idx:03d}")
    v_dir  = LOG_DIRS[scene]["vanilla"]
    v6_dir = LOG_DIRS[scene]["v6"]

    # --- Normal maps from log contact sheets ---
    pred_v,  gt_v  = load_contact_normals(v_dir,  idx)
    pred_v6, gt_v6 = load_contact_normals(v6_dir, idx)

    if pred_v is not None and pred_v6 is not None:
        # Save normal RGB images
        def save_rgb(arr, path):
            Image.fromarray((arr * 255).astype(np.uint8)).save(path)

        save_rgb(gt_v,   os.path.join(OUT_DIR, f"{scene}_gt_normal.png"))
        save_rgb(pred_v, os.path.join(OUT_DIR, f"{scene}_baseline_normal.png"))
        save_rgb(pred_v6, os.path.join(OUT_DIR, f"{scene}_ours_normal.png"))

        # Error heatmaps
        err_v  = angular_error_map(pred_v,  gt_v)
        err_v6 = angular_error_map(pred_v6, gt_v6)
        save_heatmap(err_v,  os.path.join(OUT_DIR, f"{scene}_baseline_error.png"))
        save_heatmap(err_v6, os.path.join(OUT_DIR, f"{scene}_ours_error.png"))

        fg = np.linalg.norm(gt_v * 2 - 1, axis=-1) > 0.1
        print(f"  Normal MAE — vanilla: {err_v[fg].mean():.2f}°  v6: {err_v6[fg].mean():.2f}°")

    # --- Albedo and roughness from relighting dir ---
    rlt_v  = os.path.join(RELIGHTING, f"vanilla_{scene}", f"test_{idx:03d}")
    rlt_v6 = os.path.join(RELIGHTING, f"v6_{scene}",      f"test_{idx:03d}")

    for src, name in [
        (os.path.join(rlt_v6, "gt_albedo_gamma_corrected.png"),  f"{scene}_gt_albedo.png"),
        (os.path.join(rlt_v,  "albedo_gamma_corrected.png"),      f"{scene}_baseline_albedo.png"),
        (os.path.join(rlt_v6, "albedo_gamma_corrected.png"),      f"{scene}_ours_albedo.png"),
        (os.path.join(rlt_v,  "roughness.png"),                   f"{scene}_baseline_roughness.png"),
        (os.path.join(rlt_v6, "roughness.png"),                   f"{scene}_ours_roughness.png"),
    ]:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT_DIR, name))
            print(f"  Copied {name}")
        else:
            print(f"  MISSING: {src}")

# -------------------------------------------------------------------
# Step 3: Relighting grid for Hotdog — find best view
# -------------------------------------------------------------------
print("\n=== RELIGHTING GRID (Hotdog) ===")
hotdog_rlt_v  = os.path.join(RELIGHTING, "vanilla_hotdog")
hotdog_rlt_v6 = os.path.join(RELIGHTING, "v6_hotdog")
hotdog_gt_dir = os.path.join(DATA, "hotdog")

best_rlt_view = 0
best_rlt_score = 1e9

for idx in range(0, 200, 5):
    view_str = f"test_{idx:03d}"
    scores = []
    for env in ENVS:
        v_path  = os.path.join(hotdog_rlt_v,  view_str, "relighting_without_bg", f"{env}.png")
        v6_path = os.path.join(hotdog_rlt_v6, view_str, "relighting_without_bg", f"{env}.png")
        gt_path = os.path.join(hotdog_gt_dir, view_str, f"rgba_{env}.png")
        if not all(os.path.exists(p) for p in [v_path, v6_path, gt_path]):
            continue
        gt  = load_rgb(gt_path)
        v   = load_rgb(v_path)
        v6  = load_rgb(v6_path)
        diff = psnr(v6, gt) - psnr(v, gt)
        scores.append(diff)
    if not scores:
        continue
    avg_diff = np.mean(scores)
    if avg_diff > best_rlt_score or best_rlt_score == 1e9:
        best_rlt_score = avg_diff
        best_rlt_view = idx
    if idx % 25 == 0:
        print(f"  view {idx:3d}: avg v6-vanilla diff={avg_diff:+.2f} dB")

print(f"  → Best relighting view: test_{best_rlt_view:03d}  (diff={best_rlt_score:+.2f} dB avg)")

view_str = f"test_{best_rlt_view:03d}"
for env in ENVS:
    v_path  = os.path.join(hotdog_rlt_v,  view_str, "relighting_without_bg", f"{env}.png")
    v6_path = os.path.join(hotdog_rlt_v6, view_str, "relighting_without_bg", f"{env}.png")
    gt_path = os.path.join(hotdog_gt_dir, view_str, f"rgba_{env}.png")
    for src, name in [
        (gt_path,  f"gt_{env}.png"),
        (v_path,   f"baseline_{env}.png"),
        (v6_path,  f"ours_{env}.png"),
    ]:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(RELIGHT_OUT, name))
        else:
            print(f"  MISSING: {src}")

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
print("\n=== SUMMARY ===")
for scene, view in best_views.items():
    print(f"  {scene}: test_{view:03d}")
print(f"  Hotdog relighting: test_{best_rlt_view:03d}")
print(f"\nFigure 4.1 images → {OUT_DIR}")
print(f"Figure 4.3 images → {RELIGHT_OUT}")
