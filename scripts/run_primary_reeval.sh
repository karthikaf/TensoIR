#!/usr/bin/env bash
# Run evaluation-only passes on all primary + ablation checkpoints to generate
# metrics_per_view.npz files needed for mean±std reporting in Chapter 4.
# Run this script from the TensoIR repo root on the H200:
#   bash scripts/run_primary_reeval.sh 2>&1 | tee scripts/reeval.log

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO/log/log_rotated_multi_lights"
TRAIN="$REPO/train_tensoIR_rotated_multi_lights.py"

run_eval() {
    local run_dir="$1"
    local ckpt_name="$2"
    local cfg_name="$3"
    echo "===== $(basename $run_dir) ====="
    python "$TRAIN" \
        --config "$run_dir/$cfg_name" \
        --ckpt   "$run_dir/$ckpt_name" \
        --basedir "$run_dir" \
        --expname "reeval_final" \
        --render_only 1 \
        --render_test 1
}

# ── Primary comparison (Tables 4.2, 4.3, 4.5) ──────────────────────────────
run_eval "$LOG/ficus_unified_v6-20260416-014135"          ficus_unified_v6.th             ficus_unified.txt
run_eval "$LOG/ficus_vanilla_rotated_baseline-20260409-102551" ficus_vanilla_rotated_baseline.th ficus.txt
run_eval "$LOG/hotdog_unified_v6-20260416-040838"         hotdog_unified_v6.th            hotdog_unified.txt
run_eval "$LOG/hotdog_vanilla_rotated_baseline-20260409-104734" hotdog_vanilla_rotated_baseline.th hotdog.txt
run_eval "$LOG/lego_unified_v6-20260416-062047"           lego_unified_v6.th              lego_unified.txt
run_eval "$LOG/lego_vanilla_rotated_baseline-20260409-121707" lego_vanilla_rotated_baseline.th lego.txt

# ── Curriculum ablation v1–v5 (Table 4.10) ──────────────────────────────────
# v1: only the second run has a checkpoint
run_eval "$LOG/ficus_unified_v1-20260415-100212"  ficus_unified_v1.th  ficus_unified.txt
run_eval "$LOG/ficus_unified_v2-20260415-112522"  ficus_unified_v2.th  ficus_unified.txt
run_eval "$LOG/ficus_unified_v2-20260415-112640"  ficus_unified_v2.th  ficus_unified.txt
# v3: all four runs (3 collapsed + 1 converged)
run_eval "$LOG/ficus_unified_v3-20260415-140838"  ficus_unified_v3.th  ficus_unified.txt
run_eval "$LOG/ficus_unified_v3-20260415-141349"  ficus_unified_v3.th  ficus_unified.txt
run_eval "$LOG/ficus_unified_v3-20260415-141844"  ficus_unified_v3.th  ficus_unified.txt
run_eval "$LOG/ficus_unified_v3-20260415-144059"  ficus_unified_v3.th  ficus_unified.txt
run_eval "$LOG/ficus_unified_v4-20260415-155536"  ficus_unified_v4.th  ficus_unified.txt
run_eval "$LOG/ficus_unified_v5-20260415-234054"  ficus_unified_v5.th  ficus_unified.txt

echo "===== All re-evaluations complete ====="
echo "metrics_per_view.npz files are in: <run_dir>/test_reeval_final-*/imgs_test_all/"