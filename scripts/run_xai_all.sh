#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# run_xai_all.sh — Run all 6 XAI methods on new patients (skipping existing)
#
# Usage: bash scripts/run_xai_all.sh
#
# This script is designed to be run on a rented GPU server.
# After completion, copy results/CSVs/xai_*.csv back to your local machine.
# Then run: python scripts/merge_xai_csvs.py
# ─────────────────────────────────────────────────────────────────────

CONFIG="configs/full_training_segresnet.yaml"
CHECKPOINT="models/SegResNet_f32_d0.1_lr5e-05_Full_Run/best_model.pth"

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Please update the CHECKPOINT variable with the correct path."
    exit 1
fi

echo "============================================================="
echo "  XAI Evaluation — Running all 6 methods (skipping existing)"
echo "============================================================="
echo ""
echo "  Config     : $CONFIG"
echo "  Checkpoint : $CHECKPOINT"
echo ""

# ── 1. Grad-CAM (skip 10, limit 30 → 20 new patients) ────────────
echo "[1/5] Grad-CAM (20 new patients)..."
python scripts/generate_gradcam.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --skip 10 --limit 30 --layer encoder3

# ── 2. GBP + Guided Grad-CAM (skip 5, limit 30 → 25 new each) ────
echo ""
echo "[2/5] GBP + Guided Grad-CAM (25 new patients)..."
python scripts/generate_gbp.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --skip 5 --limit 30 --guided_gradcam --layer encoder3

# ── 3. IxG / LRP Proxy (skip 5, limit 30 → 25 new patients) ──────
echo ""
echo "[3/5] Input × Gradient / LRP Proxy (25 new patients)..."
python scripts/generate_lrp.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --skip 5 --limit 30

# ── 4. MC Dropout (skip 3, limit 30 → 27 new patients) ───────────
echo ""
echo "[4/5] MC Dropout + LRP correlation (27 new patients)..."
python scripts/generate_mc_dropout.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --skip 3 --limit 30 --num_iters 20

# ── 5. Occlusion Sensitivity (skip 5, limit 30 → 25 new) ─────────
# This is the SLOWEST method — runs last so others complete sooner
echo ""
echo "[5/5] Occlusion Sensitivity (25 new patients — this will take hours)..."
python scripts/generate_occlusion.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --skip 5 --limit 30 --stride 16

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo "============================================================="
echo "  ALL DONE!"
echo "  Copy these files back to your local machine:"
echo "    results/CSVs/xai_gradcam_metrics.csv"
echo "    results/CSVs/xai_gbp_metrics.csv"
echo "    results/CSVs/xai_guided_gradcam_metrics.csv"
echo "    results/CSVs/xai_lrp_metrics.csv"
echo "    results/CSVs/xai_mc_dropout_metrics.csv"
echo "    results/CSVs/xai_occlusion_metrics.csv"
echo ""
echo "  Then on your local machine, run:"
echo "    python scripts/merge_xai_csvs.py"
echo "============================================================="
