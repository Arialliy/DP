#!/usr/bin/env bash
# DeepPro-MXC pipeline with hard stops: any failing step ends the script, M0c NO-GO included.
#
#   bash run_mx.sh check     tests (torch ones included), labels vs PNGs, and compare if VT_OLD is set
#   bash run_mx.sh compare   VT_OLD vs VT_ROOT: same frames, boxes and track relation?
#   bash run_mx.sh profile   speeds and homographies (results/profile, old one moved aside), folds
#   bash run_mx.sh m0        M0c, the go / no-go test of motion calibration; exits 3 on NO-GO
#   bash run_mx.sh train     only if M0c said GO and is newer than targets.csv; best_model.pth by SELECT
#   bash run_mx.sh eval      inference with best_model.pth and scoring (box metrics, authors' protocol)
#   bash run_mx.sh controls  the ablations with the same recipe: no exemplars, no gain, mx_lv
#   bash run_mx.sh all       check, profile, m0, train, eval; stops at the first failure
#
# Environment: VT_ROOT (the RGB-T folder, required); OFFICIAL_LISTS (default VT_ROOT);
# VT_OLD (optional, an earlier copy for compare_datasets.py); SEED (46); DEVICE (cuda);
# ARCH (mxc_lv); RUN (mxc_s$SEED); NPROC (8); FOLDS (1: run folds.py); PY (python);
# M0_SPLIT (official_train); DET_RUN (optional cached run: M0c also checks its peaks as exemplars);
# TRAIN_SPLIT (official_train); SELECT (original_test_iou: the authors' per-epoch test-window IoU;
# val_iou: the same on SELECT_SPLIT, default val, which then must not be in TRAIN_SPLIT; final);
# EVAL_SPLIT (official_test). For the validation protocol: TRAIN_SPLIT=train SELECT=val_iou
# EVAL_SPLIT=val. Outputs go to the current directory. Labels are always paper_k6_floor.
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/code"
PY="${PY:-python}"
: "${VT_ROOT:?set VT_ROOT to the RGB-T folder}"
export VT_ROOT OFFICIAL_LISTS="${OFFICIAL_LISTS:-$VT_ROOT}"
SEED="${SEED:-46}"; DEVICE="${DEVICE:-cuda}"; ARCH="${ARCH:-mxc_lv}"; RUN="${RUN:-${ARCH%%_*}_s$SEED}"
M0_SPLIT="${M0_SPLIT:-official_train}"; TRAIN_SPLIT="${TRAIN_SPLIT:-official_train}"
SELECT="${SELECT:-original_test_iou}"; SELECT_SPLIT="${SELECT_SPLIT:-val}"; EVAL_SPLIT="${EVAL_SPLIT:-official_test}"
LABEL=paper_k6_floor
M0_SUMMARY=results/calib_diag/summary.json
TARGETS=results/profile/targets.csv
case "$SELECT" in
    original_test_iou|val_iou) CKPT_NAME=best_model.pth ;;
    final) CKPT_NAME=final.pth ;;
    *) echo "run_mx.sh: SELECT must be original_test_iou, val_iou or final" >&2; exit 2 ;;
esac

die() { echo "run_mx.sh: $*" >&2; exit 1; }

check() {
    for t in "$CODE"/test_*.py; do
        echo "== $(basename "$t")"
        "$PY" "$t" > /dev/null || die "test failed: $t"
    done
    "$PY" "$CODE/check_labels.py" || die "labels differ from $VT_ROOT/segmentations"
    if [ -n "${VT_OLD:-}" ]; then
        compare
    fi
}

compare() {
    : "${VT_OLD:?set VT_OLD to the earlier copy (e.g. VT-Tiny-MOT)}"
    local status=0
    "$PY" "$CODE/compare_datasets.py" --a "$VT_OLD" --b "$VT_ROOT" || status=$?
    case "$status" in
        0) ;;
        2) die "same frames and boxes, but the track relation cannot be confirmed for some clips" \
               "(boxes at shared positions): read the report" ;;
        *) die "the two copies differ: read the report before comparing with earlier results" ;;
    esac
}

profile() {
    if [ -e results/profile ]; then
        mv results/profile "results/profile.before.$(date +%Y%m%d-%H%M%S)"
    fi
    NPROC="${NPROC:-8}" "$PY" "$CODE/profile_motion.py"
    [ -f results/profile/targets.json ] || die "profile_motion.py wrote no targets.json"
    if [ "${FOLDS:-1}" = 1 ]; then
        "$PY" "$CODE/folds.py"
    fi
}

m0() {
    [ -f "$TARGETS" ] || die "$TARGETS is missing: run 'bash run_mx.sh profile' first"
    local status=0 det=()
    if [ -n "${DET_RUN:-}" ]; then det=(--det-run "$DET_RUN"); fi
    # shellcheck disable=SC2086  # M0_SPLIT may name several splits
    "$PY" "$CODE/calib_diag.py" --split $M0_SPLIT --targets "$TARGETS" --nproc "${NPROC:-8}" ${det[@]+"${det[@]}"} || status=$?
    if [ "$status" = 3 ]; then
        echo "run_mx.sh: M0c says NO-GO; DeepPro-MXC is not trained (docs/MX.md, M0c)" >&2
        exit 3
    fi
    [ "$status" = 0 ] || die "calib_diag.py failed (status $status)"
}

require_go() {
    [ -f "$TARGETS" ] || die "$TARGETS is missing: run 'bash run_mx.sh profile' first"
    [ -f "$M0_SUMMARY" ] || die "no M0c result ($M0_SUMMARY): run 'bash run_mx.sh m0' first"
    [ "$M0_SUMMARY" -nt "$TARGETS" ] || die "M0c is older than $TARGETS: run 'bash run_mx.sh m0' again"
    "$PY" - "$M0_SUMMARY" <<'PYEOF' || { echo "run_mx.sh: M0c says NO-GO; not training" >&2; exit 3; }
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))['decision']['go'] else 3)
PYEOF
}

train_one() {   # run arch extra-arguments...
    local run="$1" arch="$2"
    shift 2
    local sel=(--checkpoint-selection "$SELECT")
    if [ "$SELECT" = val_iou ]; then sel+=(--selection-split "$SELECT_SPLIT"); fi
    "$PY" "$CODE/train.py" --run "$run" --arch "$arch" --split "$TRAIN_SPLIT" --seed "$SEED" --device "$DEVICE" \
        --label-rule "$LABEL" "${sel[@]}" "$@"
    [ -f "runs/$run/$CKPT_NAME" ] || die "training ended without runs/$run/$CKPT_NAME"
}

eval_one() {    # run
    local run="$1"
    [ -f "runs/$run/$CKPT_NAME" ] || die "runs/$run/$CKPT_NAME is missing: run 'bash run_mx.sh train' first"
    "$PY" "$CODE/infer_mx.py" --run "$run" --ckpt "runs/$run/$CKPT_NAME" --clips "$EVAL_SPLIT" --device "$DEVICE" \
        --save-arms
    "$PY" "$CODE/score.py" --run "$run" --split "$EVAL_SPLIT" --author --label-rule "$LABEL"
}

train() {
    require_go
    train_one "$RUN" "$ARCH"
}

evaluate() {
    eval_one "$RUN"
}

controls() {    # the same recipe and selection, one component removed at a time
    require_go
    train_one "${RUN}_noex" "$ARCH" --ex-p-none 1 && eval_one "${RUN}_noex"
    train_one "${RUN}_nogain" "$ARCH" --gain-aug 1 && eval_one "${RUN}_nogain"
    train_one "mx_s$SEED" mx_lv && eval_one "mx_s$SEED"
}

case "${1:-}" in
    check) check ;;
    compare) compare ;;
    profile) profile ;;
    m0) m0 ;;
    train) train ;;
    eval) evaluate ;;
    controls) controls ;;
    all) check; profile; m0; train; evaluate ;;
    *) sed -n '2,24p' "$0" >&2; exit 2 ;;
esac
