#!/usr/bin/env bash
# DeepPro-MX pipeline with hard stops: any failing step ends the script, M0 NO-GO included.
#
#   bash run_mx.sh check     tests (torch ones included), labels vs PNGs, and compare if VT_OLD is set
#   bash run_mx.sh compare   VT_OLD vs VT_ROOT: same frames, boxes and track relation?
#   bash run_mx.sh profile   speeds and homographies (results/profile, old one moved aside), folds
#   bash run_mx.sh m0        the go / no-go test; exits 3 on NO-GO
#   bash run_mx.sh train     only if M0 said GO, and M0 is newer than targets.csv
#   bash run_mx.sh eval      inference and scoring (box metrics and the authors' protocol)
#   bash run_mx.sh all       check, profile, m0, train, eval; stops at the first failure
#
# Environment: VT_ROOT (the RGB-T folder, required); OFFICIAL_LISTS (default VT_ROOT);
# VT_OLD (optional, an earlier copy for compare_datasets.py); SEED (46); DEVICE (cuda);
# RUN (mx_s$SEED); NPROC (8); M0_SPLIT (val); FOLDS (1: run folds.py); PY (python).
# Outputs go to the current directory. Labels are always paper_k6_floor.
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/code"
PY="${PY:-python}"
: "${VT_ROOT:?set VT_ROOT to the RGB-T folder}"
export VT_ROOT OFFICIAL_LISTS="${OFFICIAL_LISTS:-$VT_ROOT}"
SEED="${SEED:-46}"; DEVICE="${DEVICE:-cuda}"; RUN="${RUN:-mx_s$SEED}"; M0_SPLIT="${M0_SPLIT:-val}"
LABEL=paper_k6_floor
M0_SUMMARY=results/exemplar_diag/summary.json
TARGETS=results/profile/targets.csv

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
    local status=0
    "$PY" "$CODE/exemplar_diag.py" --split $M0_SPLIT --targets "$TARGETS" || status=$?
    if [ "$status" = 3 ]; then
        echo "run_mx.sh: M0 says NO-GO; DeepPro-MX is not trained (docs/MX.md, Step 1)" >&2
        exit 3
    fi
    [ "$status" = 0 ] || die "exemplar_diag.py failed (status $status)"
}

require_go() {
    [ -f "$TARGETS" ] || die "$TARGETS is missing: run 'bash run_mx.sh profile' first"
    [ -f "$M0_SUMMARY" ] || die "no M0 result ($M0_SUMMARY): run 'bash run_mx.sh m0' first"
    [ "$M0_SUMMARY" -nt "$TARGETS" ] || die "M0 is older than $TARGETS: run 'bash run_mx.sh m0' again"
    "$PY" - "$M0_SUMMARY" <<'PYEOF' || { echo "run_mx.sh: M0 says NO-GO; not training" >&2; exit 3; }
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))['decision']['go'] else 3)
PYEOF
}

train() {
    require_go
    "$PY" "$CODE/train.py" --run "$RUN" --arch mx_lv --seed "$SEED" --device "$DEVICE" --label-rule "$LABEL"
    [ -f "runs/$RUN/final.pth" ] || die "training ended without runs/$RUN/final.pth"
}

evaluate() {
    [ -f "runs/$RUN/final.pth" ] || die "runs/$RUN/final.pth is missing: run 'bash run_mx.sh train' first"
    "$PY" "$CODE/infer_mx.py" --run "$RUN" --ckpt "runs/$RUN/final.pth" --clips val --device "$DEVICE" --save-arms
    "$PY" "$CODE/score.py" --run "$RUN" --split val --author --label-rule "$LABEL"
}

case "${1:-}" in
    check) check ;;
    compare) compare ;;
    profile) profile ;;
    m0) m0 ;;
    train) train ;;
    eval) evaluate ;;
    all) check; profile; m0; train; evaluate ;;
    *) sed -n '2,16p' "$0" >&2; exit 2 ;;
esac
