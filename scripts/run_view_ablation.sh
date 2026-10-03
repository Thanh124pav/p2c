#!/usr/bin/env bash
# Sweep conditions along the viewpoint partialization axis, holding everything else fixed.
#
# Same cache, split, seed, optimiser settings and model size across every arm; only
# --views changes. That is harness contract C1 (docs/harness_contract.md), and it is the
# whole point of the comparison.
#
# Viewpoint is one partialization axis among several (p2c/data/partialization.py also
# covers temporal cropping, phase drop and occlusion). PLAN.md section 13.6 warns against
# letting this axis define the research problem, so read these arms as a controlled probe,
# not as the question.
#
# Arms:
#   single_primary          the partial observation everything is measured against
#   single_wrist            alternative single view
#   primary+wrist           fixed complementary pair
#   all_views               full-view oracle: the upper bound this dataset allows
#   random_two              primary + a random second view   (control, C9)
#   primary+duplicate       the same camera twice            (control, C9)
#   primary+wrist_dropout   the pair with stochastic dropout (control, C9)
#
# The three controls are what separate "complementary evidence helps" from "more input
# helps": if a random or duplicated view recovers the pair's gain, the effect is scale,
# not complementarity.
#
# No learned view selector is included. PLAN.md section 14 puts method design before
# further implementation, and section 13.5 forbids scaling before a falsifiable signal.
#
# Usage:
#   bash scripts/run_view_ablation.sh
#   CACHE=outputs/cache/Foo_target_r84_e40 SEEDS="0 1 2" bash scripts/run_view_ablation.sh
#   CONDITIONS="single_primary primary+wrist" bash scripts/run_view_ablation.sh

set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python}"
CONFIG="${CONFIG:-configs/local_debug.yaml}"
CACHE="${CACHE:-}"
SEEDS="${SEEDS:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/runs}"
EXTRA="${EXTRA:-}"

CONDITIONS="${CONDITIONS:-single_primary single_wrist primary+wrist primary+secondary all_views random_two primary+duplicate primary+wrist_dropout}"

if [[ -z "$CACHE" ]]; then
  # Fall back to the cache named in the config file.
  CACHE=$(grep -E '^cache:' "$CONFIG" | head -1 | sed 's/^cache:[[:space:]]*//' | tr -d '"')
fi

if [[ ! -f "$CACHE/index.json" ]]; then
  echo "error: no frame cache at '$CACHE'" >&2
  echo "Build one first, for example:" >&2
  echo "  python scripts/build_frame_cache.py --task NavigateKitchen --num-episodes 40 --res 84" >&2
  exit 2
fi

echo "=============================================================================="
echo "P2C view ablation"
echo "=============================================================================="
echo "config     : $CONFIG"
echo "cache      : $CACHE"
echo "seeds      : $SEEDS"
echo "conditions : $CONDITIONS"
echo "output     : $OUTPUT_ROOT"
echo

failed=()
total=0
for seed in $SEEDS; do
  for cond in $CONDITIONS; do
    total=$((total + 1))
    echo "------------------------------------------------------------------------------"
    echo ">>> seed=$seed  views=$cond"
    echo "------------------------------------------------------------------------------"
    if ! $PY scripts/train_local_bc.py \
        --config "$CONFIG" \
        --cache "$CACHE" \
        --views "$cond" \
        --seed "$seed" \
        --output-root "$OUTPUT_ROOT" \
        $EXTRA; then
      echo "!!! FAILED: seed=$seed views=$cond" >&2
      failed+=("seed=$seed/$cond")
    fi
    echo
  done
done

echo "=============================================================================="
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "$((total - ${#failed[@]}))/$total arms finished; ${#failed[@]} FAILED:" >&2
  for f in "${failed[@]}"; do echo "  - $f" >&2; done
  echo
  echo "Summarising the arms that did finish; the tables will be incomplete." >&2
else
  echo "all $total arms finished"
fi
echo "=============================================================================="
echo

$PY scripts/summarize_view_ablation.py --input "$OUTPUT_ROOT"

# Propagate failure so a CI run or a wrapper script notices.
if [[ ${#failed[@]} -gt 0 ]]; then
  exit 1
fi
