#!/usr/bin/env bash
# Run the controlled view ablation of SETUP.md sections 9 and 12.
#
# Everything except the camera subset is held fixed: same cache, same split, same seed,
# same optimiser settings, same model size. Only --views changes between arms, which is
# the whole point of the comparison.
#
# Arms:
#   B0  single_primary          partial observation (the baseline everything is measured against)
#   B1  single_wrist            alternative single view
#   B2  primary+wrist           fixed complementary pair
#   B3  all_views               full-view oracle (upper bound available from the dataset)
#   B4  random_two              primary + random second view  (Control A, section 12)
#   CD  primary+duplicate       primary twice                 (Control D, section 12)
#   CC  primary+wrist_dropout   pair with view dropout        (Control C, section 12)
#
# B5 (learned view selection) is deliberately absent: SETUP.md section 9 says not to
# implement it until B0-B4 produce a convincing phenomenon.
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
