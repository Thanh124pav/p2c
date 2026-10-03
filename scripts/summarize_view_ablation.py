#!/usr/bin/env python
"""Summarise a view ablation (SETUP.md section 19).

Produces everything section 19 asks for — an overall comparison table, a per-task table, a
per-stage table, CSV output and a figure comparing view subsets — plus the two analyses
the plan gates on those tables: the complementarity metric of section 13 and the automated
kill-criteria verdict of section 21.

Usage::

    python scripts/summarize_view_ablation.py --input outputs/runs
    python scripts/summarize_view_ablation.py --input outputs/runs --out outputs/summary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p2c.analysis.complementarity import complementarity, most_improved_frames  # noqa: E402
from p2c.analysis.kill_criteria import evaluate, report  # noqa: E402
from p2c.analysis.results import (  # noqa: E402
    group_by_condition,
    group_by_task,
    join_per_sample,
    load_runs,
)
from p2c.analysis.stage_metrics import (  # noqa: E402
    delta_by_stage,
    overall_delta,
    stage_frame_counts,
)
from p2c.analysis.visualization import (  # noqa: E402
    plot_complementarity_hist,
    plot_stage_deltas,
    render_improved_frames,
)

# Display order: partial observation first, then single views, pairs, oracle, controls.
ORDER = [
    "single_primary", "single_wrist", "single_secondary",
    "primary+wrist", "primary+secondary", "random_two",
    "primary+duplicate", "primary+wrist_dropout", "all_views",
]


def _sort_key(cond: str) -> tuple:
    return (ORDER.index(cond) if cond in ORDER else len(ORDER), cond)


def _baseline_from_cache(runs) -> list[float]:
    """Recompute the predict-the-mean baseline for runs that did not record it.

    Older runs predate this field. Recomputing is exact rather than approximate, because
    the baseline depends only on the cache and the split, both of which the run metadata
    pins down.
    """
    from p2c.data.frame_cache import FrameCache

    out = []
    seen: set[str] = set()
    for r in runs:
        cache_path = (r.meta.get("dataset") or {}).get("cache")
        split = r.meta.get("split") or {}
        if not cache_path or cache_path in seen:
            continue
        if not Path(cache_path).exists():
            continue
        try:
            cache = FrameCache(cache_path)
            tr = split.get("train_episodes")
            va = split.get("val_episodes")
            if not tr or not va:
                tr, va = cache.split_episodes(
                    float(split.get("val_fraction", 0.2)),
                    int(split.get("split_seed", 0)),
                )
            out.append(cache.trivial_baseline_mse(list(tr), list(va)))
            seen.add(cache_path)
        except Exception as e:
            print(f"[baseline] could not recompute from {cache_path}: {e}")
    return out


def write_csv(path: Path, rows: list[dict]) -> Path | None:
    if not rows:
        return None
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def overall_table(runs) -> list[dict]:
    """One row per condition, averaged over seeds, with the spread across seeds."""
    rows = []
    for cond, rs in sorted(group_by_condition(runs).items(), key=lambda kv: _sort_key(kv[0])):
        mses = [r.best_val_mse for r in rs if np.isfinite(r.best_val_mse)]
        l1s = [r.best_val_l1 for r in rs if np.isfinite(r.best_val_l1)]
        rows.append({
            "condition": cond,
            "num_views": rs[0].num_views,
            "n_seeds": len(rs),
            "val_mse_mean": float(np.mean(mses)) if mses else float("nan"),
            "val_mse_std": float(np.std(mses)) if len(mses) > 1 else 0.0,
            "val_l1_mean": float(np.mean(l1s)) if l1s else float("nan"),
            "params_trainable": rs[0].params,
            "capacity_matched": rs[0].capacity_matched,
            "seeds": ",".join(str(r.seed) for r in rs),
        })
    # Relative to the partial-observation baseline, if present.
    base = next((r for r in rows if r["condition"] == "single_primary"), None)
    for r in rows:
        if base and base["val_mse_mean"]:
            r["rel_vs_primary_pct"] = round(
                100.0 * (base["val_mse_mean"] - r["val_mse_mean"]) / base["val_mse_mean"], 2
            )
        else:
            r["rel_vs_primary_pct"] = float("nan")
    return rows


def per_task_table(runs) -> list[dict]:
    rows = []
    for task, trs in sorted(group_by_task(runs).items()):
        for cond, rs in sorted(group_by_condition(trs).items(), key=lambda kv: _sort_key(kv[0])):
            mses = [r.best_val_mse for r in rs if np.isfinite(r.best_val_mse)]
            rows.append({
                "task": task,
                "condition": cond,
                "n_seeds": len(rs),
                "val_mse_mean": float(np.mean(mses)) if mses else float("nan"),
                "val_mse_std": float(np.std(mses)) if len(mses) > 1 else 0.0,
                "num_views": rs[0].num_views,
            })
    return rows


def per_stage_table(runs, primary: str, complementary: str, seed: int) -> list[dict]:
    """``Delta_view(g)`` per task, for tasks that carry stage annotations."""
    rows = []
    for task, trs in sorted(group_by_task(runs).items()):
        keys, errors, stage = join_per_sample(trs)
        if not errors or primary not in errors or complementary not in errors:
            continue
        if stage.size == 0 or not (stage >= 0).any():
            continue
        names = next((r.stage_names for r in trs if r.stage_names), {})
        deltas = delta_by_stage(
            errors[primary], errors[complementary], stage, names, seed=seed
        )
        for d in deltas:
            row = d.as_row()
            row["task"] = task
            row["partial"] = primary
            row["multi"] = complementary
            rows.append(row)
    return rows


def make_figure(rows: list[dict], out: Path) -> Path | None:
    """One bar chart of validation MSE per view subset, with seed spread as error bars."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figure] matplotlib not installed; skipping")
        return None

    rows = [r for r in rows if np.isfinite(r["val_mse_mean"])]
    if not rows:
        return None

    labels = [r["condition"] for r in rows]
    vals = [r["val_mse_mean"] for r in rows]
    errs = [r["val_mse_std"] for r in rows]

    # Colour by role so the controls are visually separable from the real conditions.
    def colour(c: str) -> str:
        if c.startswith("single_"):
            return "#4C78A8"
        if c in ("primary+duplicate", "random_two", "primary+wrist_dropout"):
            return "#E45756"      # controls
        if c == "all_views":
            return "#54A24B"      # oracle
        return "#F58518"          # fixed complementary pairs

    fig, ax = plt.subplots(figsize=(max(6.0, 1.15 * len(rows)), 4.2))
    x = np.arange(len(rows))
    ax.bar(x, vals, yerr=errs, capsize=3, color=[colour(c) for c in labels])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("validation action MSE (normalised)")
    ax.set_title("P2C view ablation: lower is better")

    base = next((r["val_mse_mean"] for r in rows if r["condition"] == "single_primary"), None)
    if base:
        ax.axhline(base, ls="--", lw=1, color="#555")
        ax.text(
            len(rows) - 0.4, base, " B0 partial", va="bottom", ha="right",
            fontsize=7, color="#555",
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4C78A8"),
        plt.Rectangle((0, 0), 1, 1, color="#F58518"),
        plt.Rectangle((0, 0), 1, 1, color="#E45756"),
        plt.Rectangle((0, 0), 1, 1, color="#54A24B"),
    ]
    ax.legend(
        handles, ["single view", "complementary pair", "control", "all-view oracle"],
        fontsize=7, frameon=False, loc="best",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=str, default="outputs/runs")
    p.add_argument("--out", type=str, default="outputs/summary")
    p.add_argument("--primary", type=str, default="single_primary")
    p.add_argument("--complementary", type=str, default="primary+wrist")
    p.add_argument("--min-rel-gain", type=float, default=0.05)
    p.add_argument("--equivalence-margin", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-overfit", action="store_true")
    args = p.parse_args()

    runs = load_runs(args.input, include_overfit=args.include_overfit)
    if not runs:
        print(f"no runs found under {args.input}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loaded {len(runs)} runs from {args.input}")
    print(f"tasks     : {sorted(group_by_task(runs))}")
    print(f"conditions: {sorted(group_by_condition(runs), key=_sort_key)}")
    print()

    # ---------------- overall ----------------
    ov = overall_table(runs)
    print("=" * 78)
    print("OVERALL COMPARISON (mean over seeds)")
    print("=" * 78)
    print(f"  {'condition':26s} {'views':>5s} {'seeds':>5s} {'val_mse':>10s} "
          f"{'+/-':>8s} {'vs B0':>8s} {'params':>10s} cap")
    for r in ov:
        print(f"  {r['condition']:26s} {r['num_views']:5d} {r['n_seeds']:5d} "
              f"{r['val_mse_mean']:10.5f} {r['val_mse_std']:8.5f} "
              f"{r['rel_vs_primary_pct']:7.1f}% {r['params_trainable']:10,d} "
              f"{'yes' if r['capacity_matched'] else 'NO'}")
    write_csv(out / "overall.csv", ov)

    # Reference line. Printed right under the table because a condition sitting at or
    # above it has learned nothing, and two such conditions are not meaningfully ranked.
    bl = [r.trivial_baseline_mse for r in runs
          if r.trivial_baseline_mse == r.trivial_baseline_mse]
    if not bl:
        # Runs predating the recorded baseline: recompute it from the cache they used.
        bl = _baseline_from_cache(runs)
    if bl:
        base_mse = float(np.mean(bl))
        best = min((r["val_mse_mean"] for r in ov if np.isfinite(r["val_mse_mean"])),
                   default=float("nan"))
        print()
        print(f"  predict-the-mean baseline : {base_mse:.5f}")
        beaten = [r["condition"] for r in ov if r["val_mse_mean"] < base_mse]
        if not beaten:
            print("  *** NO condition beats the trivial baseline. This ablation has no "
                  "signal; ***")
            print("  *** the comparisons above are noise. See criterion 0 below.        "
                  "       ***")
        else:
            print(f"  conditions beating it     : {len(beaten)}/{len(ov)} "
                  f"(best {best:.5f})")
        n_instr = max((r.num_instructions for r in runs), default=-1)
        if n_instr > 1:
            print(f"  NOTE: cache spans {n_instr} language instructions; part of the "
                  f"action is not predictable from pixels alone.")

    # ---------------- per task ----------------
    pt = per_task_table(runs)
    if len({r["task"] for r in pt}) > 1:
        print()
        print("=" * 78)
        print("PER-TASK COMPARISON")
        print("=" * 78)
        cur = None
        for r in pt:
            if r["task"] != cur:
                cur = r["task"]
                print(f"  {cur}")
            print(f"    {r['condition']:26s} mse={r['val_mse_mean']:.5f} "
                  f"+/-{r['val_mse_std']:.5f}  ({r['n_seeds']} seed)")
    write_csv(out / "per_task.csv", pt)

    # ---------------- joined per-sample analyses ----------------
    all_results: dict = {"n_runs": len(runs), "input": str(args.input)}
    keys, errors, stage = join_per_sample(runs)
    print()
    if not errors:
        print("no overlapping per-sample validation errors across runs; "
              "skipping complementarity, stage and kill-criteria analysis")
        print("(this usually means the runs did not share a split, or none has finished "
              "an evaluation pass)")
    else:
        print(f"joined {len(keys)} shared validation frames across "
              f"{len(errors)} conditions")

        # ---- paired overall delta ----
        if args.primary in errors and args.complementary in errors:
            d = overall_delta(
                errors[args.primary], errors[args.complementary],
                label=f"{args.primary} -> {args.complementary}", seed=args.seed,
            )
            print()
            print("PAIRED COMPARISON (same validation frames)")
            print(f"  {d.label}")
            print(f"    mse {d.mean_partial:.5f} -> {d.mean_multi:.5f}  "
                  f"delta={d.delta:.5f} ({d.rel_delta*100:.1f}%)  "
                  f"95% CI [{d.ci_low:.5f}, {d.ci_high:.5f}]  p={d.p_value:.3f}  "
                  f"{'significant' if d.significant else 'NOT significant'}")
            all_results["paired_overall"] = d.as_row()

        # ---- per stage ----
        ps = per_stage_table(runs, args.primary, args.complementary, args.seed)
        if ps:
            print()
            print("=" * 78)
            print("PER-STAGE COMPARISON (SETUP.md section 11)")
            print("=" * 78)
            print(f"  {'stage':24s} {'n':>6s} {'partial':>9s} {'multi':>9s} "
                  f"{'delta':>9s} {'rel':>7s} {'sig':>4s}")
            for r in ps:
                print(f"  {r['stage']:24s} {r['n_frames']:6d} {r['mse_partial']:9.5f} "
                      f"{r['mse_multi']:9.5f} {r['delta']:9.5f} "
                      f"{r['rel_delta_pct']:6.1f}% {'yes' if r['significant'] else 'no':>4s}")
            write_csv(out / "per_stage.csv", ps)
            fig_stage = plot_stage_deltas(ps, out / "stage_deltas.png")
            if fig_stage:
                print(f"  figure: {fig_stage}")
        else:
            print()
            print("PER-STAGE COMPARISON: unavailable (no stage annotations in these runs).")
            print("  RoboCasa365 ships per-frame stage labels only for target *composite*")
            print("  tasks, so SETUP.md sections 10-11 need a composite dataset.")
        if stage.size and (stage >= 0).any():
            names = next((r.stage_names for r in runs if r.stage_names), {})
            all_results["stage_frame_counts"] = stage_frame_counts(stage, names)

        # ---- complementarity ----
        if args.primary in errors and len(errors) > 1:
            comp = complementarity(errors, args.primary)
            print()
            print("=" * 78)
            print("COMPLEMENTARITY (SETUP.md section 13)")
            print("=" * 78)
            print(comp.summary())
            write_csv(out / "complementarity.csv", comp.as_rows())
            all_results["complementarity"] = {
                "primary": comp.primary_condition,
                "mean_primary_error": comp.mean_primary_error,
                "per_view": comp.per_view,
                "best_fixed": comp.best_fixed_condition,
                "oracle_select_error": comp.oracle_select_error,
                "headroom": comp.headroom,
                "headroom_pct": comp.headroom_pct,
            }
            if args.complementary in errors:
                top = most_improved_frames(
                    keys, errors, args.primary, args.complementary, top_k=16
                )
                all_results["most_improved_frames"] = top
                with open(out / "most_improved_frames.json", "w") as f:
                    json.dump(top, f, indent=2)
                print()
                print("frames where the complementary view helps most "
                      "(SETUP.md section 11):")
                for t in top[:6]:
                    print(f"  episode {t['episode']:4d} frame {t['frame']:4d}  "
                          f"C={t['C']:.5f}  ({t['err_partial']:.5f} -> "
                          f"{t['err_candidate']:.5f})")

                # Render those frames across all cameras. This is the main qualitative
                # check: if the primary view is occluded exactly where the extra view
                # helps, the mechanism is the one P2C claims.
                from p2c.data.frame_cache import FrameCache

                cache_path = next(
                    (r.meta.get("dataset", {}).get("cache") for r in runs
                     if r.meta.get("dataset", {}).get("cache")),
                    None,
                )
                if cache_path and Path(cache_path).exists():
                    try:
                        cache = FrameCache(cache_path)
                        sheet = render_improved_frames(
                            cache, top, out / "most_improved_frames.png"
                        )
                        if sheet:
                            print(f"  rendered: {sheet}")
                        hist = plot_complementarity_hist(
                            errors[args.primary] - errors[args.complementary],
                            out / "complementarity_hist.png",
                            candidate=args.complementary,
                        )
                        if hist:
                            print(f"  rendered: {hist}")
                    except Exception as e:
                        print(f"  [viz] skipped frame rendering: {e}")

        # ---- kill criteria ----
        cap = {}
        for cond, rs in group_by_condition(runs).items():
            cap[cond] = all(r.capacity_matched for r in rs)
        baselines = [r.trivial_baseline_mse for r in runs
                     if r.trivial_baseline_mse == r.trivial_baseline_mse]
        if not baselines:
            baselines = _baseline_from_cache(runs)
        trivial = float(np.mean(baselines)) if baselines else None
        n_instr = max((r.num_instructions for r in runs), default=-1)
        verdicts = evaluate(
            errors,
            stage if (stage.size and (stage >= 0).any()) else None,
            capacity_matched=cap,
            primary=args.primary,
            complementary=args.complementary,
            min_rel_gain=args.min_rel_gain,
            equivalence_margin=args.equivalence_margin,
            seed=args.seed,
            trivial_baseline=trivial,
            num_instructions=max(n_instr, 1),
        )
        print()
        print(report(verdicts))
        all_results["kill_criteria"] = [
            {"criterion": v.criterion, "name": v.name, "status": v.status,
             "detail": v.detail, "numbers": v.numbers}
            for v in verdicts
        ]
        with open(out / "kill_criteria.txt", "w") as f:
            f.write(report(verdicts) + "\n")

    # ---------------- figure ----------------
    fig = make_figure(ov, out / "view_comparison.png")
    if fig:
        print()
        print(f"figure    : {fig}")

    with open(out / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"tables    : {out}/overall.csv, per_task.csv"
          + (", per_stage.csv" if (out / 'per_stage.csv').exists() else ""))
    print(f"json      : {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
