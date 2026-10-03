"""Rendering helpers for the analysis outputs.

PLAN.md section 9 asks for "visualization of frames where multi-view improves the
prediction most", calling these samples especially important for judging whether P2C is a
real problem. :func:`render_improved_frames` produces exactly that contact sheet: one row
per frame, one column per camera, annotated with the per-frame complementarity score.

Looking at these frames is the main qualitative check on the whole study. If the frames
where the wrist view helps most show the gripper occluded in the third-person view, the
mechanism is the one P2C claims. If they look arbitrary, the gain is probably not about
missing evidence at all — which no summary statistic will tell you.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LABEL_H = 20


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    bar = np.zeros((LABEL_H, w, 3), dtype=np.uint8)
    scale = max(0.3, min(0.45, w / 440.0))
    cv2.putText(
        bar, text[: int(w / (6.5 * scale))], (3, LABEL_H - 6),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return np.vstack([bar, img])


def render_improved_frames(
    cache,
    frames: list[dict],
    out_path: str | Path,
    cameras: list[str] | None = None,
    max_rows: int = 12,
    upscale: int = 2,
) -> Path | None:
    """Contact sheet of the frames where an extra view helped most.

    Parameters
    ----------
    cache:
        An open :class:`~p2c.data.frame_cache.FrameCache`.
    frames:
        Output of :func:`p2c.analysis.complementarity.most_improved_frames` — dicts with
        ``episode``, ``frame`` and ``C``.
    cameras:
        Which cameras to show. Defaults to all of them, which is usually what you want:
        the point is to compare what the primary view missed against what the extra view
        saw.
    """
    import cv2

    if not frames:
        return None
    cams = cameras or cache.cameras
    rows = []
    for item in frames[:max_rows]:
        ep, fr = int(item["episode"]), int(item["frame"])
        try:
            span = cache.span_of_episode(ep)
        except KeyError:
            continue
        g = span.start + fr
        if not (span.start <= g < span.stop):
            continue

        tiles = []
        for i, cam in enumerate(cams):
            img = np.asarray(cache.image(cam, g))
            if upscale > 1:
                img = cv2.resize(
                    img, (img.shape[1] * upscale, img.shape[0] * upscale),
                    interpolation=cv2.INTER_NEAREST,
                )
            if i == 0:
                stage = ""
                if cache.has_stages:
                    stage = f" {cache.stage_name(int(cache.stage[g]))}"
                text = f"ep{ep} f{fr} C={item['C']:.4f}{stage}"
            else:
                text = cam
            tiles.append(_caption(img, text))
        rows.append(np.hstack(tiles))

    if not rows:
        return None
    sheet = np.vstack(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet[:, :, ::-1])  # RGB -> BGR
    return out_path


def plot_stage_deltas(rows: list[dict], out_path: str | Path) -> Path | None:
    """Bar chart of ``Delta_view(g)`` per stage with bootstrap intervals.

    Error bars are the point of this figure. A per-stage table of point estimates invites
    reading noise as structure; showing the intervals makes it obvious which stages
    actually separate.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not rows:
        return None

    labels = [r["stage"] for r in rows]
    vals = [r["delta"] for r in rows]
    lo = [r["delta"] - r["ci_low"] for r in rows]
    hi = [r["ci_high"] - r["delta"] for r in rows]
    sig = [bool(r.get("significant")) for r in rows]

    fig, ax = plt.subplots(figsize=(max(5.0, 1.3 * len(rows)), 4.0))
    x = np.arange(len(rows))
    ax.bar(
        x, vals,
        yerr=[np.abs(lo), np.abs(hi)], capsize=4,
        color=["#54A24B" if s else "#BBBBBB" for s in sig],
    )
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(r"$\Delta_{view}(g)$  (partial $-$ multi, MSE)")
    ax.set_title("Improvement from the complementary view, by stage")
    ax.text(
        0.99, 0.02,
        "grey = 95% CI includes zero",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#555",
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_complementarity_hist(
    C: np.ndarray, out_path: str | Path, candidate: str = ""
) -> Path | None:
    """Histogram of per-frame ``C(v | p)``.

    The shape carries information a mean hides: a distribution centred just above zero
    means the extra view helps a little everywhere (consistent with generic scaling),
    while a long positive tail on an otherwise zero-centred bulk means it rescues a
    specific subset of frames — which is the P2C claim.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if C.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.hist(C, bins=60, color="#4C78A8")
    ax.axvline(0, color="#333", lw=1)
    ax.axvline(float(C.mean()), color="#E45756", lw=1.2, ls="--",
               label=f"mean {C.mean():.4f}")
    ax.set_xlabel(r"$C(v \mid p) = L(\pi_p) - L(\pi_{p+v})$  per frame")
    ax.set_ylabel("frames")
    ax.set_title(f"Per-frame complementarity{f': {candidate}' if candidate else ''}")
    ax.legend(fontsize=8, frameon=False)
    frac = float((C > 0).mean())
    ax.text(0.99, 0.95, f"{frac*100:.0f}% of frames improve",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#555")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
