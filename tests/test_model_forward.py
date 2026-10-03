"""Model tests from SETUP.md section 18.

Covers the listed cases — single-view forward, multi-view forward, varying camera counts,
loss backward, no NaNs, batch size 1 — plus the capacity-matching claim of section 12
Control B, which is the one property the whole study rests on: if a 3-view model is
simply bigger than a 1-view model, no comparison between them means anything.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from p2c.models.fusion import fusion_is_capacity_matched  # noqa: E402
from p2c.models.tiny_bc import (  # noqa: E402
    TinyBCPolicy,
    action_metrics,
    bc_loss,
    per_sample_squared_error,
)

ACTION_DIM = 12            # RoboCasa365 PandaOmron
STATE_DIM = 16
RES = 84

# Action groups as read from the dataset's own modality.json.
ACTION_GROUPS = {
    "base_motion": (0, 4),
    "control_mode": (4, 5),
    "end_effector_position": (5, 8),
    "end_effector_rotation": (8, 11),
    "gripper_close": (11, 12),
}


def make_policy(num_views=1, fusion="mean", **kw):
    return TinyBCPolicy(
        action_dim=ACTION_DIM, state_dim=STATE_DIM, num_views=num_views,
        fusion=fusion, input_res=RES, feat_dim=64, width=16, hidden_dim=128, **kw
    )


def batch(b=2, v=1, res=RES, t=None):
    if t is None:
        imgs = torch.rand(b, v, 3, res, res)
    else:
        imgs = torch.rand(b, v, t, 3, res, res)
    return imgs, torch.randn(b, STATE_DIM), torch.ones(b, v, dtype=torch.bool)


# ---------------------------------------------------------------- forward


@pytest.mark.parametrize("v", [1, 2, 3])
def test_forward_with_each_view_count(v):
    m = make_policy(v)
    imgs, st, mask = batch(2, v)
    out = m(imgs, st, mask)
    assert out.shape == (2, ACTION_DIM)
    assert torch.isfinite(out).all()


def test_batch_size_one_works():
    m = make_policy(2)
    imgs, st, mask = batch(1, 2)
    out = m(imgs, st, mask)
    assert out.shape == (1, ACTION_DIM)
    assert torch.isfinite(out).all()


def test_same_weights_accept_any_view_count():
    """A shared encoder plus view-count-independent fusion must generalise over V.

    This is what lets one trained architecture serve every ablation arm.
    """
    m = make_policy(1, fusion="mean")
    for v in (1, 2, 3, 5):
        imgs, st, mask = batch(2, v)
        out = m(imgs, st, mask)
        assert out.shape == (2, ACTION_DIM)


def test_history_frames_are_accepted():
    m = make_policy(2, n_obs_steps=3)
    imgs, st, mask = batch(2, 2, t=3)
    out = m(imgs, st, mask)
    assert out.shape == (2, ACTION_DIM)


def test_action_horizon_shapes():
    m = make_policy(2, action_horizon=4)
    imgs, st, mask = batch(2, 2)
    out = m(imgs, st, mask)
    assert out.shape == (2, 4, ACTION_DIM)


def test_bad_image_rank_is_rejected():
    m = make_policy(1)
    with pytest.raises(ValueError, match="5 or 6 dims"):
        m(torch.rand(2, 3, RES, RES), torch.randn(2, STATE_DIM))


def test_missing_state_is_rejected_when_required():
    m = make_policy(1)
    imgs, _, mask = batch(2, 1)
    with pytest.raises(ValueError, match="use_state=True"):
        m(imgs, None, mask)


def test_policy_without_state_ignores_it():
    m = make_policy(1, use_state=False)
    imgs, _, mask = batch(2, 1)
    out = m(imgs, None, mask)
    assert out.shape == (2, ACTION_DIM)


# ---------------------------------------------------------------- backward


@pytest.mark.parametrize("v", [1, 2, 3])
def test_loss_backward_produces_finite_gradients(v):
    m = make_policy(v)
    imgs, st, mask = batch(4, v)
    target = torch.randn(4, ACTION_DIM)
    loss = bc_loss(m(imgs, st, mask), target, "l2")
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients were produced"
    for g in grads:
        assert torch.isfinite(g).all(), "non-finite gradient"
    assert any(g.abs().sum() > 0 for g in grads), "all gradients are zero"


@pytest.mark.parametrize("kind", ["l2", "l1", "huber"])
def test_all_loss_kinds_backward(kind):
    m = make_policy(2)
    imgs, st, mask = batch(2, 2)
    loss = bc_loss(m(imgs, st, mask), torch.randn(2, ACTION_DIM), kind)
    loss.backward()
    assert torch.isfinite(loss)


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError, match="unknown loss"):
        bc_loss(torch.zeros(2, 3), torch.zeros(2, 3), "cosine")


def test_no_nans_on_extreme_inputs():
    """Saturated and zero images must not produce NaNs."""
    m = make_policy(3)
    st = torch.randn(2, STATE_DIM)
    mask = torch.ones(2, 3, dtype=torch.bool)
    for imgs in (
        torch.zeros(2, 3, 3, RES, RES),
        torch.ones(2, 3, 3, RES, RES),
        torch.full((2, 3, 3, RES, RES), 1e-8),
    ):
        out = m(imgs, st, mask)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------- capacity (Control B)


def test_parameter_count_is_identical_across_view_counts():
    """SETUP.md section 12, Control B.

    With a shared encoder and mean fusion, swapping the view condition cannot change the
    model size, so a multi-view gain cannot be explained by extra capacity.
    """
    counts = {v: make_policy(v, fusion="mean").num_parameters() for v in (1, 2, 3)}
    assert len(set(counts.values())) == 1, f"capacity differs across views: {counts}"


def test_attention_fusion_is_also_capacity_matched():
    counts = {v: make_policy(v, fusion="attn").num_parameters() for v in (1, 2, 3)}
    assert len(set(counts.values())) == 1, f"capacity differs across views: {counts}"


def test_concat_fusion_is_flagged_as_not_capacity_matched():
    """Concat fusion is allowed but must declare that it breaks Control B."""
    with pytest.warns(UserWarning, match="Control B"):
        m = make_policy(3, fusion="concat")
    assert m.capacity_report()["view_count_independent"] is False
    assert not fusion_is_capacity_matched("concat")

    with pytest.warns(UserWarning):
        counts = {v: make_policy(v, fusion="concat").num_parameters() for v in (1, 3)}
    assert len(set(counts.values())) > 1, "concat should scale with the view count"


def test_capacity_report_fields():
    m = make_policy(2)
    rep = m.capacity_report()
    assert rep["view_count_independent"] is True
    assert rep["encoder_shared_across_views"] is True
    assert rep["total_trainable"] == (
        rep["params_encoder"] + rep["params_fusion"] + rep["params_head"]
    )


def test_model_is_small_enough_for_a_4gb_gpu():
    """The tiny baseline must stay tiny; it is a phenomenon detector, not a policy."""
    m = TinyBCPolicy(action_dim=ACTION_DIM, state_dim=STATE_DIM, num_views=3,
                     input_res=RES)
    assert m.num_parameters() < 5_000_000, m.num_parameters()


# ---------------------------------------------------------------- view dropout


def test_dropped_views_change_the_output_under_mean_fusion():
    m = make_policy(2)
    imgs, st, _ = batch(2, 2)
    full = m(imgs, st, torch.ones(2, 2, dtype=torch.bool))
    one = m(imgs, st, torch.tensor([[True, False], [True, False]]))
    assert not torch.allclose(full, one), "the keep mask must actually affect fusion"


def test_mean_fusion_with_one_kept_view_equals_that_view_alone():
    """Masking view 1 must be the same as never passing it."""
    m = make_policy(2)
    imgs, st, _ = batch(2, 2)
    masked = m(imgs, st, torch.tensor([[True, False], [True, False]]))
    alone = m(imgs[:, :1], st, torch.ones(2, 1, dtype=torch.bool))
    assert torch.allclose(masked, alone, atol=1e-5)


def test_attention_fusion_survives_all_views_dropped():
    m = make_policy(3, fusion="attn")
    imgs, st, _ = batch(2, 3)
    out = m(imgs, st, torch.zeros(2, 3, dtype=torch.bool))
    assert torch.isfinite(out).all(), "all-dropped rows must not produce NaNs"


# ---------------------------------------------------------------- metrics


def test_action_metrics_reports_every_group():
    pred = torch.randn(8, ACTION_DIM)
    target = torch.randn(8, ACTION_DIM)
    m = action_metrics(pred, target, ACTION_GROUPS)
    assert "mse" in m and "l1" in m
    for g in ACTION_GROUPS:
        assert f"mse/{g}" in m
        assert f"l1/{g}" in m
    assert m["mse"] > 0


def test_action_metrics_are_zero_for_a_perfect_prediction():
    t = torch.randn(4, ACTION_DIM)
    m = action_metrics(t.clone(), t, ACTION_GROUPS)
    assert m["mse"] == pytest.approx(0.0, abs=1e-12)
    assert m["l1"] == pytest.approx(0.0, abs=1e-12)


def test_action_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        action_metrics(torch.randn(4, 12), torch.randn(4, 11))


def test_action_metrics_skips_out_of_range_groups():
    """A group beyond the action width must be skipped, not crash."""
    m = action_metrics(torch.randn(4, 5), torch.randn(4, 5), ACTION_GROUPS)
    assert "mse/base_motion" in m
    assert "mse/gripper_close" not in m


def test_per_sample_error_shape_and_values():
    pred = torch.zeros(5, ACTION_DIM)
    target = torch.ones(5, ACTION_DIM)
    e = per_sample_squared_error(pred, target)
    assert e.shape == (5,)
    assert torch.allclose(e, torch.ones(5))


def test_per_sample_error_handles_action_horizon():
    e = per_sample_squared_error(torch.zeros(3, 4, ACTION_DIM), torch.ones(3, 4, ACTION_DIM))
    assert e.shape == (3,)
    assert torch.allclose(e, torch.ones(3))
