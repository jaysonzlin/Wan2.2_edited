import torch
import torch.nn as nn
import pytest

from wan.modules.joint_wan_physctrl import (
    BidirectionalWanPhysCtrlBridge,
    JointWanPhysCtrlModel,
)
from wan.modules.model import WanModel
from wan.modules.pc_trajectory import PCTrajectoryModel


def test_bridge_is_an_exact_identity_while_both_residual_gates_are_zero():
    bridge = BidirectionalWanPhysCtrlBridge(video_dim=12, point_dim=8, interaction_dim=8, num_heads=2)
    video = torch.randn(1, 3, 12, requires_grad=True)
    points = torch.randn(1, 2, 5, 8, requires_grad=True)

    video_out, points_out = bridge(video, points)

    assert bridge.video_gate.shape == (12,)
    assert bridge.point_gate.shape == (8,)
    assert bridge.video_gate.eq(0).all()
    assert bridge.point_gate.eq(0).all()
    torch.testing.assert_close(video_out, video)
    torch.testing.assert_close(points_out, points)


def test_bridge_projections_use_xavier_weights_and_zero_biases():
    torch.manual_seed(7)
    bridge = BidirectionalWanPhysCtrlBridge(
        video_dim=12, point_dim=8, interaction_dim=8, num_heads=2
    )

    for projection in (
        bridge.video_to_points_q,
        bridge.video_to_points_k,
        bridge.video_to_points_v,
        bridge.video_to_points_out,
        bridge.points_to_video_q,
        bridge.points_to_video_k,
        bridge.points_to_video_v,
        bridge.points_to_video_out,
    ):
        expected_variance = 2.0 / (projection.in_features + projection.out_features)
        assert projection.bias.eq(0).all()
        assert projection.weight.var().item() == pytest.approx(expected_variance, rel=0.45)


def test_bridge_updates_both_branches_and_has_finite_gradients_when_enabled():
    bridge = BidirectionalWanPhysCtrlBridge(video_dim=12, point_dim=8, interaction_dim=8, num_heads=2)
    with torch.no_grad():
        bridge.video_gate.fill_(1)
        bridge.point_gate.fill_(1)
    video = torch.randn(1, 3, 12, requires_grad=True)
    points = torch.randn(1, 2, 5, 8, requires_grad=True)

    video_out, points_out = bridge(video, points)
    (video_out.square().mean() + points_out.square().mean()).backward()

    assert video_out.shape == video.shape
    assert points_out.shape == points.shape
    assert torch.isfinite(video.grad).all()
    assert torch.isfinite(points.grad).all()
    assert bridge.video_to_points_q.weight.grad is not None
    assert bridge.points_to_video_q.weight.grad is not None


def test_each_object_receives_only_video_tokens_while_video_receives_all_objects():
    torch.manual_seed(0)
    bridge = BidirectionalWanPhysCtrlBridge(video_dim=12, point_dim=8, interaction_dim=8, num_heads=2)
    with torch.no_grad():
        bridge.video_gate.fill_(1)
        bridge.point_gate.fill_(1)
    video = torch.randn(1, 3, 12)
    points = torch.randn(1, 3, 5, 8)

    baseline_video, baseline_points = bridge(video, points)
    changed = points.clone()
    changed[:, 0, :, 0].add_(10)
    changed_video, changed_points = bridge(video, changed)

    assert not torch.allclose(changed_video, baseline_video)
    torch.testing.assert_close(changed_points[:, 1:], baseline_points[:, 1:])


class _ZeroAttention(nn.Module):
    def forward(self, query, *args):
        return torch.zeros_like(query)


def test_joint_wrapper_supports_multiple_object_trajectories():
    wan = WanModel(
        model_type="ti2v", patch_size=(1, 2, 2), text_len=2, in_dim=1, dim=64,
        ffn_dim=128, freq_dim=16, text_dim=4, out_dim=1, num_heads=1, num_layers=8,
    )
    for block in wan.blocks:
        block.self_attn = _ZeroAttention()
        block.cross_attn = _ZeroAttention()
    pc = PCTrajectoryModel(
        n_points=2, n_future_frames=48, latent_dim=64, n_layers=8, num_heads=1,
        objective_type="ddpm",
    )
    model = JointWanPhysCtrlModel(wan, pc)
    video, trajectories = model(
        video_x=[torch.randn(1, 1, 2, 2)],
        video_t=torch.tensor([[1.0]]),
        context=[torch.zeros(2, 4)],
        seq_len=1,
        noisy_future_state=torch.randn(1, 3, 48, 1, 2, 3),
        frame_times=torch.ones(1, 3, 49),
        init_pc=torch.randn(1, 3, 1, 2, 3),
        initial_linear_velocity=torch.randn(1, 3, 1, 3),
        initial_angular_velocity=torch.randn(1, 3, 1, 3),
    )

    assert video[0].shape == (1, 1, 2, 2)
    assert trajectories.shape == (1, 3, 48, 1, 2, 3)
