import pytest
import torch

from wan.modules.pvc_trajectory import PVCTrajectoryModel


def make_model(feature_dim=5):
    return PVCTrajectoryModel(
        n_points=8, n_future_frames=45, latent_dim=64, n_layers=1,
        num_heads=1, utonia_feature_dim=feature_dim,
    )


def pvc_inputs(batch=2, feature_dim=5):
    return (
        torch.randn(batch, 45, 1, 8, 3),
        torch.tensor([[0.0] * 4 + [500.0] * 45] * batch),
        torch.randn(batch, 4, 1, 8, 3),
        torch.randn(batch, 49, 8, 3),
        torch.tensor([[True] * 3 + [False] * 5]).expand(batch, 49, -1).clone(),
        torch.randn(batch, 8, feature_dim),
        torch.randn(batch, 49, 8, feature_dim),
    )


def test_pvc_model_predicts_only_45_trajectory_frames():
    model = make_model()

    output = model(*pvc_inputs())

    assert output.shape == (2, 45, 1, 8, 3)


def test_pvc_model_rejects_invalid_view_mask_shape_or_dtype():
    model = make_model()
    inputs = list(pvc_inputs())
    inputs[4] = torch.ones((2, 49, 8), dtype=torch.float32)

    with pytest.raises(ValueError, match="point_view_mask"):
        model(*inputs)
