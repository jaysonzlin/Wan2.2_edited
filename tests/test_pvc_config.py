import pytest

from training.pvc_config import load_pvc_config


def pvc_config_text() -> str:
    return """\
data:
  dataset_root: training_dataset
  num_frames: 49
  num_points: 2048
  object_id: "000"
  utonia_cache_root: ./outputs/utonia_feature_cache
  point_view_utonia_cache_root: ./outputs/utonia_point_view_feature_cache
model:
  n_layers: 8
  latent_dim: 256
  num_heads: 4
  point_embed: true
  frame_cond: true
  transformer_block: SpatialTemporalTransformerBlock
  utonia_enabled: true
  conditioning: history
  history_frames: 4
objective:
  type: ddpm
  num_train_timesteps: 1000
  beta_schedule: linear
  time_shift: 5.0
lr_scheduler: constant
checkpoints_total_limit: 2
"""


def test_pvc_config_accepts_fixed_history_ddpm_contract(tmp_path):
    path = tmp_path / "pvc.yaml"
    path.write_text(pvc_config_text())

    config = load_pvc_config(path, [])

    assert config["model"]["conditioning"] == "history"
    assert config["data"]["point_view_utonia_cache_root"].endswith("feature_cache")


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (("conditioning: history", "conditioning: velocity"), "PVC requires model.conditioning 'history'"),
        (("type: ddpm", "type: flow"), "history conditioning requires objective.type 'ddpm'"),
        (("utonia_enabled: true", "utonia_enabled: false"), "PVC requires model.utonia_enabled true"),
    ],
)
def test_pvc_config_rejects_non_pvc_model_contract(tmp_path, replacement, message):
    old, new = replacement[:2]
    path = tmp_path / "pvc.yaml"
    path.write_text(pvc_config_text().replace(old, new))

    with pytest.raises(ValueError, match=message):
        load_pvc_config(path, [])


def test_pvc_config_requires_point_view_cache_root(tmp_path):
    path = tmp_path / "pvc.yaml"
    path.write_text(
        pvc_config_text().replace(
            "  point_view_utonia_cache_root: ./outputs/utonia_point_view_feature_cache\n", ""
        )
    )

    with pytest.raises(ValueError, match="data.point_view_utonia_cache_root"):
        load_pvc_config(path, [])
