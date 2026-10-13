import types
import sys

import torch
from torch import nn

import joint_simgen_video_only
from train_joint_wan_physctrl import create_joint_optimizer


def test_video_only_accelerator_detects_intentionally_unused_pc_tail(monkeypatch):
    """Video-only loss leaves the final PC-output path without gradients."""
    class FakeDistributedDataParallelKwargs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAccelerator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    accelerate_module = types.ModuleType("accelerate")
    accelerate_module.Accelerator = FakeAccelerator
    accelerate_utils_module = types.ModuleType("accelerate.utils")
    accelerate_utils_module.DistributedDataParallelKwargs = (
        FakeDistributedDataParallelKwargs
    )
    monkeypatch.setitem(sys.modules, "accelerate", accelerate_module)
    monkeypatch.setitem(sys.modules, "accelerate.utils", accelerate_utils_module)

    accelerator = joint_simgen_video_only._create_video_only_training_accelerator(
        {"gradient_accumulation_steps": 1, "mixed_precision": "bf16"},
        {"report_to": "wandb"},
    )

    assert accelerator.kwargs["kwargs_handlers"][0].kwargs == {
        "find_unused_parameters": True
    }


def test_build_datasets_uses_only_video_dataset_and_configured_ranges(monkeypatch):
    calls = []

    class Dataset:
        def __init__(self, root, sample_ids, **kwargs):
            calls.append((root, sample_ids, kwargs))

    monkeypatch.setattr(joint_simgen_video_only, "SimGenVideoDataset", Dataset)
    config = {
        "data": {
            "dataset_root": "simgen",
            "train_start": 0,
            "train_end": 127,
            "validation_start": 490,
            "validation_end": 499,
        }
    }

    joint_simgen_video_only.build_datasets(config)

    assert calls == [
        ("simgen", list(range(128)), {}),
        ("simgen", list(range(490, 500)), {}),
    ]


def test_video_only_losses_feed_three_synthetic_slots_and_return_video_loss(monkeypatch):
    clean_latents = torch.zeros(1, 2, 1, 2, 2)
    flow = types.SimpleNamespace(
        model_input=clean_latents,
        latent_timesteps=torch.ones(1, 2),
        velocity_target=torch.ones_like(clean_latents),
        loss_mask=torch.ones_like(clean_latents),
    )
    monkeypatch.setattr(joint_simgen_video_only, "_encode_simgen_videos", lambda *_: clean_latents)
    monkeypatch.setattr(joint_simgen_video_only, "make_flow_matching_batch", lambda *_args, **_kwargs: flow)
    monkeypatch.setattr(joint_simgen_video_only, "expand_latent_timesteps", lambda *args: args[0])
    monkeypatch.setattr(joint_simgen_video_only, "masked_velocity_mse", lambda *_args: torch.tensor(2.5))

    class Model:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return [torch.zeros_like(clean_latents[0])], torch.empty(1)

    model = Model()
    batch = {"video": torch.zeros(1, 2, 3, 4, 4)}
    loss, returned_latents, context = joint_simgen_video_only._video_only_losses(
        batch,
        model,
        vae=object(),
        text_encoder=lambda prompts, device: [torch.zeros(1)],
        generator=torch.Generator().manual_seed(5),
        device="cpu",
        data={"num_points": 5, "synthetic_object_slots": 3},
        model_config={"utonia_feature_dim": 7},
        objective={"time_shift": 5.0, "num_train_timesteps": 1000},
    )

    assert loss.item() == 2.5
    assert returned_latents is clean_latents
    assert len(context) == 1
    assert model.kwargs["init_pc"].shape == (1, 3, 4, 1, 5, 3)
    assert model.kwargs["noisy_future_state"].shape == (1, 3, 45, 1, 5, 3)
    assert model.kwargs["utonia_features"].shape == (1, 3, 5, 7)
    assert model.kwargs["frame_times"].shape == (1, 3, 49)


def test_video_only_sampler_holds_one_three_slot_conditioning_draw(monkeypatch):
    history = torch.full((3, 4, 5, 3), 2.0)
    features = torch.full((3, 5, 7), -3.0)
    monkeypatch.setattr(
        joint_simgen_video_only,
        "make_sampling_conditioning",
        lambda **_kwargs: (history, features),
    )
    calls = []

    def pipeline(**kwargs):
        calls.append(kwargs)
        return "sample"

    sample = joint_simgen_video_only._sample_video_only(
        pipeline=pipeline,
        condition_latent=torch.zeros(1, 4, 2, 2),
        video_shape=(1, 4, 2, 2),
        context=[torch.zeros(1)],
        point_count=5,
        utonia_feature_dim=7,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(42),
        num_inference_steps=50,
    )

    assert sample == "sample"
    assert calls[0]["initial_point_clouds"] is history
    assert calls[0]["utonia_features"] is features
    assert calls[0]["num_inference_steps"] == 50


def test_video_loss_can_train_wan_bridge_and_pc_optimizer_groups(monkeypatch):
    clean_latents = torch.zeros(1, 2, 1, 2, 2)
    flow = types.SimpleNamespace(
        model_input=clean_latents,
        latent_timesteps=torch.ones(1, 2),
        velocity_target=torch.zeros_like(clean_latents),
        loss_mask=torch.ones_like(clean_latents),
    )
    monkeypatch.setattr(joint_simgen_video_only, "_encode_simgen_videos", lambda *_: clean_latents)
    monkeypatch.setattr(joint_simgen_video_only, "make_flow_matching_batch", lambda *_args, **_kwargs: flow)
    monkeypatch.setattr(joint_simgen_video_only, "expand_latent_timesteps", lambda *args: args[0])
    monkeypatch.setattr(
        joint_simgen_video_only,
        "masked_velocity_mse",
        lambda prediction, *_args: prediction.sum(),
    )

    class ToyJointModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.wan_model = nn.Linear(1, 1, bias=False)
            self.bridges = nn.Linear(1, 1, bias=False)
            self.pc_model = nn.Linear(1, 1, bias=False)

        def forward(self, *, video_x, **_kwargs):
            video_offset = (
                self.wan_model.weight
                + self.bridges.weight
                + self.pc_model.weight
            ).reshape(1, 1, 1, 1)
            return [video_x[0] + video_offset], torch.empty(1)

    model = ToyJointModel()
    loss, _, _ = joint_simgen_video_only._video_only_losses(
        {"video": torch.zeros(1, 2, 3, 4, 4)},
        model,
        vae=object(),
        text_encoder=lambda prompts, device: [torch.zeros(1)],
        generator=torch.Generator().manual_seed(5),
        device="cpu",
        data={"num_points": 5, "synthetic_object_slots": 3},
        model_config={"utonia_feature_dim": 7},
        objective={"time_shift": 5.0, "num_train_timesteps": 1000},
    )
    optimizer = create_joint_optimizer(
        model,
        {
            name: {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0}
            for name in ("video", "bca", "pc")
        },
    )

    loss.backward()

    assert [group["name"] for group in optimizer.param_groups] == ["video", "bca", "pc"]
    assert model.wan_model.weight.grad is not None
    assert model.bridges.weight.grad is not None
    assert model.pc_model.weight.grad is not None


def test_sample_zero_visualization_writes_only_predicted_and_target_video(monkeypatch, tmp_path):
    writes = []
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(DDIMScheduler=lambda **_: object()))
    monkeypatch.setitem(
        sys.modules,
        "train_i2v_832x480",
        types.SimpleNamespace(
            save_visualization=lambda _vae, _latent, path, _fps: writes.append(path)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "wan.joint_pc_pipeline",
        types.SimpleNamespace(JointWanPhysCtrlPipeline=lambda *_args, **_kwargs: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "wan.utils.fm_solvers_unipc",
        types.SimpleNamespace(FlowUniPCMultistepScheduler=lambda **_: object()),
    )
    monkeypatch.setattr(
        joint_simgen_video_only,
        "_encode_simgen_videos",
        lambda *_: torch.zeros(1, 2, 5, 2, 2),
    )
    monkeypatch.setattr(
        joint_simgen_video_only,
        "_sample_video_only",
        lambda **_kwargs: types.SimpleNamespace(video_latent=torch.zeros(2, 5, 2, 2)),
    )
    monkeypatch.setattr(
        joint_simgen_video_only,
        "_save_simgen_target_video",
        lambda _video, path, _fps: writes.append(path),
    )
    config = {
        "objective": {"num_train_timesteps": 1000, "beta_schedule": "linear", "time_shift": 5.0},
        "training": {"seed": 42},
        "data": {"num_points": 5},
        "model": {"utonia_feature_dim": 7},
        "sampling": {"num_inference_steps": 50},
        "visualization": {"fps": 12},
    }

    joint_simgen_video_only._save_sample_zero_visualization(
        model=object(),
        vae=object(),
        text_encoder=lambda *_: [torch.zeros(1)],
        batch={"video": torch.zeros(1, 5, 3, 4, 4)},
        output_dir=tmp_path,
        step=1000,
        config=config,
        device="cpu",
    )

    destination = tmp_path / "visualizations" / "step_0001000"
    assert writes == [destination / "video.mp4", destination / "target.mp4"]
