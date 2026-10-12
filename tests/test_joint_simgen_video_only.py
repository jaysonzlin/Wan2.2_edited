import types

import torch

import joint_simgen_video_only


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
