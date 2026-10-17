import types

import torch
from torch import nn

import simgen_vanilla_video


def test_vanilla_video_loss_calls_wan_without_pc_inputs(monkeypatch):
    clean_latents = torch.zeros(1, 2, 5, 2, 2)
    flow = types.SimpleNamespace(
        model_input=clean_latents,
        latent_timesteps=torch.ones(1, 5),
        velocity_target=torch.ones_like(clean_latents),
        loss_mask=torch.ones_like(clean_latents, dtype=torch.bool),
    )
    monkeypatch.setattr(simgen_vanilla_video, "_encode_simgen_videos", lambda *_: clean_latents)
    monkeypatch.setattr(simgen_vanilla_video, "make_flow_matching_batch", lambda *_args, **_kwargs: flow)
    monkeypatch.setattr(simgen_vanilla_video, "expand_latent_timesteps", lambda *args: args[0])
    monkeypatch.setattr(simgen_vanilla_video, "masked_velocity_mse", lambda *_args: torch.tensor(2.5))

    class Model:
        def __call__(self, x, t, context, seq_len):
            self.args = (x, t, context, seq_len)
            return [torch.zeros_like(clean_latents[0])]

    model = Model()
    loss, returned_latents = simgen_vanilla_video.vanilla_video_loss(
        {"video": torch.zeros(1, 5, 3, 4, 4)},
        model,
        vae=object(),
        text_encoder=lambda prompts, device: [torch.zeros(1)],
        generator=torch.Generator().manual_seed(5),
        device="cpu",
        objective={"time_shift": 5.0, "num_train_timesteps": 1000},
        history_frames=4,
    )

    assert loss.item() == 2.5
    assert returned_latents is clean_latents
    assert len(model.args) == 4
    assert len(model.args[0]) == 1
    torch.testing.assert_close(model.args[0][0], clean_latents[0])
    assert model.args[3] == 5


def test_vanilla_optimizer_contains_only_wan_parameters():
    model = nn.Linear(1, 1, bias=False)

    optimizer = simgen_vanilla_video.create_vanilla_optimizer(
        model,
        {"lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1},
    )

    assert [group["name"] for group in optimizer.param_groups] == ["video"]
    assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == {
        id(parameter) for parameter in model.parameters()
    }


def test_training_metrics_reduce_on_every_rank_before_rank_zero_logging():
    calls = []

    class Accelerator:
        def reduce(self, value, reduction):
            calls.append((value.item(), reduction))
            return value + 1

    metrics = simgen_vanilla_video.training_metrics(
        Accelerator(), torch.tensor(2.0), torch.tensor(3.0), 1e-5
    )

    assert calls == [(2.0, "mean"), (3.0, "mean")]
    assert metrics == {
        "train/video_loss": 3.0,
        "train/loss": 3.0,
        "train/video_gradient_norm": 4.0,
        "train/learning_rate": 1e-5,
    }


def test_periodic_checkpoint_prunes_to_the_configured_retention_limit(monkeypatch, tmp_path):
    calls = []

    class Accelerator:
        is_main_process = True

        def save_state(self, path):
            calls.append(("save", path))

    monkeypatch.setattr(
        simgen_vanilla_video,
        "prune_joint_checkpoints",
        lambda path, limit: calls.append(("prune", path, limit)),
    )

    simgen_vanilla_video.save_periodic_checkpoint(Accelerator(), tmp_path, 1000, 2)

    assert calls == [
        ("save", tmp_path / "checkpoint-1000"),
        ("prune", tmp_path, 2),
    ]


def test_final_validation_renderer_visits_every_configured_sample(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        simgen_vanilla_video,
        "render_video_sample",
        lambda **kwargs: calls.append((kwargs["batch"]["sample_id"], kwargs["generator"].initial_seed())),
    )
    dataset = [
        {"video": torch.zeros(5, 3, 4, 4), "sample_id": f"sample_{index}"}
        for index in range(490, 500)
    ]

    simgen_vanilla_video.render_validation_videos(
        model=object(), vae=object(), text_encoder=object(), validation_dataset=dataset,
        output_dir=tmp_path, config={"training": {"seed": 42}}, device="cpu",
    )

    assert [sample_id for sample_id, _ in calls] == [f"sample_{index}" for index in range(490, 500)]
    assert [seed for _, seed in calls] == list(range(42, 52))
