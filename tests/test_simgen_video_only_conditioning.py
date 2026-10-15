import torch

from training.simgen_video_only_conditioning import (
    GeneratorStateBundle,
    make_video_only_generators,
    make_sampling_conditioning,
    make_training_conditioning,
)


def test_training_conditioning_uses_three_noise_slots_and_ddpm_frame_times():
    conditioning = make_training_conditioning(
        batch_size=2,
        point_count=5,
        utonia_feature_dim=7,
        num_train_timesteps=1_000,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(42),
    )

    assert conditioning.init_pc.shape == (2, 3, 4, 1, 5, 3)
    assert conditioning.noisy_future_state.shape == (2, 3, 45, 1, 5, 3)
    assert conditioning.utonia_features.shape == (2, 3, 5, 7)
    assert conditioning.frame_times.shape == (2, 3, 49)
    assert conditioning.frame_times[:, :, :4].eq(0).all()
    assert conditioning.frame_times[:, :, 4:].ge(0).all()
    assert conditioning.frame_times[:, :, 4:].lt(1_000).all()


def test_training_conditioning_is_reproducible_from_equal_seeded_generators():
    first = make_training_conditioning(
        batch_size=1,
        point_count=2,
        utonia_feature_dim=3,
        num_train_timesteps=10,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(9),
    )
    second = make_training_conditioning(
        batch_size=1,
        point_count=2,
        utonia_feature_dim=3,
        num_train_timesteps=10,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(9),
    )

    assert torch.equal(first.init_pc, second.init_pc)
    assert torch.equal(first.noisy_future_state, second.noisy_future_state)
    assert torch.equal(first.utonia_features, second.utonia_features)
    assert torch.equal(first.frame_times, second.frame_times)


def test_sampling_conditioning_returns_pipeline_ready_three_slot_noise():
    history, features = make_sampling_conditioning(
        point_count=5,
        utonia_feature_dim=7,
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(42),
    )

    assert history.shape == (3, 4, 5, 3)
    assert features.shape == (3, 5, 7)


def test_video_rng_stream_is_unaffected_by_synthetic_pc_rng_consumption():
    video_first, pc_first = make_video_only_generators("cpu", 42)
    video_second, _ = make_video_only_generators("cpu", 42)

    torch.randn(100, generator=pc_first)

    assert torch.equal(
        torch.randn(20, generator=video_first),
        torch.randn(20, generator=video_second),
    )


def test_generator_state_bundle_restores_the_training_rng_streams():
    video, pc = make_video_only_generators("cpu", 42)
    state = GeneratorStateBundle(video=video, pc=pc)
    checkpoint = state.state_dict()
    expected_video = torch.randn(5, generator=video)
    expected_pc = torch.randn(5, generator=pc)

    torch.randn(11, generator=video)
    torch.randn(11, generator=pc)
    state.load_state_dict(checkpoint)

    assert torch.equal(torch.randn(5, generator=video), expected_video)
    assert torch.equal(torch.randn(5, generator=pc), expected_pc)
