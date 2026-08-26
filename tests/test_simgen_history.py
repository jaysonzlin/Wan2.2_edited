import torch

from training.wan_i2v_training import make_flow_matching_batch


def test_flow_matching_masks_four_clean_history_slots():
    clean = torch.zeros(1, 2, 6, 2, 2)
    batch = make_flow_matching_batch(
        clean,
        torch.Generator().manual_seed(0),
        time_shift=5.0,
        num_train_timesteps=1000,
        history_frames=4,
    )

    assert batch.model_input[:, :, :4].eq(0).all()
    assert not batch.loss_mask[:, :, :4].any()
    assert batch.loss_mask[:, :, 4:].all()
    assert batch.latent_timesteps[:, :4].eq(0).all()
