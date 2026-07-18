import unittest

import torch

from training.wan_i2v_training import make_flow_matching_batch, masked_velocity_mse


class WanI2VTrainingTests(unittest.TestCase):
    def test_first_latent_slot_stays_clean_and_has_no_loss(self):
        clean = torch.zeros(1, 16, 13, 2, 2)
        batch = make_flow_matching_batch(
            clean, torch.Generator().manual_seed(0), time_shift=5.0, num_train_timesteps=1000
        )

        self.assertTrue(torch.equal(batch.model_input[:, :, :1], clean[:, :, :1]))
        self.assertFalse(batch.loss_mask[:, :, :1].any())
        self.assertTrue(batch.loss_mask[:, :, 1:].all())
        self.assertTrue(torch.equal(batch.latent_timesteps[:, :1], torch.zeros(1, 1)))

    def test_masked_loss_ignores_the_conditioned_slot(self):
        prediction = torch.ones(1, 1, 2, 1, 1)
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[[[[0]], [[1]]]]], dtype=torch.float32)

        self.assertAlmostEqual(masked_velocity_mse(prediction, target, mask).item(), 1.0)
