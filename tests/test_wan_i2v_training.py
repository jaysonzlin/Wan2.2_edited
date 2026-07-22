import sys
import types
import unittest
from unittest.mock import patch

import torch

from training.wan_i2v_training import (
    apply_classifier_free_dropout,
    classifier_free_guidance,
    denoised_latent_mse,
    load_frozen_encoders,
    make_flow_matching_batch,
    masked_velocity_mse,
)


class WanI2VTrainingTests(unittest.TestCase):
    def test_cfg_scale_one_returns_the_conditional_velocity(self):
        unconditional = torch.tensor([[-2.0, 3.0]])
        conditional = torch.tensor([[4.0, -1.0]])

        result = classifier_free_guidance(unconditional, conditional, scale=1.0)

        self.assertTrue(torch.equal(result, conditional))

    def test_classifier_free_dropout_replaces_only_selected_contexts(self):
        conditional = [torch.tensor([[1.0]]), torch.tensor([[2.0]])]
        unconditional = [torch.tensor([[0.0]]), torch.tensor([[0.0]])]

        result = apply_classifier_free_dropout(
            conditional, unconditional, torch.tensor([False, True])
        )

        self.assertTrue(torch.equal(result[0], conditional[0]))
        self.assertTrue(torch.equal(result[1], unconditional[1]))

    def test_loader_freezes_torch_modules_inside_wan_wrappers(self):
        class InnerModule:
            def __init__(self):
                self.eval_called = False
                self.requires_grad_value = None

            def eval(self):
                self.eval_called = True
                return self

            def requires_grad_(self, value):
                self.requires_grad_value = value
                return self

        class Wrapper:
            def __init__(self, *args, **kwargs):
                self.model = InnerModule()

        fake_t5_module = types.ModuleType("wan.modules.t5")
        fake_t5_module.T5EncoderModel = Wrapper
        fake_vae_module = types.ModuleType("wan.modules.vae2_2")
        fake_vae_module.Wan2_2_VAE = Wrapper
        fake_modules = types.ModuleType("wan.modules")
        fake_wan = types.ModuleType("wan")
        fake_wan.modules = fake_modules

        config = types.SimpleNamespace(
            text_len=512,
            t5_dtype=torch.bfloat16,
            t5_checkpoint="t5.pth",
            t5_tokenizer="tokenizer",
            vae_checkpoint="vae.pth",
        )
        with patch.dict(
            sys.modules,
            {
                "wan": fake_wan,
                "wan.modules": fake_modules,
                "wan.modules.t5": fake_t5_module,
                "wan.modules.vae2_2": fake_vae_module,
            },
        ):
            vae, text_encoder = load_frozen_encoders("checkpoint", config, torch.device("cpu"))

        self.assertTrue(vae.model.eval_called)
        self.assertEqual(vae.model.requires_grad_value, False)
        self.assertTrue(text_encoder.model.eval_called)
        self.assertEqual(text_encoder.model.requires_grad_value, False)

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

    def test_denoised_latent_mse_excludes_the_first_latent_slot(self):
        prediction = torch.tensor([[[[[100.0]], [[3.0]], [[5.0]]]]])
        target = torch.tensor([[[[[0.0]], [[1.0]], [[2.0]]]]])

        result = denoised_latent_mse(prediction, target)

        self.assertAlmostEqual(result.item(), 6.5)

    def test_denoised_latent_mse_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            denoised_latent_mse(
                torch.zeros(1, 1, 2, 1, 1), torch.zeros(1, 1, 3, 1, 1)
            )

    def test_denoised_latent_mse_accepts_an_unbatched_visualization_latent(self):
        prediction = torch.tensor([[[[100.0]], [[3.0]], [[5.0]]]])
        target = torch.tensor([[[[0.0]], [[1.0]], [[2.0]]]])

        result = denoised_latent_mse(prediction, target)

        self.assertAlmostEqual(result.item(), 6.5)
