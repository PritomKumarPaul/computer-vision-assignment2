import unittest
from pathlib import Path

import torch
import yaml

from src.training import mask_images, patchify, unpatchify


ROOT = Path(__file__).resolve().parents[1]


class Checkpoint3ContractTests(unittest.TestCase):
    def test_joint_mae_configs_are_matched_and_leakage_free(self):
        observed = set()
        for path in sorted(ROOT.glob("configs/checkpoint3_*.yaml")):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            observed.add(
                (
                    config["model"]["name"],
                    config["objective"]["classification_weight"],
                    config["objective"]["reconstruction_weight"],
                    config["objective"]["mask_ratio"],
                )
            )
            data_values = " ".join(map(str, config["data"].values())).lower()
            self.assertNotIn("test", data_values)
            self.assertEqual(config["model"]["initialization"], "random")
            self.assertAlmostEqual(
                config["objective"]["classification_weight"]
                + config["objective"]["reconstruction_weight"],
                1.0,
            )
            self.assertEqual(config["objective"]["patch_size"], 16)
            self.assertEqual(config["training"]["early_stopping_patience"], 20)
        expected = {
            (backbone, classification, reconstruction, mask)
            for backbone in ("resnet18", "vit_tiny")
            for classification, reconstruction in ((0.75, 0.25), (0.50, 0.50))
            for mask in (0.50, 0.75)
        }
        self.assertEqual(observed, expected)

    def test_patch_round_trip_and_mask_count(self):
        images = torch.randn(2, 3, 32, 32)
        patches = patchify(images, 16)
        restored = unpatchify(
            patches,
            patch_size=16,
            image_height=32,
            image_width=32,
        )
        self.assertTrue(torch.equal(images, restored))
        masked_images, targets, mask = mask_images(
            images, patch_size=16, mask_ratio=0.50
        )
        self.assertEqual(mask.sum(dim=1).tolist(), [2, 2])
        self.assertEqual(targets.shape, patches.shape)
        self.assertEqual(masked_images.shape, images.shape)

    def test_runner_contains_both_backbones_and_no_test_evaluation(self):
        source = (ROOT / "run_checkpoint3.sh").read_text(encoding="utf-8")
        self.assertIn("joint_mae_resnet18_seed0", source)
        self.assertIn("joint_mae_vit_tiny_seed0", source)
        self.assertNotIn("data/test", source)


if __name__ == "__main__":
    unittest.main()
