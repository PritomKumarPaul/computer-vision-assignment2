import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class Checkpoint1ContractTests(unittest.TestCase):
    def test_configs_use_only_train_manifests(self):
        for name in ("checkpoint1_resnet18.yaml", "checkpoint1_vit_tiny.yaml"):
            with (ROOT / "configs" / name).open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            data = config["data"]
            self.assertEqual(data["train_root"], "data/train")
            self.assertEqual(data["train_manifest"], "splits/train.csv")
            self.assertEqual(data["validation_manifest"], "splits/val.csv")
            self.assertNotIn("test", " ".join(map(str, data.values())).lower())

    def test_models_explicitly_use_random_initialization(self):
        for name in ("checkpoint1_resnet18.yaml", "checkpoint1_vit_tiny.yaml"):
            with (ROOT / "configs" / name).open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            self.assertEqual(config["model"]["initialization"], "random")

        factory = (ROOT / "src" / "models" / "factory.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("resnet18(weights=None", factory)
        self.assertNotIn("DEFAULT", factory)

    def test_clean_train_diagnostic_cannot_select_a_test_manifest(self):
        source = (ROOT / "scripts" / "evaluate_clean_train.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('ROOT / "splits" / "train.csv"', source)
        self.assertNotIn('add_argument("--manifest"', source)


if __name__ == "__main__":
    unittest.main()

