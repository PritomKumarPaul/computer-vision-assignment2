import unittest
from pathlib import Path

import torch
import yaml

from src.models import build_encoder
from src.ssl_methods import DINO


ROOT = Path(__file__).resolve().parents[1]


class Checkpoint2ContractTests(unittest.TestCase):
    def test_four_pretraining_configs_form_two_by_two_matrix(self):
        observed = set()
        for path in ROOT.glob("configs/checkpoint2_*.yaml"):
            if "downstream" in path.name:
                continue
            config = yaml.safe_load(path.read_text())
            observed.add((config["method"], config["backbone"]))
            self.assertEqual(config["data"]["train_manifest"], "splits/train.csv")
            self.assertNotIn("test", " ".join(map(str, config["data"].values())).lower())
            self.assertEqual(config["training"]["epochs"], 200)
        self.assertEqual(observed, {
            ("moco", "resnet18"), ("moco", "vit_tiny"),
            ("dino", "resnet18"), ("dino", "vit_tiny"),
        })

    def test_runner_contains_eight_downstream_combinations(self):
        source = (ROOT / "run_checkpoint2.sh").read_text()
        self.assertIn("for method in moco dino", source)
        self.assertIn("for backbone in resnet18 vit_tiny", source)
        self.assertIn("linear_probe", source)
        self.assertIn("full_finetune", source)
        self.assertIn("visualize_ssl_features.py", source)

    def test_feature_visualization_is_validation_only(self):
        source = (ROOT / "scripts/visualize_ssl_features.py").read_text()
        self.assertIn('config["data"]["validation_manifest"]', source)
        self.assertNotIn('config["data"]["test', source)
        self.assertIn('"test_evaluated": False', source)

    def test_dino_teacher_head_can_be_created_and_matches_student(self):
        encoder, feature_dim = build_encoder("vit_tiny", image_size=224)
        model = DINO(
            encoder,
            feature_dim,
            hidden_dim=64,
            bottleneck_dim=32,
            output_dim=128,
            student_temperature=0.1,
            center_momentum=0.9,
        )
        student_state = model.student_head.state_dict()
        teacher_state = model.teacher_head.state_dict()
        self.assertEqual(student_state.keys(), teacher_state.keys())
        for name in student_state:
            self.assertTrue(torch.equal(student_state[name], teacher_state[name]), name)
        self.assertFalse(any(p.requires_grad for p in model.teacher_head.parameters()))


if __name__ == "__main__":
    unittest.main()
