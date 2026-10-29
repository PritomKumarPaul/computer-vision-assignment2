import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FinalEvaluationContractTests(unittest.TestCase):
    def test_final_evaluation_is_locked_to_frozen_checkpoint(self):
        source = (ROOT / "scripts/evaluate_final_test.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("dino_resnet18_full_finetune_seed0", source)
        self.assertIn(
            "93bbf0877348a81b167ee5fe0bedb475e3ed9aeff0d0754c644ec6a5b8976d95",
            source,
        )
        self.assertIn("--confirm-final-selection", source)
        self.assertIn("refusing repeated evaluation", source)
        self.assertNotIn('add_argument("--checkpoint"', source)

    def test_reporting_comparison_is_locked_to_preselected_checkpoints(self):
        source = (ROOT / "scripts/evaluate_reporting_comparison.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("supervised_resnet18_seed0", source)
        self.assertIn("moco_resnet18_full_finetune_seed0", source)
        self.assertIn("joint_mae_vit_tiny_seed0", source)
        self.assertIn("--confirm-reporting-only", source)
        self.assertIn("no_further_model_selection_or_tuning", source)
        self.assertNotIn('add_argument("--checkpoint"', source)


if __name__ == "__main__":
    unittest.main()
