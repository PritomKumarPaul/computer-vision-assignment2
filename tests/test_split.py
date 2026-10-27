import csv
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_manifest(name):
    with (ROOT / "splits" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class SplitTests(unittest.TestCase):
    def test_split_sizes_and_balance(self):
        train = read_manifest("train.csv")
        val = read_manifest("val.csv")
        self.assertEqual(len(train), 1920)
        self.assertEqual(len(val), 480)
        self.assertEqual(set(Counter(row["class_name"] for row in train).values()), {120})
        self.assertEqual(set(Counter(row["class_name"] for row in val).values()), {30})

    def test_paths_and_hashes_do_not_cross_split(self):
        train = read_manifest("train.csv")
        val = read_manifest("val.csv")
        self.assertTrue(
            {row["relative_path"] for row in train}.isdisjoint(
                row["relative_path"] for row in val
            )
        )
        self.assertTrue(
            {row["sha256"] for row in train}.isdisjoint(
                row["sha256"] for row in val
            )
        )

    def test_every_manifest_path_exists(self):
        for row in read_manifest("train.csv") + read_manifest("val.csv"):
            self.assertTrue((ROOT / "data" / "train" / row["relative_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
