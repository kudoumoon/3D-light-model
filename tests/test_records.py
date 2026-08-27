import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RecordTests(unittest.TestCase):
    def test_source_manifest_hashes(self):
        manifest = json.loads((ROOT / "results/provenance.json").read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            value = hashlib.sha256((ROOT / entry["published"]).read_bytes()).hexdigest()
            self.assertEqual(value, entry["published_sha256"], entry["published"])

    def test_pair_count_and_unique_keys(self):
        data = json.loads((ROOT / "results/audited/quality_pairs.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["rows"]), 59)
        self.assertEqual(len(data["failures"]), 1)
        keys = {(r["track"], r["source_frame"], r["target_frame"]) for r in data["rows"]}
        self.assertEqual(len(keys), 59)

    def test_tile_oracle_includes_each_candidate(self):
        data = json.loads((ROOT / "results/audited/quality_pairs.json").read_text(encoding="utf-8"))
        for row in data["rows"]:
            for t in ["10", "20", "30", "40"]:
                oracle = row["tile_candidate_oracle_pass_fraction"][t]
                self.assertGreaterEqual(oracle + 1e-10, row["copy_safe_tile_fraction_of_full"][t])
                self.assertGreaterEqual(oracle + 1e-10, row["safe_tile_fraction_of_full"][t])


if __name__ == "__main__":
    unittest.main()
