import tempfile
import unittest
from pathlib import Path
from experiments.analyze import aggregate, loss_file, parse_log, read_tokens, validate_loss, validate_tokens, write_results

SAMPLE = """SM partition (explicit): infer 24, doom 8, train 116, idle 0 of 148 SMs
GPU NVIDIA B200 (sm_100), 148 SMs
kernel time 8.000 s (event) / 8.400 s (wall incl. launch)
steady state: 12.50 ms/step (median 12.4) -> 327,680 tok/s;  4,000 steps = 50.0 s
decode: 12,345 tokens generated during 8.00 s of training -> 1,543 tok/s
decode answers: 48 answers (2 prompt tokens + 256 sampled each), 12,000 sampled tokens -> 1,500 tok/s sampled (prompt steps excluded)
doom: 280 frames in the run, 34.9 fps over the run (frame 0 = boot excluded); 35.0 fps while training steps 3..599
"""

class TestAnalysis(unittest.TestCase):
    def test_parse(self):
        a = parse_log(SAMPLE)
        self.assertEqual(a["train_tokens_s"], 327680)
        self.assertEqual(a["sampled_tokens_s"], 1500)
        self.assertEqual(a["doom_fps"], 35.0)
        self.assertEqual(a["doom_frames"], 280)
        self.assertEqual(a["kernel_seconds"], 8)
        self.assertIsNone(parse_log("irrelevant")["sampled_tokens_s"])

    def test_compare(self):
        self.assertEqual(validate_loss([1, 2], [1.0001, 2.0001])[0], True)
        self.assertFalse(validate_loss([1, 2], [1, 2, 3])[0])
        self.assertFalse(validate_tokens([1, 2], [1, 2], compare_n=3))
        self.assertTrue(validate_tokens(list(range(200)), list(range(150)), compare_n=128))
        self.assertFalse(validate_tokens(list(range(200)), [999] + list(range(149)), compare_n=128))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "loss"
            p.write_text("0 1.0\n1 2.0\n")
            self.assertEqual(loss_file(p), [1.0, 2.0])
            p.write_text("0 1.0\n4 2.0\n")
            with self.assertRaises(ValueError):
                loss_file(p)
            t = Path(d) / "tokens"
            t.write_text("1 2 3")
            self.assertEqual(read_tokens(t), [1, 2, 3])

    def test_rank(self):
        rows = []
        for c, tr, de in [("base", 300000, 1500), ("better", 306000, 1800), ("bad", 260000, 1900)]:
            rows += [dict(config=c, ok=True, train_tokens_s=tr, sampled_tokens_s=de, doom_fps=35) for _ in range(2)]
        base = dict(train_tokens_s=300000, sampled_tokens_s=1500)
        ranked = aggregate(rows, base)
        self.assertEqual(ranked[0]["config"], "better")
        self.assertFalse(next(r for r in ranked if r["config"] == "bad")["valid"])
        with tempfile.TemporaryDirectory() as d:
            report = write_results(Path(d), rows, ranked, ["test only"])
            self.assertIn("better", report)
            self.assertTrue((Path(d) / "results.csv").exists())

if __name__ == "__main__":
    unittest.main()
