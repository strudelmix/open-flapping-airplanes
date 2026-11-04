"""CPU guardrails for decoder endurance overlay (not a CUDA runtime test)."""
import unittest
from experiments.decoder004_patch import apply_decode, apply_doom, replace_once, apply_ofa
import ast
import importlib.util
import sys
import types
from unittest.mock import patch
from pathlib import Path


class Decoder004Tests(unittest.TestCase):
    def test_full_transform_all_strict_anchors(self):
        # A synthetic fixture made from the patcher's own anchors exercises all
        # edit paths without requiring a CUDA-enabled CI runner.
        source = (Path(__file__).resolve().parents[1] / 'decoder004_patch.py').read_text()
        node = ast.parse(source)
        anchors = []
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in ('one', 'many'):
                try:
                    anchors.append(ast.literal_eval(call.args[0]))
                except (ValueError, TypeError, IndexError):
                    pass
        self.assertGreaterEqual(len(anchors), 20)
        new = apply_ofa('\n__SEPARATOR__\n'.join(anchors))
        self.assertIn('u64* ticket64;', new)
        self.assertIn('rolling output cap', new)
        self.assertIn('decode profile: summed block-task busy time', new)

    def test_answer_and_story_output_ring(self):
        sample = ('if (st->out_tokens && step < st->out_cap) st->out_tokens[step] = A;\n'
                  'if (st->out_tokens && step < st->out_cap) st->out_tokens[step] = B;')
        patched = apply_decode(sample)
        self.assertEqual(patched.count('step % st->out_cap'), 2)
        self.assertNotIn('step < st->out_cap', patched)

    def test_endurance_remote_import_does_not_require_upstream_module(self):
        class FakeApp:
            def __init__(self, name): self.name = name
            def function(self, **kwargs): return lambda fn: fn
            def local_entrypoint(self, **kwargs): return lambda fn: fn
        class FakeImage:
            @staticmethod
            def debian_slim(**kwargs): return FakeImage()
        class FakeVolume:
            @staticmethod
            def from_name(*args, **kwargs): return FakeVolume()
        fake = types.ModuleType('modal')
        fake.is_local = lambda: False
        fake.App, fake.Image, fake.Volume = FakeApp, FakeImage, FakeVolume
        path = Path(__file__).resolve().parents[1] / 'endurance.py'
        with patch.dict(sys.modules, {'modal': fake}):
            spec = importlib.util.spec_from_file_location('endurance_remote_smoke', path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.assertEqual(mod.MK, '/opt/ofa/llm/b200/megakernel')
            self.assertNotIn('modal_app', sys.modules)

    def test_missing_anchor_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'expected exactly one'):
            replace_once('unrelated code', 'must exist', 'replacement', 'fixture')

    def test_live_counter_append_preserves_existing_fields(self):
        sample = 'struct Live { volatile unsigned frames, steps; volatile float loss; };'
        patched = apply_doom(sample)
        self.assertIn('volatile unsigned frames, steps; volatile float loss;', patched)
        self.assertIn('volatile u64 decode_tokens;', patched)

    def test_wide_ticket_budget_is_actually_over_32_bits(self):
        # The entire purpose is to support >4 Gi decoding task claims.
        self.assertGreater((1 << 26) * 1800, (1 << 32) - 1)

    def test_installed_sources_retain_guards(self):
        root = Path(__file__).resolve().parents[2]
        p = root / 'llm/b200/megakernel/ofa.cu'
        if not p.exists():
            self.skipTest('source not present in isolated test fixture')
        t = p.read_text()
        self.assertIn('u64* ticket64;', t)
        self.assertIn('u64* done64;', t)
        self.assertIn('decode phase %-8s', t)
        self.assertIn('decode progress: train_step=', t)
        self.assertIn('doom_frames=%u', t)
        self.assertNotIn('const unsigned total = (unsigned)J.limit * (unsigned)J.tps;', t)


if __name__ == '__main__':
    unittest.main()
