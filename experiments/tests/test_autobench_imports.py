"""Regression test: remote Modal module import must not require b200-run/modal_app.py."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "autobench.py"


class FakeImage:
    @staticmethod
    def debian_slim(**kwargs):
        return FakeImage()


class FakeApp:
    def __init__(self, name):
        self.name = name

    def function(self, **kwargs):
        return lambda f: f

    def local_entrypoint(self, **kwargs):
        return lambda f: f


class FakeVolume:
    @staticmethod
    def from_name(*args, **kwargs):
        return FakeVolume()

    def reload(self):
        pass

    def commit(self):
        pass


class TestRemoteImports(unittest.TestCase):
    def test_import_with_modal_remote_context(self):
        fake = types.ModuleType("modal")
        fake.is_local = lambda: False
        fake.App = FakeApp
        fake.Image = FakeImage
        fake.Volume = FakeVolume
        with patch.dict(sys.modules, {"modal": fake}):
            spec = importlib.util.spec_from_file_location("autobench_remote_smoke", PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.MK, "/opt/ofa/llm/b200/megakernel")
            self.assertEqual(module.preflight()["status"], "ready")
            cmd = module.build_command(64, 100, 0, Path("/tmp/test"), greedy=True)
            self.assertIn("--no-opt", cmd)
            self.assertIn("--no-doom", cmd)
            self.assertEqual(module.SUITE_VERSION, "ofa001-v2")
            self.assertNotIn("modal_app", sys.modules)


if __name__ == "__main__":
    unittest.main()
