"""Test device routing without importing torch or loading capture models.

Extract the real selector from the entry point; its module-level imports and
CUDA compatibility setup are unrelated to this routing contract.
"""
import ast
import contextlib
import io
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'save_mhr_data.py'


def select_device(*, cuda=False, mps=False, directml=False, force_cpu=False):
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_get_device')
    namespace = {
        'os': os,
        'torch': types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: cuda),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: mps))),
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(SOURCE), 'exec'), namespace)
    directml_module = types.SimpleNamespace(is_available=lambda: directml)
    with patch.dict(os.environ, {'BLENDCAP_FORCE_CPU': '1' if force_cpu else '0'}), \
            patch.dict('sys.modules', {'torch_directml': directml_module}), \
            contextlib.redirect_stdout(io.StringIO()):
        return namespace['_get_device']()


class CaptureDeviceTests(unittest.TestCase):
    def test_apple_rig_uses_cpu_for_float64(self):
        self.assertEqual(select_device(mps=True), 'cpu')

    def test_cuda_remains_preferred(self):
        self.assertEqual(select_device(cuda=True, mps=True), 'cuda')

    def test_explicit_cpu_overrides_cuda(self):
        self.assertEqual(select_device(cuda=True, force_cpu=True), 'cpu')

    def test_directml_keeps_cpu_hybrid(self):
        self.assertEqual(select_device(directml=True), 'cpu')

    def test_no_accelerator_uses_cpu(self):
        self.assertEqual(select_device(), 'cpu')


if __name__ == '__main__':
    unittest.main()
