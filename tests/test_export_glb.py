import numpy as np
import os
from pathlib import Path
import importlib.util, sys

module_path = Path(__file__).resolve().parents[1] / "ply_converter_gui_4.0.py"
spec = importlib.util.spec_from_file_location("ply_converter_gui_mod", str(module_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_export_simple_gaussian_glb_tmpfile(tmp_path, monkeypatch):
    out = tmp_path / "out.glb"
    conv = mod.ConversionWorker("in.ply", str(tmp_path))
    # small synthetic data
    positions = np.array([[0.0,0.0,0.0],[0.01,0.0,0.0]], dtype=np.float32)
    colors = np.ones((2,4), dtype=np.float32)
    # monkeypatch GLTF2.save_binary to just write a file so the method returns quickly
    def fake_save_binary(self, path):
        with open(path, "wb") as f:
            f.write(b"GLB")
    monkeypatch.setattr(mod.GLTF2, "save_binary", fake_save_binary)
    conv.export_simple_gaussian_glb(positions, colors, str(out))
    assert out.exists()
    assert out.stat().st_size > 0