import numpy as np
from pathlib import Path
import pytest
import importlib.util, sys
module_path = Path(__file__).resolve().parents[1] / "ply_converter_gui_4.0.py"
spec = importlib.util.spec_from_file_location("ply_converter_gui_mod", str(module_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_enhance_colors_vectorized_basic():
    arr = np.array([[255, 128, 0], [0, 128, 255]], dtype=np.float32)
    out = mod.ConversionWorker("in.ply", "out").enhance_colors_vectorized(arr)
    assert out.shape == (2, 3)
    assert out.max() <= 1.0 and out.min() >= 0.0

def test_create_ellipsoid_mesh_shapes():
    conv = mod.ConversionWorker("in.ply", "out")
    verts, faces = conv.create_ellipsoid_mesh(lat_segments=8, lon_segments=16)
    assert verts.shape[1] == 3
    assert faces.shape[1] == 3  # triangles
    assert len(verts) > 0 and len(faces) > 0

def test_transform_ellipsoid_identity_rotation():
    conv = mod.ConversionWorker("in.ply", "out")
    base = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity quat w,x,y,z
    t = conv.transform_ellipsoid(base, pos, scale, rotation)
    assert np.allclose(t[0], base[0])