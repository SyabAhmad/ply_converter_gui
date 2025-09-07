import importlib.util
from pathlib import Path
import numpy as np
from plyfile import PlyElement, PlyData
import pytest

# load module by file path to avoid import-name issues
module_path = Path(__file__).resolve().parents[1] / "ply_converter_gui_4.0.py"
spec = importlib.util.spec_from_file_location("ply_converter_gui_mod", str(module_path))
ply_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ply_mod)


def _write_flower_ply(path):
    """Create a tiny 'flower' point set: center + 6 petals with RGB colors (0-255)."""
    pts = []
    # center (yellow)
    pts.append((0.0, 0.0, 0.0, 255, 220, 0))
    # petals around circle (reds/oranges)
    n_petals = 6
    radius = 0.05
    for i in range(n_petals):
        a = 2.0 * np.pi * i / n_petals
        x = radius * np.cos(a)
        y = radius * np.sin(a)
        z = 0.0
        r = int(200 + 55 * np.cos(a) * 0.2)  # slight variation
        g = int(80 + 40 * np.sin(a) * 0.2)
        b = int(30)
        pts.append((x, y, z, r, g, b))

    vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(path)


def test_flower_conversion_end_to_end(tmp_path, monkeypatch):
    # prepare files and output dir
    ply_path = tmp_path / "flower.ply"
    out_dir = tmp_path
    _write_flower_ply(str(ply_path))

    # Build small arrays used by fake_read_point_cloud
    positions = np.array([
        [0.0, 0.0, 0.0],
        *[[0.05 * np.cos(2*np.pi*i/6), 0.05 * np.sin(2*np.pi*i/6), 0.0] for i in range(6)]
    ], dtype=np.float32)
    colors_255 = np.array([
        [255, 220, 0],
        *[[200, 80, 30] for _ in range(6)]
    ], dtype=np.float32)
    colors_norm = (colors_255 / 255.0).astype(np.float32)

    # Stub Open3D read_point_cloud to return a minimal PointCloud with points+colors
    try:
        import open3d as o3d

        def fake_read_point_cloud(path):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)
            pcd.colors = o3d.utility.Vector3dVector(colors_norm)
            return pcd

        monkeypatch.setattr(ply_mod.o3d.io, "read_point_cloud", fake_read_point_cloud)
    except Exception:
        # If open3d not available, stub ConversionWorker to avoid using it at all below.
        pass

    # Replace heavy export methods with lightweight stubs that write small files
    def _stub_stl(self, pcd, out_path):
        with open(out_path, "wb") as f:
            f.write(b"STL_FLOWER")

    def _stub_glb(self, pcd, out_path):
        with open(out_path, "wb") as f:
            f.write(b"GLB_FLOWER")

    def _stub_3mf(self, pcd, out_path):
        with open(out_path, "wb") as f:
            f.write(b"3MF_FLOWER")

    def _stub_dxf(self, pcd, out_path):
        with open(out_path, "wb") as f:
            f.write(b"DXF_FLOWER")

    monkeypatch.setattr(ply_mod.ConversionWorker, "export_point_cloud_stl", _stub_stl)
    monkeypatch.setattr(ply_mod.ConversionWorker, "export_point_cloud_glb", _stub_glb)
    monkeypatch.setattr(ply_mod.ConversionWorker, "export_point_cloud_3mf", _stub_3mf)
    monkeypatch.setattr(ply_mod.ConversionWorker, "export_point_cloud_dxf", _stub_dxf)

    # Instantiate worker with our flower PLY and run synchronously
    worker = ply_mod.ConversionWorker(str(ply_path), str(out_dir))
    # call run() directly to avoid thread scheduling in test
    worker.run()

    # Assert outputs were created
    stl = out_dir / "conversion_output.stl"
    glb = out_dir / "conversion_output.glb"
    mf = out_dir / "conversion_output.3mf"
    dxf = out_dir / "conversion_output.dxf"

    assert stl.exists() and stl.read_bytes().startswith(b"STL_FLOWER")
    assert glb.exists() and glb.read_bytes().startswith(b"GLB_FLOWER")
    assert mf.exists() and mf.read_bytes().startswith(b"3MF_FLOWER")
    assert dxf.exists() and dxf.read_bytes().startswith(b"DXF_FLOWER")