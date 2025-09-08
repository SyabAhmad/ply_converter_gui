import sys
import os
import threading
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QProgressBar, QTextEdit, QMessageBox, QFrame, QSizePolicy)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QIcon # QPixmap - Removed as not used in this snippet
import open3d as o3d
import numpy as np
import trimesh
import ezdxf
import plyfile
from pygltflib import GLTF2, Scene, Node, Mesh, Buffer, BufferView, Accessor, Asset, Primitive

class ConversionWorker(QThread): 
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, input_path, output_dir,
                 keep_all_clusters=True,
                 disable_plane_removal=True,
                 disable_z_trim=True,
                 high_detail=True):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.keep_all_clusters = keep_all_clusters
        self.disable_plane_removal = disable_plane_removal
        self.disable_z_trim = disable_z_trim
        self.high_detail = high_detail
        # NEW: caches
        self._cached_pcd = None
        self._cached_surface_pcd = None
        self._cached_mesh = None

    def run(self):
        try:
            # Load / cache base point cloud once
            pcd = self._get_base_pcd()
            self.progress.emit(f"Loaded {len(pcd.points)} Gaussian splats")
            # Light optional downsample already done inside _get_base_pcd
            # Color enhancement already applied there if present
            # Build (once) surface mesh if needed by mesh formats
            surface_mesh = self._get_surface_mesh(pcd)
            # Exports reuse cached data (no repeated Poisson)
            self.progress.emit("Exporting to STL...")
            self.export_point_cloud_stl(pcd, os.path.join(self.output_dir, "conversion_output.stl"), mesh=surface_mesh)
            self.progress.emit("Exporting to GLB...")
            self.export_point_cloud_glb(pcd, os.path.join(self.output_dir, "conversion_output.glb"))
            self.progress.emit("Exporting to 3MF...")
            self.export_point_cloud_3mf(pcd, os.path.join(self.output_dir, "conversion_output.3mf"), mesh=surface_mesh)
            self.progress.emit("Exporting to DXF...")
            self.export_point_cloud_dxf(pcd, os.path.join(self.output_dir, "conversion_output.dxf"), mesh=surface_mesh)
            self.progress.emit("Conversion complete!")
            self.finished.emit(True, "Conversion completed successfully!")
        except Exception as e:
            self.finished.emit(False, f"Conversion failed: {str(e)}")

    def enhance_colors_vectorized(self, rgb_colors):
        """Preserve original colors without enhancement"""
        # Simply normalize colors to 0-1 range
        rgb = rgb_colors[:, :3]
        # Only normalize if values are above 1.0 (likely 0-255 range)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        # Ensure values stay in valid range
        rgb = np.clip(rgb, 0, 1)
        # Combine with original alpha if available
        if rgb_colors.shape[1] > 3:
            result = np.column_stack((rgb, rgb_colors[:, 3]))
        else:
            result = rgb
        return result

    def transfer_colors_to_mesh(self, pcd, mesh):
        """Transfer colors from point cloud to mesh vertices using nearest neighbor"""
        try:
            import open3d as o3d
            # Get point cloud points and colors
            pcd_points = np.asarray(pcd.points)
            pcd_colors = np.asarray(pcd.colors)
            # Get mesh vertices
            mesh_vertices = np.asarray(mesh.vertices)
            # Create a KDTree for efficient nearest neighbor search
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            # For each mesh vertex, find the nearest point cloud point and use its color
            mesh_colors = np.zeros((len(mesh_vertices), 3), dtype=np.float32)
            for i, vertex in enumerate(mesh_vertices):
                # Find nearest neighbor
                [k, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
                if len(idx) > 0:
                    mesh_colors[i] = pcd_colors[idx[0]]
            # Assign colors to mesh
            mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
            print(f"Transferred colors from {len(pcd_points)} points to {len(mesh_vertices)} mesh vertices")
            return mesh
        except Exception as e:
            print(f"Color transfer failed: {e}")
            return mesh

    def transfer_colors_to_mesh_enhanced(self, pcd, mesh, original_colors):
        """Enhanced color transfer with better color preservation and smoothing"""
        try:
            import open3d as o3d
            # Get point cloud points and colors
            pcd_points = np.asarray(pcd.points)
            mesh_vertices = np.asarray(mesh.vertices)
            # Create a KDTree for efficient nearest neighbor search
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            # For each mesh vertex, find multiple nearest neighbors and interpolate colors
            mesh_colors = np.zeros((len(mesh_vertices), 3), dtype=np.float32)
            for i, vertex in enumerate(mesh_vertices):
                # Find 5 nearest neighbors for better color interpolation
                [k, idx, dist] = pcd_tree.search_knn_vector_3d(vertex, 5)
                if len(idx) > 0:
                    # Weight colors by distance (closer points have more influence)
                    weights = 1.0 / (dist + 1e-6)  # Add small epsilon to avoid division by zero
                    weights = weights / np.sum(weights)  # Normalize weights
                    # Interpolate colors using weighted average
                    interpolated_color = np.zeros(3, dtype=np.float32)
                    for j, neighbor_idx in enumerate(idx):
                        interpolated_color += original_colors[neighbor_idx] * weights[j]
                    mesh_colors[i] = interpolated_color
            # Apply color enhancement to make colors more vivid (like real ham)
            enhanced_colors = self.enhance_colors_vectorized(mesh_colors)
            # Assign enhanced colors to mesh
            mesh.vertex_colors = o3d.utility.Vector3dVector(enhanced_colors)
            print(f"Enhanced color transfer completed: {len(mesh_vertices)} vertices with vivid colors")
            return mesh
        except Exception as e:
            print(f"Enhanced color transfer failed: {e}")
            # Fallback to simple color transfer
            return self.transfer_colors_to_mesh(pcd, mesh)

    def export_point_cloud_stl(self, pcd, output_path, mesh=None):
        """Export STL reusing cached mesh if provided."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                tri = trimesh.Trimesh(vertices=np.asarray(orig_mesh.vertices),
                                      faces=np.asarray(orig_mesh.triangles),
                                      process=True)
                tri.export(output_path, file_type='stl')
                return
            if mesh is None:
                mesh = self._get_surface_mesh(pcd)
            if mesh is None or len(mesh.triangles) == 0:
                print("No mesh for STL.")
                return
            tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                  faces=np.asarray(mesh.triangles),
                                  process=True)
            tri.export(output_path, file_type='stl')
        except Exception as e:
            print(f"STL export failed: {e}")

    def export_point_cloud_3mf(self, pcd, output_path, mesh=None):
        """Reuse cached mesh; skip cube fallback on large clouds."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                trimesh.Trimesh(np.asarray(orig_mesh.vertices),
                                np.asarray(orig_mesh.triangles),
                                process=True).export(output_path, file_type='3mf')
                return
            if mesh is None:
                mesh = self._get_surface_mesh(pcd)
            if mesh is None or len(mesh.triangles) == 0:
                print("No surface mesh; skipping 3MF (avoid huge cube fallback).")
                return
            tri_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                       faces=np.asarray(mesh.triangles),
                                       process=True)
            tri_mesh.export(output_path, file_type='3mf')
        except Exception as e:
            print(f"3MF export failed: {e}")

    def export_point_cloud_dxf(self, pcd, output_path, mesh=None):
        """Reuse cached mesh for DXF."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                verts = np.asarray(orig_mesh.vertices)
                faces = np.asarray(orig_mesh.triangles)
            else:
                if mesh is None:
                    mesh = self._get_surface_mesh(pcd)
                if mesh is None or len(mesh.triangles) == 0:
                    print("No surface mesh; DXF skipped.")
                    return
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
            import ezdxf
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            for f in faces:
                v1, v2, v3 = verts[f[0]], verts[f[1]], verts[f[2]]
                msp.add_3dface([
                    [float(v1[0]), float(v1[1]), float(v1[2])],
                    [float(v2[0]), float(v2[1]), float(v2[2])],
                    [float(v3[0]), float(v3[1]), float(v3[2])],
                    [float(v1[0]), float(v1[1]), float(v1[2])]
                ])
            doc.saveas(output_path)
        except Exception as e:
            print(f"DXF export failed: {e}")

    def export_point_cloud_glb(self, pcd, output_path):
        """If mesh already cached, reuse it for surface GLB; else fallback to point cloud."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                trimesh.Trimesh(vertices=np.asarray(orig_mesh.vertices),
                                faces=np.asarray(orig_mesh.triangles),
                                process=True).export(output_path, file_type='glb')
                return
            mesh = self._cached_mesh  # may already exist
            if mesh is not None and len(mesh.triangles) > 0:
                tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                      faces=np.asarray(mesh.triangles),
                                      process=True)
                tri.export(output_path, file_type='glb')
                return
            # Fallback: simple point cloud GLB
            positions = np.asarray(pcd.points).astype(np.float32)
            if pcd.has_colors():
                cols = np.asarray(pcd.colors).astype(np.float32)
                if cols.max() > 1.0: cols /= 255.0
            else:
                cols = np.ones((len(positions), 3), dtype=np.float32)
            colors_rgba = np.column_stack((cols, np.ones(len(cols), dtype=np.float32)))
            self.export_simple_gaussian_glb(positions, colors_rgba, output_path)
        except Exception as e:
            print(f"GLB export failed: {e}")

    def enhance_colors_vectorized(self, rgb_colors):
        """Preserve original colors without enhancement"""
        # Simply normalize colors to 0-1 range
        rgb = rgb_colors[:, :3]
        # Only normalize if values are above 1.0 (likely 0-255 range)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        # Ensure values stay in valid range
        rgb = np.clip(rgb, 0, 1)
        # Combine with original alpha if available
        if rgb_colors.shape[1] > 3:
            result = np.column_stack((rgb, rgb_colors[:, 3]))
        else:
            result = rgb
        return result

    def transfer_colors_to_mesh(self, pcd, mesh):
        """Transfer colors from point cloud to mesh vertices using nearest neighbor"""
        try:
            import open3d as o3d
            # Get point cloud points and colors
            pcd_points = np.asarray(pcd.points)
            pcd_colors = np.asarray(pcd.colors)
            # Get mesh vertices
            mesh_vertices = np.asarray(mesh.vertices)
            # Create a KDTree for efficient nearest neighbor search
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            # For each mesh vertex, find the nearest point cloud point and use its color
            mesh_colors = np.zeros((len(mesh_vertices), 3), dtype=np.float32)
            for i, vertex in enumerate(mesh_vertices):
                # Find nearest neighbor
                [k, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
                if len(idx) > 0:
                    mesh_colors[i] = pcd_colors[idx[0]]
            # Assign colors to mesh
            mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
            print(f"Transferred colors from {len(pcd_points)} points to {len(mesh_vertices)} mesh vertices")
            return mesh
        except Exception as e:
            print(f"Color transfer failed: {e}")
            return mesh

    def transfer_colors_to_mesh_enhanced(self, pcd, mesh, original_colors):
        """Enhanced color transfer with better color preservation and smoothing"""
        try:
            import open3d as o3d
            # Get point cloud points and colors
            pcd_points = np.asarray(pcd.points)
            mesh_vertices = np.asarray(mesh.vertices)
            # Create a KDTree for efficient nearest neighbor search
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            # For each mesh vertex, find multiple nearest neighbors and interpolate colors
            mesh_colors = np.zeros((len(mesh_vertices), 3), dtype=np.float32)
            for i, vertex in enumerate(mesh_vertices):
                # Find 5 nearest neighbors for better color interpolation
                [k, idx, dist] = pcd_tree.search_knn_vector_3d(vertex, 5)
                if len(idx) > 0:
                    # Weight colors by distance (closer points have more influence)
                    weights = 1.0 / (dist + 1e-6)  # Add small epsilon to avoid division by zero
                    weights = weights / np.sum(weights)  # Normalize weights
                    # Interpolate colors using weighted average
                    interpolated_color = np.zeros(3, dtype=np.float32)
                    for j, neighbor_idx in enumerate(idx):
                        interpolated_color += original_colors[neighbor_idx] * weights[j]
                    mesh_colors[i] = interpolated_color
            # Apply color enhancement to make colors more vivid (like real ham)
            enhanced_colors = self.enhance_colors_vectorized(mesh_colors)
            # Assign enhanced colors to mesh
            mesh.vertex_colors = o3d.utility.Vector3dVector(enhanced_colors)
            print(f"Enhanced color transfer completed: {len(mesh_vertices)} vertices with vivid colors")
            return mesh
        except Exception as e:
            print(f"Enhanced color transfer failed: {e}")
            # Fallback to simple color transfer
            return self.transfer_colors_to_mesh(pcd, mesh)

    def export_point_cloud_stl(self, pcd, output_path, mesh=None):
        """Export STL reusing cached mesh if provided."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                tri = trimesh.Trimesh(vertices=np.asarray(orig_mesh.vertices),
                                      faces=np.asarray(orig_mesh.triangles),
                                      process=True)
                tri.export(output_path, file_type='stl')
                return
            if mesh is None:
                mesh = self._get_surface_mesh(pcd)
            if mesh is None or len(mesh.triangles) == 0:
                print("No mesh for STL.")
                return
            tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                  faces=np.asarray(mesh.triangles),
                                  process=True)
            tri.export(output_path, file_type='stl')
        except Exception as e:
            print(f"STL export failed: {e}")

    def export_point_cloud_3mf(self, pcd, output_path, mesh=None):
        """Reuse cached mesh; skip cube fallback on large clouds."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                trimesh.Trimesh(np.asarray(orig_mesh.vertices),
                                np.asarray(orig_mesh.triangles),
                                process=True).export(output_path, file_type='3mf')
                return
            if mesh is None:
                mesh = self._get_surface_mesh(pcd)
            if mesh is None or len(mesh.triangles) == 0:
                print("No surface mesh; skipping 3MF (avoid huge cube fallback).")
                return
            tri_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                       faces=np.asarray(mesh.triangles),
                                       process=True)
            tri_mesh.export(output_path, file_type='3mf')
        except Exception as e:
            print(f"3MF export failed: {e}")

    def export_point_cloud_dxf(self, pcd, output_path, mesh=None):
        """Reuse cached mesh for DXF."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                verts = np.asarray(orig_mesh.vertices)
                faces = np.asarray(orig_mesh.triangles)
            else:
                if mesh is None:
                    mesh = self._get_surface_mesh(pcd)
                if mesh is None or len(mesh.triangles) == 0:
                    print("No surface mesh; DXF skipped.")
                    return
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
            import ezdxf
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            for f in faces:
                v1, v2, v3 = verts[f[0]], verts[f[1]], verts[f[2]]
                msp.add_3dface([
                    [float(v1[0]), float(v1[1]), float(v1[2])],
                    [float(v2[0]), float(v2[1]), float(v2[2])],
                    [float(v3[0]), float(v3[1]), float(v3[2])],
                    [float(v1[0]), float(v1[1]), float(v1[2])]
                ])
            doc.saveas(output_path)
        except Exception as e:
            print(f"DXF export failed: {e}")

    def export_point_cloud_glb(self, pcd, output_path):
        """If mesh already cached, reuse it for surface GLB; else fallback to point cloud."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                trimesh.Trimesh(vertices=np.asarray(orig_mesh.vertices),
                                faces=np.asarray(orig_mesh.triangles),
                                process=True).export(output_path, file_type='glb')
                return
            mesh = self._cached_mesh  # may already exist
            if mesh is not None and len(mesh.triangles) > 0:
                tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                      faces=np.asarray(mesh.triangles),
                                      process=True)
                tri.export(output_path, file_type='glb')
                return
            # Fallback: simple point cloud GLB
            positions = np.asarray(pcd.points).astype(np.float32)
            if pcd.has_colors():
                cols = np.asarray(pcd.colors).astype(np.float32)
                if cols.max() > 1.0: cols /= 255.0
            else:
                cols = np.ones((len(positions), 3), dtype=np.float32)
            colors_rgba = np.column_stack((cols, np.ones(len(cols), dtype=np.float32)))
            self.export_simple_gaussian_glb(positions, colors_rgba, output_path)
        except Exception as e:
            print(f"GLB export failed: {e}")

    def _get_base_pcd(self):
        """Load & lightly preprocess once."""
        if self._cached_pcd is not None:
            return self._cached_pcd
        self.progress.emit("Loading point cloud...")
        pcd = o3d.io.read_point_cloud(self.input_path)
        if len(pcd.points) > 200000:  # adaptive quick downsample
            vs = max(0.001, float(np.ptp(np.asarray(pcd.points), axis=0).max()) * 0.003)
            pcd = pcd.voxel_down_sample(vs)
            self.progress.emit(f"Downsampled to {len(pcd.points)} points (voxel={vs:.4g})")
        if pcd.has_colors():
            cols = np.asarray(pcd.colors)
            if cols.max() > 1.0:
                cols = cols / 255.0
            pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
        self._cached_pcd = pcd
        return pcd

    def _get_surface_mesh(self, pcd):
        """Build Poisson surface once (cached)."""
        if self._cached_mesh is not None:
            return self._cached_mesh
        orig_mesh = self._load_original_mesh_if_present()
        if orig_mesh is not None and len(orig_mesh.triangles) > 0:
            self._cached_mesh = orig_mesh
            return self._cached_mesh
        self.progress.emit("Preparing surface reconstruction...")
        prep = self._prepare_point_cloud_for_reconstruction(pcd)
        if not prep.has_normals():
            prep.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=60))
        # Adaptive Poisson depth
        n = len(prep.points)
        if n < 50000:
            depth = 8
        elif n < 200000:
            depth = 9
        else:
            depth = 10 if not self.high_detail else 11
        self.progress.emit(f"Running Poisson (depth={depth}) once...")
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(prep, depth=depth, scale=1.02)
        # Light cleanup (skip heavy per‑export)
        mesh.remove_unreferenced_vertices()
        if len(mesh.vertices) > 0:
            mesh.compute_vertex_normals()
        self._cached_surface_pcd = prep
        self._cached_mesh = mesh
        return mesh

    def export_point_cloud_stl(self, pcd, output_path, mesh=None):
        """Export STL reusing cached mesh if provided."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                tri = trimesh.Trimesh(vertices=np.asarray(orig_mesh.vertices),
                                      faces=np.asarray(orig_mesh.triangles),
                                      process=True)
                tri.export(output_path, file_type='stl')
                return
            if mesh is None:
                mesh = self._get_surface_mesh(pcd)
            if mesh is None or len(mesh.triangles) == 0:
                print("No mesh for STL.")
                return
            tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                  faces=np.asarray(mesh.triangles),
                                  process=True)
            tri.export(output_path, file_type='stl')
        except Exception as e:
            print(f"STL export failed: {e}")

    def export_point_cloud_3mf(self, pcd, output_path, mesh=None):
        """Reuse cached mesh; skip cube fallback on large clouds."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                trimesh.Trimesh(np.asarray(orig_mesh.vertices),
                                np.asarray(orig_mesh.triangles),
                                process=True).export(output_path, file_type='3mf')
                return
            if mesh is None:
                mesh = self._get_surface_mesh(pcd)
            if mesh is None or len(mesh.triangles) == 0:
                print("No surface mesh; skipping 3MF (avoid huge cube fallback).")
                return
            tri_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                       faces=np.asarray(mesh.triangles),
                                       process=True)
            tri_mesh.export(output_path, file_type='3mf')
        except Exception as e:
            print(f"3MF export failed: {e}")

    def export_point_cloud_dxf(self, pcd, output_path, mesh=None):
        """Reuse cached mesh for DXF."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                verts = np.asarray(orig_mesh.vertices)
                faces = np.asarray(orig_mesh.triangles)
            else:
                if mesh is None:
                    mesh = self._get_surface_mesh(pcd)
                if mesh is None or len(mesh.triangles) == 0:
                    print("No surface mesh; DXF skipped.")
                    return
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
            import ezdxf
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            for f in faces:
                v1, v2, v3 = verts[f[0]], verts[f[1]], verts[f[2]]
                msp.add_3dface([
                    [float(v1[0]), float(v1[1]), float(v1[2])],
                    [float(v2[0]), float(v2[1]), float(v2[2])],
                    [float(v3[0]), float(v3[1]), float(v3[2])],
                    [float(v1[0]), float(v1[1]), float(v1[2])]
                ])
            doc.saveas(output_path)
        except Exception as e:
            print(f"DXF export failed: {e}")

    def export_point_cloud_glb(self, pcd, output_path):
        """If mesh already cached, reuse it for surface GLB; else fallback to point cloud."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                trimesh.Trimesh(vertices=np.asarray(orig_mesh.vertices),
                                faces=np.asarray(orig_mesh.triangles),
                                process=True).export(output_path, file_type='glb')
                return
            mesh = self._cached_mesh  # may already exist
            if mesh is not None and len(mesh.triangles) > 0:
                tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                      faces=np.asarray(mesh.triangles),
                                      process=True)
                tri.export(output_path, file_type='glb')
                return
            # Fallback: simple point cloud GLB
            positions = np.asarray(pcd.points).astype(np.float32)
            if pcd.has_colors():
                cols = np.asarray(pcd.colors).astype(np.float32)
                if cols.max() > 1.0: cols /= 255.0
            else:
                cols = np.ones((len(positions), 3), dtype=np.float32)
            colors_rgba = np.column_stack((cols, np.ones(len(cols), dtype=np.float32)))
            self.export_simple_gaussian_glb(positions, colors_rgba, output_path)
        except Exception as e:
            print(f"GLB export failed: {e}")

    def enhance_colors_vectorized(self, rgb_colors):
        """Preserve original colors without enhancement"""
        # Simply normalize colors to 0-1 range
        rgb = rgb_colors[:, :3]
        # Only normalize if values are above 1.0 (likely 0-255 range)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        # Ensure values stay in valid range
        rgb = np.clip(rgb, 0, 1)
        # Combine with original alpha if available
        if rgb_colors.shape[1] > 3:
            result = np.column_stack((rgb, rgb_colors[:, 3]))
        else:
            result = rgb
        return result

    def transfer_colors_to_mesh(self, pcd, mesh):
        """Transfer colors from point cloud to mesh vertices using nearest neighbor"""
        try:
            import open3d as o3d
            # Get point cloud points and colors
            pcd_points = np.asarray(pcd.points)
            pcd_colors = np.asarray(pcd.colors)
            # Get mesh vertices
            mesh_vertices = np.asarray(mesh.vertices)
            # Create a KDTree for efficient nearest neighbor search
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            # For each mesh vertex, find the nearest point cloud point and use its color
            mesh_colors = np.zeros((len(mesh_vertices), 3), dtype=np.float32)
            for i, vertex in enumerate(mesh_vertices):
                # Find nearest neighbor
                [k, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
                if len(idx) > 0:
                    mesh_colors[i] = pcd_colors[idx[0]]
            # Assign colors to mesh
            mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
            print(f"Transferred colors from {len(pcd_points)} points to {len(mesh_vertices)} mesh vertices")
            return mesh
        except Exception as e:
            print(f"Color transfer failed: {e}")
            return mesh

    def transfer_colors_to_mesh_enhanced(self, pcd, mesh, original_colors):
        """Enhanced color transfer with better color preservation and smoothing"""
        try:
            import open3d as o3d
            # Get point cloud points and colors
            pcd_points = np.asarray(pcd.points)
            mesh_vertices = np.asarray(mesh.vertices)
            # Create a KDTree for efficient nearest neighbor search
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            # For each mesh vertex, find multiple nearest neighbors and interpolate colors
            mesh_colors = np.zeros((len(mesh_vertices), 3), dtype=np.float32)
            for i, vertex in enumerate(mesh_vertices):
                # Find 5 nearest neighbors for better color interpolation
                [k, idx, dist] = pcd_tree.search_knn_vector_3d(vertex, 5)
                if len(idx) > 0:
                    # Weight colors by distance (closer points have more influence)
                    weights = 1.0 / (dist + 1e-6)  # Add small epsilon to avoid division by zero
                    weights = weights / np.sum(weights)  # Normalize weights
                    # Interpolate colors using weighted average
                    interpolated_color = np.zeros(3, dtype=np.float32)
                    for j, neighbor_idx in enumerate(idx):
                        interpolated_color += original_colors[neighbor_idx] * weights[j]
                    mesh_colors[i] = interpolated_color
            # Apply color enhancement to make colors more vivid (like real ham)
            enhanced_colors = self.enhance_colors_vectorized(mesh_colors)
            # Assign enhanced colors to mesh
            mesh.vertex_colors = o3d.utility.Vector3dVector(enhanced_colors)
            print(f"Enhanced color transfer completed: {len(mesh_vertices)} vertices with vivid colors")
            return mesh
        except Exception as e:
            print(f"Enhanced color transfer failed: {e}")
            # Fallback to simple color transfer
            return self.transfer_colors_to_mesh(pcd, mesh)

    def export_point_cloud_stl(self, pcd, output_path, mesh=None):
        """Export STL reusing cached mesh if provided."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                tri = trimesh.Trimesh(vertices=np.asarray(orig_mesh.vertices),
                                      faces=np.asarray(orig_mesh.triangles),
                                      process=True)
                tri.export(output_path, file_type='stl')
                return
            if mesh is None:
                mesh = self._get_surface_mesh(pcd)
            if mesh is None or len(mesh.triangles) == 0:
                print("No mesh for STL.")
                return
            tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                  faces=np.asarray(mesh.triangles),
                                  process=True)
            tri.export(output_path, file_type='stl')
        except Exception as e:
            print(f"STL export failed: {e}")

    def export_point_cloud_3mf(self, pcd, output_path, mesh=None):
        """Reuse cached mesh; skip cube fallback on large clouds."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                trimesh.Trimesh(np.asarray(orig_mesh.vertices),
                                np.asarray(orig_mesh.triangles),
                                process=True).export(output_path, file_type='3mf')
                return
            if mesh is None:
                mesh = self._get_surface_mesh(pcd)
            if mesh is None or len(mesh.triangles) == 0:
                print("No surface mesh; skipping 3MF (avoid huge cube fallback).")
                return
            tri_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                       faces=np.asarray(mesh.triangles),
                                       process=True)
            tri_mesh.export(output_path, file_type='3mf')
        except Exception as e:
            print(f"3MF export failed: {e}")

    def export_point_cloud_dxf(self, pcd, output_path, mesh=None):
        """Reuse cached mesh for DXF."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                verts = np.asarray(orig_mesh.vertices)
                faces = np.asarray(orig_mesh.triangles)
            else:
                if mesh is None:
                    mesh = self._get_surface_mesh(pcd)
                if mesh is None or len(mesh.triangles) == 0:
                    print("No surface mesh; DXF skipped.")
                    return
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
            import ezdxf
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            for f in faces:
                v1, v2, v3 = verts[f[0]], verts[f[1]], verts[f[2]]
                msp.add_3dface([
                    [float(v1[0]), float(v1[1]), float(v1[2])],
                    [float(v2[0]), float(v2[1]), float(v2[2])],
                    [float(v3[0]), float(v3[1]), float(v3[2])],
                    [float(v1[0]), float(v1[1]), float(v1[2])]
                ])
            doc.saveas(output_path)
        except Exception as e:
            print(f"DXF export failed: {e}")

    def export_point_cloud_glb(self, pcd, output_path):
        """If mesh already cached, reuse it for surface GLB; else fallback to point cloud."""
        try:
            orig_mesh = self._load_original_mesh_if_present()
            if orig_mesh is not None:
                trimesh.Trimesh(vertices=np.asarray(orig_mesh.vertices),
                                faces=np.asarray(orig_mesh.triangles),
                                process=True).export(output_path, file_type='glb')
                return
            mesh = self._cached_mesh  # may already exist
            if mesh is not None and len(mesh.triangles) > 0:
                tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                      faces=np.asarray(mesh.triangles),
                                      process=True)
                tri.export(output_path, file_type='glb')
                return
            # Fallback: simple point cloud GLB
            positions = np.asarray(pcd.points).astype(np.float32)
            if pcd.has_colors():
                cols = np.asarray(pcd.colors).astype(np.float32)
                if cols.max() > 1.0: cols /= 255.0
            else:
                cols = np.ones((len(positions), 3), dtype=np.float32)
            colors_rgba = np.column_stack((cols, np.ones(len(cols), dtype=np.float32)))
            self.export_simple_gaussian_glb(positions, colors_rgba, output_path)
        except Exception as e:
            print(f"GLB export failed: {e}")

    def create_ellipsoid_mesh(self, lat_segments, lon_segments):
        """Create a high-quality base ellipsoid mesh (unit sphere) with smooth surface"""
        vertices = []
        faces = []
        normals = []
        # Generate vertices with better distribution
        for lat in range(lat_segments + 1):
            # Use cosine distribution for better vertex spacing
            theta = lat * np.pi / lat_segments
            for lon in range(lon_segments + 1):
                phi = lon * 2 * np.pi / lon_segments
                # Create smooth ellipsoid shape
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                # No random variation - preserve exact mathematical shape
                vertices.append([x, y, z])
                # Calculate normal for smooth shading
                normal = np.array([x, y, z])
                normal = normal / np.linalg.norm(normal)
                normals.append(normal)
        # Generate faces with proper winding
        for lat in range(lat_segments):
            for lon in range(lon_segments):
                # Calculate vertex indices
                v0 = lat * (lon_segments + 1) + lon
                v1 = v0 + 1
                v2 = (lat + 1) * (lon_segments + 1) + lon
                v3 = v2 + 1
                # Add two triangles to form a quad with proper winding
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)

    def transform_ellipsoid(self, base_vertices, position, scale, rotation):
        """Transform base ellipsoid vertices by position, scale, and rotation without modifications"""
        # Apply scale without any adjustments
        scaled_vertices = base_vertices * scale
        # Apply rotation (quaternion)
        # Convert quaternion to rotation matrix
        w, x, y, z = rotation
        rotation_matrix = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*x*z+2*w*y],
            [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2-w*x],
            [2*x*z-2*w*y,   2*y*z+2*w*x,   1-2*x*x-2*y*y]
        ])
        # Apply rotation
        rotated_vertices = scaled_vertices @ rotation_matrix.T
        # Apply translation without position variation
        transformed_vertices = rotated_vertices + position
        return transformed_vertices

    def export_simple_gaussian_glb(self, positions, colors_rgba, output_path):
        """Fallback: Export Gaussian splats as simple points with colors"""
        try:
            print("Fallback: Exporting as simple point cloud GLB...")
            # Build binary blob
            positions_float32 = positions.astype(np.float32)
            colors_float32 = colors_rgba.astype(np.float32)
            bin_blob = positions_float32.tobytes() + colors_float32.tobytes()
            # Buffer definitions
            buffer = Buffer(byteLength=len(bin_blob))
            bv_positions = BufferView(buffer=0, byteOffset=0, byteLength=positions_float32.nbytes, target=34962)
            bv_colors = BufferView(buffer=0, byteOffset=positions_float32.nbytes, byteLength=colors_float32.nbytes, target=34962)
            # Accessors
            acc_positions = Accessor(bufferView=0, byteOffset=0, componentType=5126, count=len(positions), type="VEC3")
            acc_colors = Accessor(bufferView=1, byteOffset=0, componentType=5126, count=len(colors_rgba), type="VEC4")
            # Primitive with POINTS mode
            prim = Primitive(attributes={"POSITION": 0, "COLOR_0": 1}, mode=0)  # POINTS
            mesh = Mesh(primitives=[prim])
            node = Node(mesh=0)
            scene = Scene(nodes=[0])
            # Build GLTF object
            gltf = GLTF2(
                asset=Asset(version="2.0"),
                scenes=[scene],
                nodes=[node],
                meshes=[mesh],
                buffers=[buffer],
                bufferViews=[bv_positions, bv_colors],
                accessors=[acc_positions, acc_colors]
            )
            # Attach binary data and save
            gltf.set_binary_blob(bin_blob)
            gltf.save_binary(output_path)
            print(f"Fallback point cloud GLB exported: {output_path}")
        except Exception as e:
            print(f"Fallback GLB export also failed: {e}")

    def smooth_mesh_for_ham(self, mesh):
        """Apply specialized smoothing to make the mesh look more like real ham"""
        try:
            # Convert to Open3D mesh for better smoothing
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh.vertex_colors)
            # Apply multiple smoothing passes for organic appearance
            print("Applying organic smoothing passes...")
            # First pass: light smoothing to reduce sharp edges
            o3d_mesh = o3d_mesh.filter_smooth_simple(number_of_iterations=2)
            # Second pass: Laplacian smoothing for more natural curves
            o3d_mesh = o3d_mesh.filter_smooth_laplacian(number_of_iterations=3)
            # Third pass: Taubin smoothing to prevent over-smoothing
            o3d_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=2)
            # Recompute normals for better lighting
            o3d_mesh.compute_vertex_normals()
            # Convert back to trimesh
            smoothed_mesh = trimesh.Trimesh(
                vertices=np.asarray(o3d_mesh.vertices),
                faces=np.asarray(o3d_mesh.triangles),
                vertex_colors=np.asarray(o3d_mesh.vertex_colors)
            )
            print("Mesh smoothing completed for realistic ham appearance")
            return smoothed_mesh
        except Exception as e:
            print(f"Mesh smoothing failed: {e}")
            return mesh

    def smooth_mesh_for_ham_o3d(self, o3d_mesh):
        """Apply specialized smoothing to Open3D mesh for realistic ham appearance"""
        try:
            print("Applying organic smoothing passes to Open3D mesh...")
            # First pass: light smoothing to reduce sharp edges
            o3d_mesh = o3d_mesh.filter_smooth_simple(number_of_iterations=3)
            # Second pass: Laplacian smoothing for more natural curves
            o3d_mesh = o3d_mesh.filter_smooth_laplacian(number_of_iterations=4)
            # Third pass: Taubin smoothing to prevent over-smoothing
            o3d_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=3)
            # Recompute normals for better lighting
            o3d_mesh.compute_vertex_normals()
            print("Open3D mesh smoothing completed for realistic ham appearance")
            return o3d_mesh
        except Exception as e:
            print(f"Open3D mesh smoothing failed: {e}")
            return o3d_mesh

    def transfer_colors_to_surface_mesh(self, pcd, mesh, colors_rgba):
        """Transfer colors from point cloud to surface mesh vertices using nearest neighbor"""
        try:
            # Get point cloud points and colors
            pcd_points = np.asarray(pcd.points)
            mesh_vertices = np.asarray(mesh.vertices)
            # Create a KDTree for efficient nearest neighbor search
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            # For each mesh vertex, find the nearest point cloud point and use its color
            mesh_colors = np.zeros((len(mesh_vertices), 3), dtype=np.float32)
            for i, vertex in enumerate(mesh_vertices):
                # Find nearest neighbor
                [k, idx, dist] = pcd_tree.search_knn_vector_3d(vertex, 1)
                if len(idx) > 0:
                    # Get color from nearest point
                    mesh_colors[i] = colors_rgba[idx[0]][:3]
            # Apply color enhancement for realistic ham appearance
            enhanced_colors = self.apply_ham_color_enhancement(mesh_colors)
            # Assign enhanced colors to mesh
            mesh.vertex_colors = o3d.utility.Vector3dVector(enhanced_colors)
            print(f"Transferred and enhanced colors from {len(pcd_points)} points to {len(mesh_vertices)} mesh vertices")
            return mesh
        except Exception as e:
            print(f"Color transfer to surface mesh failed: {e}")
            return mesh

    def optimize_mesh_topology(self, mesh):
        """Optimize mesh topology for better surface quality and ham-like appearance"""
        try:
            print("Applying mesh topology optimization...")
            # If an Open3D mesh was passed in, convert to trimesh first
            if isinstance(mesh, o3d.geometry.TriangleMesh):
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                # convert to trimesh for topology ops
                tri = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
            else:
                tri = mesh  # assume trimesh.Trimesh
            # Use safe attribute checks because implementations vary
            if hasattr(tri, "remove_duplicated_vertices"):
                try:
                    tri.remove_duplicated_vertices()
                except Exception:
                    pass
            if hasattr(tri, "remove_duplicated_triangles"):
                try:
                    tri.remove_duplicated_triangles()
                except Exception:
                    pass
            if hasattr(tri, "remove_degenerate_triangles"):
                try:
                    tri.remove_degenerate_triangles()
                except Exception:
                    pass
            # Avoid calling fill_holes on Open3D objects. For trimesh, try repair but guard missing deps.
            try:
                if hasattr(tri, "repair") and hasattr(tri.repair, "fill_holes"):
                    tri.repair.fill_holes()
            except Exception:
                # repair.fill_holes may require networkx; skip if unavailable
                pass
            # Conservative simplification if extremely dense
            try:
                if hasattr(tri, "triangles") and len(tri.triangles) > 100000:
                    target = max(50000, len(tri.triangles) // 2)
                    tri = tri.simplify_quadratic_decimation(target)
            except Exception:
                pass
            # Ensure normals are valid
            try:
                tri.rezero() if hasattr(tri, "rezero") else None
                tri.compute_vertex_normals()
            except Exception:
                pass
            print(f"Mesh optimization completed: {len(tri.vertices)} vertices, {len(tri.faces) if hasattr(tri, 'faces') else len(tri.triangles)} faces")
            return tri
        except Exception as e:
            print(f"Mesh topology optimization failed: {e}")
            return mesh

    def advanced_point_cloud_preprocessing(self, pcd):
        """Advanced preprocessing with intelligent parameter selection"""
        try:
            print("🧠 Analyzing point cloud characteristics...")
            # Analyze point cloud properties
            points = np.asarray(pcd.points)
            bbox = pcd.get_axis_aligned_bounding_box()
            volume = bbox.volume()
            density = len(points) / (volume + 1e-6)
            extent = bbox.extent()
            print(f"📊 Point cloud analysis:")
            print(f"   - Density: {density:.2f} points/unit³")
            print(f"   - Extent: {extent}")
            print(f"   - Volume: {volume:.6f}")
            # Intelligent normal estimation based on analysis
            if not pcd.has_normals():
                print("🧭 Estimating normals with AI-powered parameters...")
                # Adaptive search parameters based on density and extent
                if density > 1000:  # High density
                    radius = min(0.02, extent.min() * 0.01)
                    max_nn = min(100, int(density * 0.1))
                elif density > 100:  # Medium density
                    radius = min(0.05, extent.min() * 0.02)
                    max_nn = min(80, int(density * 0.2))
                else:  # Low density
                    radius = min(0.1, extent.min() * 0.05)
                    max_nn = min(50, int(density * 0.5))
                print(f"   - Adaptive radius: {radius:.4f}")
                print(f"   - Adaptive max_nn: {max_nn}")
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius, max_nn=max_nn))
                # Intelligent normal orientation
                k_orientation = min(200, max(50, int(len(points) * 0.001)))
                pcd.orient_normals_consistent_tangent_plane(k=k_orientation)
                print(f"   - Normal orientation k: {k_orientation}")
            # Advanced outlier removal
            print("🧹 Advanced outlier removal...")
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            # Density-based filtering for very dense point clouds
            if density > 5000:
                print("📉 Applying density-based filtering...")
                voxel_size = min(0.001, extent.min() * 0.001)
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                print(f"   - Voxel size: {voxel_size:.6f}")
                print(f"   - Filtered to: {len(pcd.points)} points")
            return pcd
        except Exception as e:
            print(f"⚠️ Advanced preprocessing failed: {e}")
            return pcd

    def apply_ham_color_enhancement(self, colors):
        """Preserve original colors without ham-specific enhancement"""
        try:
            # Simply return the original colors without any modification
            print("Preserving original colors without enhancement")
            return colors
        except Exception as e:
            print(f"Color preservation failed: {e}")
            return colors

    def adjust_scales_for_ham(self, scales):
        """Preserve original scales without ham-specific adjustments"""
        try:
            # Simply return the original scales without any modification
            print("Preserving original scales without adjustment")
            return scales
        except Exception as e:
            print(f"Scale preservation failed: {e}")
            return scales

    def _remove_invalid_points_and_colors(self, pcd):
        """Return a cleaned point cloud with non-finite points removed (keeps colors aligned)."""
        pts = np.asarray(pcd.points)
        mask = np.isfinite(pts).all(axis=1)
        if mask.all():
            return pcd
        pts_clean = pts[mask]
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(pts_clean)
        if pcd.has_colors():
            cols = np.asarray(pcd.colors)[mask]
            new_pcd.colors = o3d.utility.Vector3dVector(cols)
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)[mask]
            new_pcd.normals = o3d.utility.Vector3dVector(normals)
        return new_pcd

    def _keep_largest_cluster(self, pcd, eps=0.02, min_points=10):
        """Keep only the largest DBSCAN cluster in the point cloud."""
        try:
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
            if labels.size == 0:
                return pcd
            unique, counts = np.unique(labels[labels >= 0], return_counts=True)
            if unique.size == 0:
                return pcd
            largest_label = unique[np.argmax(counts)]
            indices = np.where(labels == largest_label)[0].tolist()
            return pcd.select_by_index(indices)
        except Exception as e:
            print(f"_keep_largest_cluster skipped: {e}")
            return pcd

    def _remove_planes(self, pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000, max_planes=3, min_ratio=0.01):
        """Iteratively remove large planar components from the point cloud."""
        try:
            remaining = pcd
            for i in range(max_planes):
                if len(remaining.points) < 50:
                    break
                plane_model, inliers = remaining.segment_plane(distance_threshold=distance_threshold,
                                                               ransac_n=ransac_n,
                                                               num_iterations=num_iterations)
                if len(inliers) == 0:
                    break
                frac = len(inliers) / max(1, len(remaining.points))
                if frac < min_ratio:
                    break
                print(f"Removing plane #{i+1}: {len(inliers)} points ({frac:.2%})")
                remaining = remaining.select_by_index(inliers, invert=True)
            return remaining
        except Exception as e:
            print(f"_remove_planes skipped: {e}")
            return pcd

    def _prune_mesh_by_point_distance(self, tri_mesh, pcd,
                                      distance_factor=2.8,
                                      absolute_max=0.05,
                                      keep_ratio=0.90):
        """
        Remove vertices (and attached faces) whose nearest distance to original
        points is too large (likely Poisson sheet / plate).
        distance_factor * median_distance sets threshold (clamped by absolute_max if given).
        keep_ratio: ensure we don't delete too much; abort if deletions would exceed (1-keep_ratio).
        """
        try:
            pts = np.asarray(pcd.points)
            if len(pts) == 0 or len(tri_mesh.vertices) == 0:
                return tri_mesh
            # KDTree on original points
            import scipy.spatial
            tree = scipy.spatial.cKDTree(pts)
            v = tri_mesh.vertices
            dists, _ = tree.query(v, k=1)
            med = np.median(dists)
            thr = med * distance_factor
            if absolute_max is not None:
                thr = min(thr, absolute_max)
            mask_remove = dists > thr
            remove_count = int(mask_remove.sum())
            frac_remove = remove_count / len(v)
            print(f"Vertex distance pruning: median={med:.6f}  thr={thr:.6f}  "
                  f"remove={remove_count} ({frac_remove:.2%})")
            # Safety: if removal would nuke most of mesh, skip
            if frac_remove > (1 - keep_ratio):
                print("Pruning skipped (too many vertices marked).")
                return tri_mesh
            if remove_count == 0:
                return tri_mesh
            # Remove vertices and faces
            keep_mask = ~mask_remove
            keep_indices = np.where(keep_mask)[0]
            # Create vertex mapping
            vertex_map = np.full(len(tri_mesh.vertices), -1)
            vertex_map[keep_indices] = np.arange(len(keep_indices))
            # Filter vertices
            new_vertices = tri_mesh.vertices[keep_indices]
            # Filter faces - keep only faces where all vertices are kept
            old_faces = tri_mesh.faces
            new_faces = []
            for face in old_faces:
                if all(vertex_map[v] >= 0 for v in face):
                    new_face = [vertex_map[v] for v in face]
                    new_faces.append(new_face)
            if len(new_faces) == 0:
                print("Warning: All faces removed during pruning")
                return tri_mesh
            # Create new mesh
            new_mesh = trimesh.Trimesh(vertices=new_vertices, 
                                       faces=np.array(new_faces),
                                       process=True)
            print(f"Pruned mesh: {len(new_vertices)} verts, {len(new_faces)} faces")
            return new_mesh
        except Exception as e:
            print(f"_prune_mesh_by_point_distance failed: {e}")
            return tri_mesh

    def _remove_flat_mesh_components(self, tri_mesh,
                                     thickness_tol=0.015,        # More aggressive
                                     area_ratio_threshold=0.02): # Lower threshold
        """
        Remove large, nearly planar components (e.g., base plate).
        thickness_tol: max thickness (smallest bbox axis) to treat as flat
        area_ratio_threshold: min area fraction of total to consider for removal
        """
        try:
            # Split into connected components
            components = tri_mesh.split(only_watertight=False)
            if len(components) <= 1:
                return tri_mesh
            total_area = sum(comp.area for comp in components if hasattr(comp, 'area') and comp.area > 0)
            if total_area == 0:
                return tri_mesh
            kept_components = []
            removed_count = 0
            for idx, comp in enumerate(components):
                if not hasattr(comp, 'area') or comp.area <= 0:
                    continue
                # Get bounding box
                bounds = comp.bounds
                if bounds is None or len(bounds) != 2:
                    kept_components.append(comp)
                    continue
                extents = bounds[1] - bounds[0]
                smallest_extent = np.min(extents)
                area_ratio = comp.area / total_area
                print(f"Component {idx}: area={comp.area:.4f} ratio={area_ratio:.3f} "
                      f"extents={extents} thickness={smallest_extent:.5f}")
                # Check if it's a flat sheet
                is_flat = smallest_extent < thickness_tol
                is_large = area_ratio > area_ratio_threshold
                if is_flat and is_large:
                    print(f"Removing flat component {idx} (area_ratio={area_ratio:.3f}, thickness={smallest_extent:.5f})")
                    removed_count += 1
                else:
                    kept_components.append(comp)
            if removed_count == 0:
                print("No flat components found to remove")
                return tri_mesh
            if len(kept_components) == 0:
                print("Warning: All components would be removed, keeping original")
                return tri_mesh
            # Combine remaining components
            try:
                combined = trimesh.util.concatenate(kept_components)
                print(f"Removed {removed_count} flat component(s); new mesh: {len(combined.vertices)} verts, {len(combined.faces)} faces")
                return combined
            except Exception as concat_error:
                print(f"Failed to concatenate components: {concat_error}")
                return tri_mesh
        except Exception as e:
            print(f"_remove_flat_mesh_components failed: {e}")
            return tri_mesh

    def _prepare_point_cloud_for_reconstruction(self, pcd,
                                                cluster_eps=0.02,
                                                cluster_min_points=8,
                                                plane_dist=0.01,
                                                max_planes=4,
                                                plane_min_ratio=0.01,
                                                z_trim_quantile=0.02):
        """Petal‑safe preprocessing: honors keep_all_clusters / disable_plane_removal / disable_z_trim."""
        try:
            original_count = len(pcd.points)
            pcd = self._remove_invalid_points_and_colors(pcd)
            print(f"[Pre] start points: {original_count}")
            if not self.keep_all_clusters:
                before = len(pcd.points)
                pcd = self._keep_largest_cluster(pcd, eps=cluster_eps, min_points=cluster_min_points)
                print(f"[Pre] largest cluster kept: {before} -> {len(pcd.points)}")
            else:
                print("[Pre] skipping cluster removal (keep_all_clusters=True)")
            if not self.disable_plane_removal:
                before = len(pcd.points)
                pcd = self._remove_planes(pcd,
                                          distance_threshold=plane_dist,
                                          max_planes=max_planes,
                                          min_ratio=plane_min_ratio)
                print(f"[Pre] plane removal: {before} -> {len(pcd.points)}")
            else:
                print("[Pre] skipping plane removal (disable_plane_removal=True)")
            if not self.disable_z_trim:
                pts = np.asarray(pcd.points)
                if len(pts) > 30:
                    z = pts[:, 2]
                    z_min = np.quantile(z, z_trim_quantile)
                    keep = z > z_min + (np.ptp(z) * 0.005)
                    if keep.sum() > 30 and keep.sum() < len(pts):
                        removed = len(pts) - keep.sum()
                        pcd = pcd.select_by_index(np.where(keep)[0])
                        print(f"[Pre] z-trim removed {removed}")
            else:
                print("[Pre] skipping z-trim (disable_z_trim=True)")
            # Adaptive normals for small thin petals
            if not pcd.has_normals():
                pts = np.asarray(pcd.points)
                extent = np.ptp(pts, axis=0)
                min_axis = max(1e-6, extent.min())
                radius = min_axis * 0.12  # smaller neighborhood
                radius = max(radius, min_axis * 0.05)
                max_nn = 60
                print(f"[Pre] estimating normals radius={radius:.5f} max_nn={max_nn}")
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius, max_nn=max_nn))
                pcd.orient_normals_consistent_tangent_plane(k=min(120, max(30, len(pcd.points)//200)))
            print(f"[Pre] final points: {len(pcd.points)} (lost {original_count - len(pcd.points)})")
            return pcd
        except Exception as e:
            print(f"_prepare_point_cloud_for_reconstruction failed: {e}")
            return pcd

    def _normalize_point_cloud_for_reconstruction(self, pcd):
        """
        Return (normalized_pcd, center, scale).
        Scales so max axis extent becomes 1.0 (isotropic), centers at origin.
        """
        try:
            pts = np.asarray(pcd.points)
            center = pts.mean(axis=0)
            shifted = pts - center
            extents = np.ptp(shifted, axis=0)   # NumPy 2.0 compatible
            scale = extents.max() if extents.max() > 0 else 1.0
            norm_pts = shifted / scale
            np_pcd = o3d.geometry.PointCloud()
            np_pcd.points = o3d.utility.Vector3dVector(norm_pts)
            if pcd.has_colors():
                np_pcd.colors = pcd.colors
            if pcd.has_normals():
                np_pcd.normals = pcd.normals
            return np_pcd, center, scale
        except Exception as e:
            print(f"_normalize_point_cloud_for_reconstruction failed: {e}")
            return pcd, np.zeros(3), 1.0

    def _load_original_mesh_if_present(self):
        """Check if the input PLY contains mesh data (faces) and return it if so."""
        try:
             # Attempt to read the PLY file as a mesh
            mesh = o3d.io.read_triangle_mesh(self.input_path)
            if len(mesh.triangles) > 0:
                print("Original mesh (faces) detected in PLY file.")
                return mesh
            else:
                print("PLY file contains only points, no faces found.")
                return None
        except Exception as e:
            print(f"Could not load original mesh from PLY: {e}")
            return None


# --- Main GUI Application Class ---
class PLYConverterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_path = ""
        self.output_dir = ""
        self.worker_thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("PLY Converter")
        self.setGeometry(100, 100, 800, 600)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # --- Input Section ---
        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.StyledPanel)
        input_layout = QVBoxLayout()
        input_frame.setLayout(input_layout)

        input_label = QLabel("Input PLY File:")
        input_label.setFont(QFont("Arial", 10, QFont.Bold))
        input_layout.addWidget(input_label)

        input_hbox = QHBoxLayout()
        self.input_line_edit = QLabel("No file selected")
        self.input_line_edit.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.input_line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        input_hbox.addWidget(self.input_line_edit)

        self.browse_input_btn = QPushButton("Browse...")
        self.browse_input_btn.clicked.connect(self.browse_input)
        input_hbox.addWidget(self.browse_input_btn)
        input_layout.addLayout(input_hbox)

        # --- Output Section ---
        output_frame = QFrame()
        output_frame.setFrameShape(QFrame.StyledPanel)
        output_layout = QVBoxLayout()
        output_frame.setLayout(output_layout)

        output_label = QLabel("Output Directory:")
        output_label.setFont(QFont("Arial", 10, QFont.Bold))
        output_layout.addWidget(output_label)

        output_hbox = QHBoxLayout()
        self.output_line_edit = QLabel("No directory selected")
        self.output_line_edit.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.output_line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        output_hbox.addWidget(self.output_line_edit)

        self.browse_output_btn = QPushButton("Browse...")
        self.browse_output_btn.clicked.connect(self.browse_output)
        output_hbox.addWidget(self.browse_output_btn)
        output_layout.addLayout(output_hbox)

        # --- Progress Section ---
        progress_frame = QFrame()
        progress_frame.setFrameShape(QFrame.StyledPanel)
        progress_layout = QVBoxLayout()
        progress_frame.setLayout(progress_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setMaximumHeight(150)
        progress_layout.addWidget(self.log_text_edit)

        # --- Control Section ---
        control_layout = QHBoxLayout()
        
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.clicked.connect(self.start_conversion)
        self.convert_btn.setEnabled(False) # Disabled until both paths are selected
        control_layout.addWidget(self.convert_btn)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)
        control_layout.addWidget(self.exit_btn)

        # --- Add sections to main layout ---
        main_layout.addWidget(input_frame)
        main_layout.addWidget(output_frame)
        main_layout.addWidget(progress_frame)
        main_layout.addLayout(control_layout)

        # Status Bar
        self.statusBar().showMessage("Ready")

    def browse_input(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input PLY File", "", "PLY Files (*.ply)")
        if file_path:
            self.input_path = file_path
            self.input_line_edit.setText(file_path)
            self.check_ready_to_convert()

    def browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir = dir_path
            self.output_line_edit.setText(dir_path)
            self.check_ready_to_convert()

    def check_ready_to_convert(self):
        """Enable the Convert button only if both input and output are selected."""
        self.convert_btn.setEnabled(bool(self.input_path) and bool(self.output_dir))

    def start_conversion(self):
        if not self.input_path or not self.output_dir:
            QMessageBox.warning(self, "Input Error", "Please select both input file and output directory.")
            return

        # Disable UI elements during conversion
        self.browse_input_btn.setEnabled(False)
        self.browse_output_btn.setEnabled(False)
        self.convert_btn.setEnabled(False)
        self.exit_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text_edit.clear()

        # Create and start worker thread
        # Using default parameters as in the original worker init
        self.worker_thread = ConversionWorker(
            self.input_path,
            self.output_dir,
            keep_all_clusters=True,
            disable_plane_removal=True,
            disable_z_trim=True,
            high_detail=True
        )
        self.worker_thread.progress.connect(self.update_log)
        self.worker_thread.finished.connect(self.on_conversion_finished)
        self.worker_thread.start()

    def update_log(self, message):
        self.log_text_edit.append(message)
        self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())

    def on_conversion_finished(self, success, message):
        # Re-enable UI elements
        self.browse_input_btn.setEnabled(True)
        self.browse_output_btn.setEnabled(True)
        self.convert_btn.setEnabled(True)
        self.exit_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if success:
            self.statusBar().showMessage("Conversion completed successfully!")
            QMessageBox.information(
                self, 
                "Success", 
                f"Conversion completed successfully!\n"
                f"Files saved to:\n{self.output_dir}\n\n"
                f"Generated formats:\n"
                f"• conversion_output.stl (STL - Gaussian Splats)\n"
                f"• conversion_output.glb (GLB - 3D Gaussian Splats)\n"
                f"• conversion_output.3mf (3MF - Gaussian Splats)\n"
                f"• conversion_output.dxf (DXF - Gaussian Splats)"
            )
        else:
            self.statusBar().showMessage("Conversion failed")
            QMessageBox.critical(self, "Error", f"Conversion failed:\n{message}")

# --- Main Execution ---
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    # Set application properties
    app.setApplicationName("PLY Converter")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("PLY Converter")
    window = PLYConverterGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()