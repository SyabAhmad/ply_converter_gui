import os, numpy as np, open3d as o3d, trimesh
from PyQt5.QtCore import QThread, pyqtSignal
from pygltflib import GLTF2, Scene, Node, Mesh, Buffer, BufferView, Accessor, Asset, Primitive
from utils.color_utils import (
    enhance_colors_vectorized,
    apply_ham_color_enhancement,
    adjust_scales_for_ham,
    transfer_colors_to_mesh,
    transfer_colors_to_mesh_enhanced,
    transfer_colors_to_surface_mesh
)
from utils.mesh_utils import (
    create_ellipsoid_mesh,
    transform_ellipsoid,
    smooth_mesh_for_ham,
    smooth_mesh_for_ham_o3d,
    optimize_mesh_topology,
    prune_mesh_by_point_distance,
    remove_flat_mesh_components
)
from utils.pointcloud_utils import (
    remove_invalid_points_and_colors,
    keep_largest_cluster,
    remove_planes,
    prepare_point_cloud_for_reconstruction,
    normalize_point_cloud_for_reconstruction,
    advanced_point_cloud_preprocessing
)

class ConversionWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool,str)

    def __init__(self, input_path, output_dir,
                 keep_all_clusters=True,
                 disable_plane_removal=True,
                 disable_z_trim=True,
                 high_detail=True,
                 preserve_thin_features=True):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.keep_all_clusters = keep_all_clusters
        self.disable_plane_removal = disable_plane_removal
        self.disable_z_trim = disable_z_trim
        self.high_detail = high_detail
        self.preserve_thin_features = preserve_thin_features

    # Wrapper methods (keep old names if needed elsewhere)
    def enhance_colors_vectorized(self, c): return enhance_colors_vectorized(c)
    def apply_ham_color_enhancement(self,c): return apply_ham_color_enhancement(c)
    def adjust_scales_for_ham(self,s): return adjust_scales_for_ham(s)
    def _remove_invalid_points_and_colors(self,p): return remove_invalid_points_and_colors(p)
    def _keep_largest_cluster(self,p,**k): return keep_largest_cluster(p,**k)
    def _remove_planes(self,p,**k): return remove_planes(p,**k)
    def _prepare_point_cloud_for_reconstruction(self,p,**k): return prepare_point_cloud_for_reconstruction(p,**k)
    def _normalize_point_cloud_for_reconstruction(self,p): return normalize_point_cloud_for_reconstruction(p)
    def transfer_colors_to_surface_mesh(self,pcd,mesh,colors): return transfer_colors_to_surface_mesh(pcd,mesh,colors)
    def optimize_mesh_topology(self,m): return optimize_mesh_topology(m)
    def smooth_mesh_for_ham_o3d(self,m): return smooth_mesh_for_ham_o3d(m)

    def _duplicate_for_thin_features(self, pcd, offset=0.0008, repeats=1):
        """
        Duplicate points along +/- normal to give Poisson some thickness.
        """
        if (not pcd.has_normals()) or len(pcd.points) == 0:
            return pcd
        pts = np.asarray(pcd.points)
        nrm = np.asarray(pcd.normals)
        cols = np.asarray(pcd.colors) if pcd.has_colors() else None
        dup_pts = [pts]
        dup_cols = [cols] if cols is not None else None
        for i in range(1, repeats+1):
            d = offset * i
            dup_pts.append(pts + nrm * d)
            dup_pts.append(pts - nrm * d)
            if cols is not None:
                dup_cols.append(cols)
                dup_cols.append(cols)
        all_pts = np.vstack(dup_pts)
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(all_pts)
        if cols is not None:
            all_cols = np.vstack(dup_cols)
            new_pcd.colors = o3d.utility.Vector3dVector(all_cols)
        return new_pcd

    def _alpha_shape_reconstruct(self, pcd, alpha=None):
        """
        Alpha shape surface (captures thin shells better).
        """
        try:
            if alpha is None:
                # heuristic: average nearest neighbor distance * factor
                pts = np.asarray(pcd.points)
                if len(pts) < 20:
                    return None
                pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                nn = []
                for i, p in enumerate(pts[:min(500, len(pts))]):
                    _, idx, dist = pcd_tree.search_knn_vector_3d(p, 2)
                    if idx and len(dist) == 2:
                        nn.append(np.sqrt(dist[1]))
                mean_nn = np.median(nn) if nn else 0.001
                alpha = mean_nn * 1.8
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()
            return mesh
        except Exception as e:
            print("Alpha shape failed:", e)
            return None

    def run(self):
        try:
            self.progress.emit("Loading 3D Gaussian Splatting PLY...")
            pcd = o3d.io.read_point_cloud(self.input_path)
            self.progress.emit(f"Loaded {len(pcd.points)} raw points")

            # Basic validity
            pcd = self._remove_invalid_points_and_colors(pcd)

            # Adaptive voxel (gentle)
            if len(pcd.points) > 400_000:
                # compute bbox size
                pts = np.asarray(pcd.points)
                extent = np.max(np.ptp(pts, axis=0))
                voxel = extent * 0.002  # finer than before
                pcd = pcd.voxel_down_sample(voxel_size=voxel)
                self.progress.emit(f"Downsampled adaptively (voxel {voxel:.5f}) -> {len(pcd.points)} points")

            # Normals for later (small radius to preserve petals)
            self.progress.emit("Estimating normals...")
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.02 if self.high_detail else 0.04,
                max_nn=90))
            try:
                pcd.orient_normals_consistent_tangent_plane(80)
            except:
                pcd.normalize_normals()

            if pcd.has_colors():
                self.progress.emit("Enhancing colors...")
                cols = np.asarray(pcd.colors).astype(np.float32)
                pcd.colors = o3d.utility.Vector3dVector(self.enhance_colors_vectorized(cols))

            # Reconstruction (STL surface)
            self.progress.emit("Preparing for reconstruction...")
            prep_pcd = self._prepare_point_cloud_for_reconstruction(
                pcd,
                keep_all_clusters=self.keep_all_clusters,
                plane_removal=not self.disable_plane_removal,
                cluster_eps=0.015,
                cluster_min_points=6
            )
            self.progress.emit(f"Prepared point cloud: {len(prep_pcd.points)} points")

            # Optional thickness
            if self.preserve_thin_features:
                self.progress.emit("Duplicating points for thin feature support...")
                thick_pcd = self._duplicate_for_thin_features(prep_pcd,
                                                              offset=0.0006,
                                                              repeats=1 if len(prep_pcd.points) < 200000 else 0)
            else:
                thick_pcd = prep_pcd

            # Normalize
            norm_pcd, center, scale = self._normalize_point_cloud_for_reconstruction(thick_pcd)

            self.progress.emit("Poisson reconstruction (high depth)...")
            depth = 11 if self.high_detail else 9
            try:
                mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    norm_pcd,
                    depth=depth,
                    scale=1.02,
                    linear_fit=True
                )
            except Exception as e:
                mesh = None
                print("Poisson failed:", e)

            valid_mesh = False
            if mesh and len(mesh.vertices):
                # Reverse normalization
                verts = np.asarray(mesh.vertices) * scale + center
                mesh.vertices = o3d.utility.Vector3dVector(verts)
                # Optional density pruning (skip in high detail to avoid losing petals)
                if (not self.high_detail) and dens is not None:
                    try:
                        d = np.asarray(dens)
                        cutoff = np.quantile(d, 0.02)
                        mask = d < cutoff
                        mesh.remove_vertices_by_mask(mask)
                    except Exception as e:
                        print("Density prune skipped:", e)
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_unreferenced_vertices()
                mesh.compute_vertex_normals()
                valid_mesh = len(mesh.triangles) > 0

            if not valid_mesh or len(mesh.vertices) < 0.25 * len(prep_pcd.points):
                self.progress.emit("Poisson insufficient; trying ball pivot...")
                try:
                    prep_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=60))
                    radii = o3d.utility.DoubleVector([0.0008, 0.0015, 0.0025, 0.004, 0.006, 0.01])
                    bp_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(prep_pcd, radii)
                    if len(bp_mesh.triangles):
                        mesh = bp_mesh
                        mesh.compute_vertex_normals()
                        valid_mesh = True
                except Exception as e:
                    print("Ball pivot failed:", e)

            if (not valid_mesh) or len(mesh.vertices) < 0.25 * len(prep_pcd.points):
                self.progress.emit("Trying alpha shape fallback...")
                a_mesh = self._alpha_shape_reconstruct(prep_pcd)
                if a_mesh and len(a_mesh.triangles):
                    mesh = a_mesh
                    valid_mesh = True

            if valid_mesh:
                self.progress.emit("Topology optimize & color transfer...")
                mesh = self.optimize_mesh_topology(mesh)
                if pcd.has_colors():
                    # Nearest color transfer
                    self.transfer_colors_to_surface_mesh(prep_pcd, mesh, np.hstack((
                        np.asarray(prep_pcd.colors),
                        np.ones((len(prep_pcd.points),1))
                    )))
                # Save STL improved (overwrite later exporter)
                stl_path = os.path.join(self.output_dir, "conversion_output_surface.stl")
                trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                faces=np.asarray(mesh.triangles),
                                process=True).export(stl_path, 'stl')
                self.progress.emit("Surface reconstruction saved (surface STL).")

            # Continue with legacy exporters for compatibility
            self.progress.emit("Exporting legacy STL...")
            self.export_point_cloud_stl(pcd, os.path.join(self.output_dir,"conversion_output.stl"))
            self.progress.emit("Exporting GLB (points)...")
            self.export_point_cloud_glb(pcd, os.path.join(self.output_dir,"conversion_output.glb"))
            self.progress.emit("Exporting 3MF...")
            self.export_point_cloud_3mf(pcd, os.path.join(self.output_dir,"conversion_output.3mf"))
            self.progress.emit("Exporting DXF...")
            self.export_point_cloud_dxf(pcd, os.path.join(self.output_dir,"conversion_output.dxf"))
            self.progress.emit("Conversion complete")
            self.finished.emit(True,"OK")
        except Exception as e:
            self.finished.emit(False,str(e))

    # -------- (Exporter methods copied from original file; keep logic unchanged) --------
    # Keep only representative ones here; replicate others similarly as needed.

    def _load_original_mesh_if_present(self):
        try:
            mesh = o3d.io.read_triangle_mesh(self.input_path)
            if mesh and len(mesh.vertices) and len(mesh.triangles):
                if not mesh.has_vertex_normals():
                    mesh.compute_vertex_normals()
                return mesh
        except:
            pass
        return None

    def export_point_cloud_stl(self, pcd, path):
        try:
            orig = self._load_original_mesh_if_present()
            if orig:
                tri = trimesh.Trimesh(vertices=np.asarray(orig.vertices),
                                      faces=np.asarray(orig.triangles),
                                      process=True)
                tri.export(path,'stl')
                return
            pcd = self._prepare_point_cloud_for_reconstruction(pcd)
            norm_pcd, center, scale = self._normalize_point_cloud_for_reconstruction(pcd)
            mesh,_ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(norm_pcd, depth=8, scale=1.1)
            verts = np.asarray(mesh.vertices)*scale + center
            tri = trimesh.Trimesh(verts, np.asarray(mesh.triangles), process=True)
            tri.export(path,'stl')
        except Exception as e:
            print("STL export failed:",e)

    def export_point_cloud_glb(self, pcd, path):
        try:
            orig = self._load_original_mesh_if_present()
            if orig:
                tri = trimesh.Trimesh(vertices=np.asarray(orig.vertices),
                                      faces=np.asarray(orig.triangles),
                                      process=True)
                tri.export(path,'glb')
                return
            # Simplified fallback: export points
            pts = np.asarray(pcd.points).astype(np.float32)
            if pcd.has_colors():
                colors = np.asarray(pcd.colors).astype(np.float32)
            else:
                colors = np.ones((len(pts),3),dtype=np.float32)*0.8
            colors_rgba = np.column_stack((colors, np.ones(len(colors),dtype=np.float32)))
            blob = pts.tobytes() + colors_rgba.tobytes()
            buffer = Buffer(byteLength=len(blob))
            bv_pos = BufferView(buffer=0, byteOffset=0, byteLength=pts.nbytes, target=34962)
            bv_col = BufferView(buffer=0, byteOffset=pts.nbytes, byteLength=colors_rgba.nbytes, target=34962)
            acc_pos = Accessor(bufferView=0, byteOffset=0, componentType=5126, count=len(pts), type="VEC3")
            acc_col = Accessor(bufferView=1, byteOffset=0, componentType=5126, count=len(colors_rgba), type="VEC4")
            prim = Primitive(attributes={"POSITION":0,"COLOR_0":1}, mode=0)
            mesh = Mesh(primitives=[prim])
            node = Node(mesh=0)
            scene = Scene(nodes=[0])
            gltf = GLTF2(asset=Asset(version="2.0"),
                         scenes=[scene], nodes=[node], meshes=[mesh],
                         buffers=[buffer], bufferViews=[bv_pos,bv_col],
                         accessors=[acc_pos,acc_col])
            gltf.set_binary_blob(blob)
            gltf.save_binary(path)
        except Exception as e:
            print("GLB export failed:",e)

    def export_point_cloud_3mf(self, pcd, path):
        try:
            orig = self._load_original_mesh_if_present()
            if orig:
                tri = trimesh.Trimesh(vertices=np.asarray(orig.vertices),
                                      faces=np.asarray(orig.triangles),
                                      process=True)
                tri.export(path,'3mf')
                return
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
            mesh,_ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, linear_fit=True)
            tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                  faces=np.asarray(mesh.triangles),
                                  process=True)
            tri.export(path,'3mf')
        except Exception as e:
            print("3MF export failed:",e)

    def export_point_cloud_dxf(self, pcd, path):
        import ezdxf
        try:
            orig = self._load_original_mesh_if_present()
            if orig:
                verts = np.asarray(orig.vertices)
                faces = np.asarray(orig.triangles)
            else:
                pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
                mesh,_ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, linear_fit=True)
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
            doc = ezdxf.new('R2010'); msp = doc.modelspace()
            for f in faces:
                v1,v2,v3 = verts[f[0]],verts[f[1]],verts[f[2]]
                msp.add_3dface([v1,v2,v3,v1])
            doc.saveas(path)
        except Exception as e:
            print("DXF export failed:",e)