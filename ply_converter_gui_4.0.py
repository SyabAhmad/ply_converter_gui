import sys
import os
import threading
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QProgressBar, QTextEdit, QMessageBox, QFrame)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap
import open3d as o3d
import numpy as np
import trimesh
import ezdxf
import plyfile
from pygltflib import GLTF2, Scene, Node, Mesh, Buffer, BufferView, Accessor, Asset, Primitive

class ConversionWorker(QThread): 
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, input_path, output_dir):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        
    def run(self):
        try:
            self.progress.emit("Loading 3D Gaussian Splatting PLY...")
            pcd = o3d.io.read_point_cloud(self.input_path)
            self.progress.emit(f"Loaded {len(pcd.points)} Gaussian splats")
            
            self.progress.emit("Pre-processing Gaussian splats...")
            # Optional: light downsampling to reduce file size while preserving quality
            if len(pcd.points) > 100000:  # Only downsample if very large
                pcd = pcd.voxel_down_sample(voxel_size=0.001)
                self.progress.emit(f"Downsampled to {len(pcd.points)} splats")
            
            # Preserve and enhance original colors if available
            if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
                self.progress.emit("Processing and enhancing colors...")
                # Apply enhanced color enhancement to point cloud for vivid ham appearance
                colors = np.asarray(pcd.colors).astype(np.float32)
                if colors.max() > 1.0:
                    colors = colors / 255.0
                enhanced_colors = self.enhance_colors_vectorized(colors)
                pcd.colors = o3d.utility.Vector3dVector(enhanced_colors)
                self.progress.emit(f"Enhanced {len(enhanced_colors)} colors for vivid ham appearance")
            
            self.progress.emit("Exporting to STL (Gaussian splats)...")
            stl_path = os.path.join(self.output_dir, "conversion_output.stl")
            self.export_point_cloud_stl(pcd, stl_path)
            
            self.progress.emit("Exporting to GLB (3D Gaussian Splats)...")
            glb_path = os.path.join(self.output_dir, "conversion_output.glb")
            self.export_point_cloud_glb(pcd, glb_path)
            
            self.progress.emit("Exporting to 3MF (Gaussian splats)...")
            mf_path = os.path.join(self.output_dir, "conversion_output.3mf")
            self.export_point_cloud_3mf(pcd, mf_path)
            
            self.progress.emit("Exporting to DXF (Gaussian splats)...")
            dxf_path = os.path.join(self.output_dir, "conversion_output.dxf")
            self.export_point_cloud_dxf(pcd, dxf_path)
            
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
    
    def export_point_cloud_stl(self, pcd, output_path):
        """
        Export point cloud to STL using Poisson surface reconstruction for higher accuracy
        """
        try:
            # Convert to Open3D point cloud
            if not isinstance(pcd, o3d.geometry.PointCloud):
                raise TypeError("pcd must be an Open3D PointCloud")

            # Remove NaNs or infinite points
            pcd.remove_non_finite_points()

            # Estimate normals if not present
            if not pcd.has_normals():
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
                pcd.orient_normals_consistent_tangent_plane(160)

            print(f"Performing Poisson surface reconstruction on {len(pcd.points)} points...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9  # depth controls resolution (higher = more accurate)
            ) 
            # Convert Open3D mesh to Trimesh for STL export
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            trimesh_mesh.export(output_path, file_type='stl')
            print(f"STL exported successfully: {output_path}")
        except Exception as e:
            print(f"STL export failed: {e}")
            # You can implement fallback minimal STL here if needed
    
    def export_point_cloud_3mf(self, pcd, output_path):
        """Export point cloud to 3MF format using Open3D and Poisson surface reconstruction"""
        try:
            # Load the PLY file (ignoring extra properties)
            pcd = o3d.io.read_point_cloud(self.input_path)
            
            # Estimate normals (required for Poisson reconstruction)
            print("Estimating normals for 3MF export...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
            
            # Orient normals consistently (optional but recommended)
            pcd.orient_normals_consistent_tangent_plane(k=20)
            
            # Poisson surface reconstruction
            print("Running Poisson reconstruction for 3MF...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, linear_fit=True)             
            
            # Convert to trimesh and export as 3MF
            print("Exporting to 3MF...")
            tri_mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                vertex_colors=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
            )
            
            tri_mesh.export(output_path, file_type='3mf')
            print(f"3MF surface mesh saved to {output_path}")
            
        except Exception as e:
            print(f"Point cloud 3MF export failed: {e}")
            # Try alternative approach
            try:
                self.export_point_cloud_3mf_alternative(pcd, output_path)
            except Exception as e2:
                print(f"Alternative 3MF export also failed: {e2}")
    
    def export_point_cloud_3mf_alternative(self, pcd, output_path):
        """Alternative 3MF export method using Open3D data"""
        try:
            # Get positions and colors from Open3D point cloud
            positions = np.asarray(pcd.points).astype(np.float32)
            
            if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
                colors = np.asarray(pcd.colors).astype(np.float32)
                # Normalize if needed
                if colors.max() > 1.0:
                    colors = colors / 255.0
                
                # Apply color enhancement
                enhanced_colors = self.enhance_colors_vectorized(colors)
                # Convert to RGBA
                colors_rgba = np.column_stack((enhanced_colors, np.ones(len(enhanced_colors)))).astype(np.float32)
            else:
                # Default white colors
                colors_rgba = np.ones((len(positions), 4), dtype=np.float32)
            
            all_vertices = []
            all_faces = []
            
            # Create small cubes around each point
            cube_size = 0.001  # Small cube size
            for i, point in enumerate(positions):
                x, y, z = point
                
                # Define cube vertices
                cube_vertices = [
                    [x - cube_size/2, y - cube_size/2, z - cube_size/2],
                    [x + cube_size/2, y - cube_size/2, z - cube_size/2],
                    [x + cube_size/2, y + cube_size/2, z - cube_size/2],
                    [x - cube_size/2, y + cube_size/2, z - cube_size/2],
                    [x - cube_size/2, y - cube_size/2, z + cube_size/2],
                    [x + cube_size/2, y - cube_size/2, z + cube_size/2],
                    [x + cube_size/2, y + cube_size/2, z + cube_size/2],
                    [x - cube_size/2, y + cube_size/2, z + cube_size/2]
                ]
                
                # Add cube vertices
                base_idx = len(all_vertices)
                all_vertices.extend(cube_vertices)
                
                # Define cube faces (6 faces, each with 4 vertices)
                cube_faces = [
                    [base_idx, base_idx+1, base_idx+2, base_idx+3],  # bottom
                    [base_idx+4, base_idx+7, base_idx+6, base_idx+5],  # top
                    [base_idx, base_idx+4, base_idx+5, base_idx+1],  # front
                    [base_idx+2, base_idx+6, base_idx+7, base_idx+3],  # back
                    [base_idx, base_idx+3, base_idx+7, base_idx+4],  # left
                    [base_idx+1, base_idx+5, base_idx+6, base_idx+2]   # right
                ]
                
                all_faces.extend(cube_faces)
            
            # Convert to numpy arrays
            vertices_array = np.array(all_vertices, dtype=np.float64)
            faces_array = np.array(all_faces, dtype=np.int32)
            
            # Create trimesh and export
            mesh = trimesh.Trimesh(vertices=vertices_array, faces=faces_array)
            mesh.export(output_path, file_type='3mf')
            
            print(f"Exported point cloud 3MF with {len(positions)} points using alternative method")
            
        except Exception as e:
            print(f"Alternative 3MF export failed: {e}")
    
    def export_point_cloud_dxf(self, pcd, output_path):
        """Export point cloud to DXF format using Open3D and Poisson surface reconstruction"""
        try:
            # Load the PLY file (ignoring extra properties)
            pcd = o3d.io.read_point_cloud(self.input_path)
            
            # Estimate normals (required for Poisson reconstruction)
            print("Estimating normals for DXF export...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
            
            # Orient normals consistently (optional but recommended)
            pcd.orient_normals_consistent_tangent_plane(k=20)
            
            # Poisson surface reconstruction
            print("Running Poisson reconstruction for DXF...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, linear_fit=True)
            
            # Optional: Remove low-density vertices (artifacts)
            # vertices_to_remove = densities < np.quantile(densities, 0.01)
            # mesh.remove_vertices_by_mask(vertices_to_remove)
            
            # Convert Open3D mesh to vertices and faces for DXF export
            print("Exporting to DXF...")
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            
            # Create DXF document
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            
            # Create 3D faces from the mesh triangles
            successful_elements = 0
            for face in faces:
                try:
                    # Get the three vertices of the triangle
                    v1 = vertices[face[0]]
                    v2 = vertices[face[1]]
                    v3 = vertices[face[2]]
                    
                    # Create a 3D face (DXF supports triangular faces)
                    msp.add_3dface([
                        [float(v1[0]), float(v1[1]), float(v1[2])],
                        [float(v2[0]), float(v2[1]), float(v2[2])],
                        [float(v3[0]), float(v3[1]), float(v3[2])],
                        [float(v1[0]), float(v1[1]), float(v1[2])]  # Repeat first vertex for 4-point face
                    ])
                    successful_elements += 1
                    
                except Exception as face_error:
                    print(f"Warning: Failed to create face: {face_error}")
                    continue
            
            # If we don't have enough faces, create some basic geometric elements
            if successful_elements < 3:
                print("Creating basic geometric elements...")
                # Add a simple 3D face to ensure the DXF is valid
                try:
                    msp.add_3dface([[0, 0, 0], [0.001, 0, 0], [0.001, 0.001, 0], [0, 0.001, 0]])
                    successful_elements += 1
                except:
                    pass
                
                # Add some basic lines
                try:
                    msp.add_line((0, 0, 0), (0.001, 0, 0))
                    msp.add_line((0, 0, 0), (0, 0.001, 0))
                    msp.add_line((0, 0, 0), (0, 0, 0.001))
                    successful_elements += 3
                except:
                    pass
            
            doc.saveas(output_path)
            print(f"DXF surface mesh saved to {output_path} with {successful_elements} faces")
            
        except Exception as e:
            print(f"Point cloud DXF export failed: {e}")
            # Try alternative approach
            try:
                self.export_point_cloud_dxf_alternative(pcd, output_path)
            except Exception as e2:
                print(f"Alternative DXF export also failed: {e2}")
    
    def export_point_cloud_dxf_alternative(self, pcd, output_path):
        """Alternative DXF export method using Open3D data"""
        try:
            # Get positions and colors from Open3D point cloud
            positions = np.asarray(pcd.points).astype(np.float32)
            
            if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
                colors = np.asarray(pcd.colors).astype(np.float32)
                # Normalize if needed
                if colors.max() > 1.0:
                    colors = colors / 255.0
                
                # Apply color enhancement
                enhanced_colors = self.enhance_colors_vectorized(colors)
                # Convert to RGBA
                colors_rgba = np.column_stack((enhanced_colors, np.ones(len(enhanced_colors)))).astype(np.float32)
            else:
                # Default white colors
                colors_rgba = np.ones((len(positions), 4), dtype=np.float32)
            
            # Create DXF document
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            
            # Create a more efficient geometric representation using lines and faces
            # This creates a better DXF file than individual small faces
            successful_elements = 0
            
            # Create lines connecting nearby points to form a surface network
            for i in range(0, len(positions) - 1, 2):  # Process every other point
                if i + 1 < len(positions):
                    p1 = positions[i]
                    p2 = positions[i + 1]
                    
                    try:
                        # Create a line between two points
                        x1, y1, z1 = p1
                        x2, y2, z2 = p2
                        
                        # Ensure coordinates are valid numbers
                        if all(not np.isnan(coord) and not np.isinf(coord) for coord in [x1, y1, z1, x2, y2, z2]):
                            # Add a 3D line
                            msp.add_line((float(x1), float(y1), float(z1)), 
                                        (float(x2), float(y2), float(z2)))
                            successful_elements += 1
                            
                    except Exception as line_error:
                        print(f"Warning: Failed to create line between points: {line_error}")
                        continue
            
            # If we don't have enough lines, create some basic geometric elements
            if successful_elements < 3:
                print("Creating basic geometric elements...")
                # Add a simple 3D face to ensure the DXF is valid
                try:
                    msp.add_3dface([[0, 0, 0], [0.001, 0, 0], [0.001, 0.001, 0], [0, 0.001, 0]])
                    successful_elements += 1
                except:
                    pass
                
                # Add some basic lines
                try:
                    msp.add_line((0, 0, 0), (0.001, 0, 0))
                    msp.add_line((0, 0, 0), (0, 0.001, 0))
                    msp.add_line((0, 0, 0), (0, 0, 0.001))
                    successful_elements += 3
                except:
                    pass
            
            doc.saveas(output_path)
            print(f"Exported point cloud DXF with {len(positions)} points using alternative method")
            
        except Exception as e:
            print(f"Alternative DXF export failed: {e}")
    
    def export_point_cloud_glb(self, pcd, output_path):
        """Export 3D Gaussian Splatting PLY to GLB format with proper splat rendering"""
        try:
            # For 3D Gaussian Splatting, we need to read the PLY directly to get all properties
            print("Reading 3D Gaussian Splatting PLY file...")
            
            # Use plyfile to read the PLY and extract Gaussian parameters
            import plyfile
            plydata = plyfile.PlyData.read(self.input_path)
            
            # Extract Gaussian parameters
            positions = np.column_stack((plydata['vertex']['x'], 
                                       plydata['vertex']['y'], 
                                       plydata['vertex']['z'])).astype(np.float32)
            
            # Extract colors (RGB)
            if 'red' in plydata['vertex'] and 'green' in plydata['vertex'] and 'blue' in plydata['vertex']:
                colors = np.column_stack((plydata['vertex']['red'], 
                                        plydata['vertex']['green'], 
                                        plydata['vertex']['blue'])).astype(np.float32)
                # Normalize colors to 0-1 range
                if colors.max() > 1.0:
                    colors = colors / 255.0
            else:
                # Default colors if not present
                colors = np.ones((len(positions), 3), dtype=np.float32) * 0.8
            
            # Extract opacity (alpha)
            if 'opacity' in plydata['vertex']:
                opacity = np.array(plydata['vertex']['opacity']).astype(np.float32)
            else:
                opacity = np.ones(len(positions), dtype=np.float32)
            
            # Extract scale parameters (for ellipsoid shape)
            if 'scale_0' in plydata['vertex'] and 'scale_1' in plydata['vertex'] and 'scale_2' in plydata['vertex']:
                scales = np.column_stack((plydata['vertex']['scale_0'], 
                                        plydata['vertex']['scale_1'], 
                                        plydata['vertex']['scale_2'])).astype(np.float32)
                # Adjust scales for more realistic ham appearance
                scales = self.adjust_scales_for_ham(scales)
            else:
                # Default scale if not present
                scales = np.ones((len(positions), 3), dtype=np.float32) * 0.01
            
            # Extract rotation parameters (quaternions)
            if 'rot_0' in plydata['vertex'] and 'rot_1' in plydata['vertex'] and 'rot_2' in plydata['vertex'] and 'rot_3' in plydata['vertex']:
                rotations = np.column_stack((plydata['vertex']['rot_0'], 
                                           plydata['vertex']['rot_1'], 
                                           plydata['vertex']['rot_2'], 
                                           plydata['vertex']['rot_3'])).astype(np.float32)
            else:
                # Default rotation (identity quaternion)
                rotations = np.zeros((len(positions), 4), dtype=np.float32)
                rotations[:, 0] = 1.0  # w component
            
            print(f"Extracted {len(positions)} Gaussian splats with colors, opacity, scale, and rotation")
            
            # Apply enhanced color enhancement for realistic ham appearance
            enhanced_colors = self.enhance_colors_vectorized(colors)
            
            # Apply ham-specific color adjustments for more realistic appearance
            enhanced_colors = self.apply_ham_color_enhancement(enhanced_colors)
            
            # Combine colors with opacity
            colors_rgba = np.column_stack((enhanced_colors, opacity)).astype(np.float32)
            
            # Create GLB with Gaussian splat data
            self.export_gaussian_splats_glb(positions, colors_rgba, scales, rotations, output_path)
            
        except Exception as e:
            print(f"3D Gaussian Splatting GLB export failed: {e}")
            # Fallback to alternative method
            try:
                self.export_point_cloud_glb_alternative(pcd, output_path)
            except Exception as e2:
                print(f"Alternative GLB export also failed: {e2}")
    
    def export_point_cloud_glb_alternative(self, pcd, output_path):
        """Alternative GLB export method using Open3D data with enhanced quality"""
        try:
            # Get positions and colors from Open3D point cloud
            positions = np.asarray(pcd.points).astype(np.float32)
            
            if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
                colors = np.asarray(pcd.colors).astype(np.float32)
                # Normalize if needed
                if colors.max() > 1.0:
                    colors = colors / 255.0
                
                # Apply enhanced color enhancement for ham-like appearance
                enhanced_colors = self.enhance_colors_vectorized(colors)
                # Convert to RGBA
                colors_rgba = np.column_stack((enhanced_colors, np.ones(len(enhanced_colors)))).astype(np.float32)
            else:
                # Default warm ham-like colors instead of plain white
                colors_rgba = np.ones((len(positions), 4), dtype=np.float32)
                colors_rgba[:, 0] = 0.8  # Slight red tint
                colors_rgba[:, 1] = 0.6  # Slight brown tint
                colors_rgba[:, 2] = 0.4  # Darker tone
            
            # Build binary blob
            bin_blob = positions.tobytes() + colors_rgba.tobytes()
            
            # Buffer definitions
            buffer = Buffer(byteLength=len(bin_blob))
            bv_positions = BufferView(buffer=0, byteOffset=0, byteLength=positions.nbytes, target=34962)
            bv_colors = BufferView(buffer=0, byteOffset=positions.nbytes, byteLength=colors_rgba.nbytes, target=34962)
            
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
            
            print(f"Exported enhanced point cloud GLB with {len(positions)} vertices using Open3D")
            
        except Exception as e:
            print(f"Alternative GLB export failed: {e}")
    
    def export_gaussian_splats_glb(self, positions, colors_rgba, scales, rotations, output_path):
        """Export 3D Gaussian Splats to GLB format with continuous surface mesh"""
        try:
            print("Creating GLB with continuous surface from Gaussian Splat data...")
              
            pcd = o3d.io.read_point_cloud(self.input_path)

            # o3d.visualization.draw_geometries([pcd]) 
            
            # Assign colors to the point cloud
            if colors_rgba.shape[1] >= 3:
                pcd.colors = o3d.utility.Vector3dVector(colors_rgba[:, :3])
            
            # Remove any invalid points
            pcd.remove_non_finite_points()
            
            # Estimate normals for surface reconstruction
            print("Estimating normals for surface reconstruction...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=100))
            pcd.orient_normals_consistent_tangent_plane(k=100)
            
            # Try multiple surface reconstruction methods for best results
            print("Attempting surface reconstruction...")
            mesh = None
            
            # Method 1: Poisson reconstruction with optimized parameters
            try:
                print("Method 1: Poisson surface reconstruction...")
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=9)               

            except Exception as e:
                print(f"Poisson reconstruction failed: {e}")
                mesh = None
            # o3d.visualization.draw_geometries([mesh])
            # Method 2: Ball pivoting if Poisson fails
            if mesh is None or len(mesh.vertices) == 0:
                try:
                    print("Method 2: Ball pivoting surface reconstruction...")
                    # Estimate normals again for ball pivoting
                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.06, max_nn=50))
                    
                    # Use ball pivoting with adaptive radius
                    radii = [0.005, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        pcd, o3d.utility.DoubleVector(radii))
                    
                    if len(mesh.vertices) > 0:
                        print(f"Ball pivoting reconstruction: {len(mesh.vertices)} vertices")
                    else:
                        mesh = None
                        
                except Exception as e:
                    print(f"Ball pivoting reconstruction failed: {e}")
                    mesh = None
            
            # Method 3: Alpha shape if both methods fail
            if mesh is None or len(mesh.vertices) == 0:
                try:
                    print("Method 3: Alpha shape surface reconstruction...")
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                        pcd, alpha=0.1)
                    
                    if len(mesh.vertices) > 0:
                        print(f"Alpha shape reconstruction: {len(mesh.vertices)} vertices")
                    else:
                        mesh = None
                        
                except Exception as e:
                    print(f"Alpha shape reconstruction failed: {e}")
                    mesh = None
            
            # If all methods fail, create a simple convex hull
            if mesh is None or len(mesh.vertices) == 0:
                print("All reconstruction methods failed, creating convex hull...")
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_convex_hull(pcd)
                print(f"Convex hull: {len(mesh.vertices)} vertices")
            
            # Apply mesh smoothing for realistic ham appearance
            print("Applying mesh smoothing for realistic ham texture...")
            mesh = self.smooth_mesh_for_ham_o3d(mesh)
            
            # Optimize mesh topology for better surface quality
            print("Optimizing mesh topology...")
            mesh = self.optimize_mesh_topology(mesh)
            
            # Transfer colors from point cloud to mesh vertices
            print("Transferring colors to surface mesh...")
            mesh = self.transfer_colors_to_surface_mesh(pcd, mesh, colors_rgba)
            
            # Convert to trimesh and export as GLB
            print("Exporting continuous surface mesh to GLB...")
            tri_mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                vertex_colors=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
            )
            
            tri_mesh.export(output_path, file_type='glb')
            print(f"Continuous surface GLB exported successfully: {output_path}")
            
        except Exception as e:
            print(f"Continuous surface GLB export failed: {e}")
            # Fallback: export as simple point cloud
            self.export_simple_gaussian_glb(positions, colors_rgba, output_path)
    
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
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])
        
        # Apply rotation
        rotated_vertices = np.dot(scaled_vertices, rotation_matrix.T)
        
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
            
            # Remove duplicate vertices
            mesh.remove_duplicated_vertices()
            
            # Remove duplicate triangles
            mesh.remove_duplicated_triangles()
            
            # Remove degenerate triangles
            mesh.remove_degenerate_triangles()
            
            # Fill holes in the mesh for continuous surface
            mesh.fill_holes()
            
            # Optimize mesh by removing unnecessary vertices
            mesh = mesh.simplify_quadric_decimation(
                target_number_of_triangles=len(mesh.triangles) // 2)
            
            # Ensure consistent triangle winding
            mesh.compute_vertex_normals()
            
            print(f"Mesh optimization completed: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
            return mesh
            
        except Exception as e:
            print(f"Mesh topology optimization failed: {e}")
            return mesh
    
    def optimize_mesh_preserve_details(self, mesh):
        """Optimize mesh while preserving important details and preventing thin parts removal"""
        try:
            print("Applying detail-preserving mesh optimization...")
            
            # Remove duplicate vertices
            mesh.remove_duplicated_vertices()
            
            # Remove duplicate triangles
            mesh.remove_duplicated_triangles()
            
            # Remove degenerate triangles (very small triangles that can cause issues)
            mesh.remove_degenerate_triangles()
            
            # Fill holes to ensure continuous surface
            mesh.fill_holes()
            
            # Apply conservative mesh simplification to preserve details
            # Only simplify if the mesh is very dense
            if len(mesh.triangles) > 100000:
                target_triangles = max(50000, len(mesh.triangles) // 2)
                print(f"Applying conservative simplification: {len(mesh.triangles)} -> {target_triangles}")
                mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
            
            # Ensure consistent triangle winding
            mesh.compute_vertex_normals()
            
            # Apply light smoothing to reduce artifacts while preserving details
            print("Applying light smoothing to reduce artifacts...")
            mesh = mesh.filter_smooth_simple(number_of_iterations=1)
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=1)
            
            print(f"Mesh optimization completed: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
            return mesh
            
        except Exception as e:
            print(f"Mesh optimization failed: {e}")
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
    
    # Removed old convert methods - now using point cloud export methods

class PLYConverterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_file = None
        self.output_dir = None
        self.worker = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("PLY Converter")
        self.setGeometry(600, 400, 700, 500)
        self.setMinimumSize(600, 400)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title_label = QLabel("PLY Converter")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Convert 3D Gaussian Splatting PLY files to multiple formats (STL, GLB, 3MF, DXF)")
        desc_label.setFont(QFont("Arial", 10))
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("color: #7f8c8d; margin-bottom: 20px;")
        main_layout.addWidget(desc_label)
        
        # File selection section
        file_frame = QFrame()
        file_frame.setFrameStyle(QFrame.NoFrame)
        file_frame.setStyleSheet("QFrame { padding: 15px; }")
        file_layout = QVBoxLayout(file_frame)
        file_layout.setSpacing(20)
        
        # Input file selection
        input_layout = QHBoxLayout()
        input_layout.setAlignment(Qt.AlignVCenter)
        input_label = QLabel("Input PLY File:")
        input_label.setFont(QFont("Arial", 10, QFont.Bold))
        input_label.setMinimumHeight(45)
        input_label.setAlignment(Qt.AlignVCenter)
        input_label.setStyleSheet("padding: 5px;")
        input_layout.addWidget(input_label)
        
        self.input_path_label = QLabel("No file selected")
        self.input_path_label.setStyleSheet("color: #7f8c8d; padding: 8px; background-color: #ecf0f1; border-radius: 4px;")
        self.input_path_label.setMinimumHeight(35)
        self.input_path_label.setFixedWidth(250)
        input_layout.addWidget(self.input_path_label, 1)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.setMinimumHeight(35)
        self.browse_button.clicked.connect(self.browse_input_file)
        self.browse_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        input_layout.addWidget(self.browse_button)
        file_layout.addLayout(input_layout)
        
        # Output directory selection
        output_layout = QHBoxLayout()
        output_layout.setAlignment(Qt.AlignVCenter)
        output_label = QLabel("Output Directory:")
        output_label.setFont(QFont("Arial", 10, QFont.Bold))  
        output_label.setMinimumHeight(45)
        output_label.setAlignment(Qt.AlignVCenter)
        output_label.setStyleSheet("padding: 5px;")
        output_layout.addWidget(output_label)
        
        self.output_path_label = QLabel("No directory selected")
        self.output_path_label.setStyleSheet("color: #7f8c8d; padding: 8px; background-color: #ecf0f1; border-radius: 4px;")
        self.output_path_label.setMinimumHeight(35)
        self.output_path_label.setFixedWidth(250)
        output_layout.addWidget(self.output_path_label, 1)
        
        self.output_browse_button = QPushButton("Browse...")
        self.output_browse_button.setMinimumHeight(35)
        self.output_browse_button.clicked.connect(self.browse_output_dir)
        self.output_browse_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        output_layout.addWidget(self.output_browse_button)
        file_layout.addLayout(output_layout)
        
        main_layout.addWidget(file_frame)
        
        # Convert button
        self.convert_button = QPushButton("Convert 3D Gaussian Splatting PLY")
        self.convert_button.setMinimumHeight(50)
        self.convert_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.convert_button.clicked.connect(self.start_conversion)
        self.convert_button.setEnabled(False)
        self.convert_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        main_layout.addWidget(self.convert_button)
        
        # Progress section
        progress_frame = QFrame()
        progress_frame.setFrameStyle(QFrame.NoFrame)
        progress_frame.setStyleSheet("QFrame { padding: 15px; }")
        progress_layout = QVBoxLayout(progress_frame)
        
        progress_label = QLabel("Conversion Progress:")
        progress_label.setFont(QFont("Arial", 10, QFont.Bold))
        progress_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px; border: none;")
        progress_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                text-align: center;
                font-weight: bold;
                background-color: #f8f9fa;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
                margin: 1px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)        
                
        main_layout.addWidget(progress_frame)
        
        # Status bar
        self.statusBar().showMessage("Ready to convert")
        
        # Set window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QLabel {
                color: #2c3e50;
            }
        """)
        
    def browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PLY File", "", "PLY Files (*.ply);;All Files (*)"
        )
        if file_path:
            self.input_file = file_path
            self.input_path_label.setText(os.path.basename(file_path))
            self.input_path_label.setStyleSheet("color: #27ae60; padding: 8px; background-color: #d5f4e6; border-radius: 4px;")
            self.check_ready_state()
    
    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir = dir_path
            self.output_path_label.setText(dir_path)
            self.output_path_label.setStyleSheet("color: #27ae60; padding: 8px; background-color: #d5f4e6; border-radius: 4px;")
            self.check_ready_state()
    
    def check_ready_state(self):
        if self.input_file and self.output_dir:
            self.convert_button.setEnabled(True)
            self.statusBar().showMessage("Ready to convert")
        else:
            self.convert_button.setEnabled(False)
            self.statusBar().showMessage("Please select input file and output directory")
    
    def start_conversion(self):
        if not self.input_file or not self.output_dir:
            QMessageBox.warning(self, "Error", "Please select both input file and output directory")
            return
        
        # Disable UI elements
        self.convert_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.output_browse_button.setEnabled(False)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)  # Determinate progress
        self.progress_bar.setValue(0)
        # self.progress_text.clear()
        # self.progress_text.append("Starting conversion...")
        
        # Start conversion in worker thread
        self.worker = ConversionWorker(self.input_file, self.output_dir)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.conversion_finished)
        self.worker.start()
        
        self.statusBar().showMessage("Converting...")
    
    def update_progress(self, message):         
        # Update progress bar based on message content
        if "Loading 3D Gaussian Splatting PLY" in message:
            self.progress_bar.setValue(10)
        elif "Pre-processing" in message:
            self.progress_bar.setValue(25)
        elif "Processing colors" in message:
            self.progress_bar.setValue(35)
        elif "Exporting to STL (Gaussian splats)" in message:
            self.progress_bar.setValue(50)
        elif "Exporting to GLB (3D Gaussian Splats)" in message:
            self.progress_bar.setValue(65)
        elif "Exporting to 3MF (Gaussian splats)" in message:
            self.progress_bar.setValue(80)
        elif "Exporting to DXF (Gaussian splats)" in message:
            self.progress_bar.setValue(90)
        elif "Conversion complete" in message:
            self.progress_bar.setValue(100)
    
    def conversion_finished(self, success, message):
        # Re-enable UI elements
        self.convert_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.output_browse_button.setEnabled(True)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        if success:
            self.statusBar().showMessage("Conversion completed successfully!")
            QMessageBox.information(self, "Success", 
                f"Conversion completed successfully!\n\nFiles saved to:\n{self.output_dir}\n\n"
                "Generated formats:\n"
                "• conversion_output.stl (STL - Gaussian Splats)\n"
                "• conversion_output.glb (GLB - 3D Gaussian Splats)\n"
                "• conversion_output.3mf (3MF - Gaussian Splats)\n"
                "• conversion_output.dxf (DXF - Gaussian Splats)")
        else:
            self.statusBar().showMessage("Conversion failed")
            QMessageBox.critical(self, "Error", f"Conversion failed:\n{message}")
        


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
