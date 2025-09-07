# save as create_flower.py
import numpy as np
import open3d as o3d

def create_flower_ply():
    # Create flower components
    vertices = []
    faces = []
    colors = []
    
    # Stem (green)
    stem_height = 1.0
    stem_radius = 0.05
    stem_segments = 12
    
    # Create stem vertices
    for i in range(stem_segments + 1):
        angle = 2 * np.pi * i / stem_segments
        x = stem_radius * np.cos(angle)
        y = stem_radius * np.sin(angle)
        z = np.linspace(0, stem_height, stem_segments + 1)[i]
        vertices.append([x, y, z])
        colors.append([0.2, 0.8, 0.2])  # Green stem
    
    # Connect stem faces
    for i in range(stem_segments):
        faces.append([i, (i + 1) % stem_segments, stem_segments + (i + 1) % stem_segments])
        faces.append([i, stem_segments + (i + 1) % stem_segments, stem_segments + i])
    
    # Flower center (yellow)
    center_radius = 0.1
    center_height = stem_height + 0.05
    center_segments = 8
    
    # Add center vertices
    start_idx = len(vertices)
    for i in range(center_segments):
        angle = 2 * np.pi * i / center_segments
        x = center_radius * np.cos(angle)
        y = center_radius * np.sin(angle)
        vertices.append([x, y, center_height])
        colors.append([1.0, 1.0, 0.0])  # Yellow center
    
    # Add center top point
    vertices.append([0, 0, center_height + 0.05])
    colors.append([1.0, 1.0, 0.0])
    top_idx = len(vertices) - 1
    
    # Create center faces
    for i in range(center_segments):
        faces.append([start_idx + i, start_idx + (i + 1) % center_segments, top_idx])
    
    # Petals (red/pink)
    petal_count = 8
    petal_length = 0.3
    petal_width = 0.15
    petal_height = stem_height + 0.1
    
    for i in range(petal_count):
        angle = 2 * np.pi * i / petal_count
        # Petal base points
        x1 = (center_radius + 0.02) * np.cos(angle)
        y1 = (center_radius + 0.02) * np.sin(angle)
        
        # Petal tip
        x2 = (center_radius + petal_length) * np.cos(angle)
        y2 = (center_radius + petal_length) * np.sin(angle)
        
        # Add petal vertices
        petal_start = len(vertices)
        vertices.append([x1, y1, center_height])  # Base left
        vertices.append([x2, y2, petal_height])   # Tip
        vertices.append([x1, y1, center_height])  # Base right (same as left for simplicity)
        
        # Pink color for petals
        colors.extend([[1.0, 0.6, 0.8], [1.0, 0.4, 0.7], [1.0, 0.6, 0.8]])
        
        # Add petal face
        faces.append([petal_start, petal_start + 1, petal_start + 2])
    
    # Convert to Open3D format
    vertices = np.array(vertices)
    faces = np.array(faces)
    colors = np.array(colors)
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save as PLY
    o3d.io.write_point_cloud("flower.ply", pcd)
    print("Created flower.ply with colorful flower model")
    
    # Also create a mesh version for better visualization
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh("flower_mesh.ply", mesh)
    print("Created flower_mesh.ply with mesh version")

if __name__ == "__main__":
    create_flower_ply()