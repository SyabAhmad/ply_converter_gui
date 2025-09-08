import numpy as np, trimesh, open3d as o3d

# -----------------------------
# Geometry creation / transforms
# -----------------------------
def create_ellipsoid_mesh(lat_segments: int, lon_segments: int):
    """
    Create a UV sphere (can be scaled later to ellipsoid).
    Returns (vertices, faces)
    """
    verts = []
    faces = []
    for lat in range(lat_segments + 1):
        theta = lat * np.pi / lat_segments
        st = np.sin(theta)
        ct = np.cos(theta)
        for lon in range(lon_segments + 1):
            phi = lon * 2 * np.pi / lon_segments
            cp = np.cos(phi)
            sp = np.sin(phi)
            verts.append([st * cp, st * sp, ct])
    verts = np.asarray(verts, dtype=np.float32)
    row = lon_segments + 1
    for lat in range(lat_segments):
        for lon in range(lon_segments):
            v0 = lat * row + lon
            v1 = v0 + 1
            v2 = (lat + 1) * row + lon
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return verts, np.asarray(faces, dtype=np.int32)

def transform_ellipsoid(base_vertices, position, scale, rotation):
    """
    Apply scale, quaternion rotation (w,x,y,z), and translation.
    """
    w, x, y, z = rotation
    # Proper quaternion to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x*x + y*y)]
    ], dtype=np.float32)
    return (base_vertices * scale) @ R.T + position

# -----------------------------
# Smoothing / cleanup
# -----------------------------
def smooth_mesh_for_ham(mesh: trimesh.Trimesh):
    """
    Convert to Open3D, apply smoothing filters, back to trimesh.
    """
    try:
        o3 = o3d.geometry.TriangleMesh()
        o3.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3.triangles = o3d.utility.Vector3iVector(mesh.faces)
        if mesh.visual.kind == 'vertex' and hasattr(mesh.visual, 'vertex_colors'):
            cols = mesh.visual.vertex_colors[:, :3] / 255.0
            o3.vertex_colors = o3d.utility.Vector3dVector(cols)
        o3 = o3.filter_smooth_simple(2)
        o3 = o3.filter_smooth_laplacian(3)
        o3 = o3.filter_smooth_taubin(2)
        o3.compute_vertex_normals()
        return trimesh.Trimesh(vertices=np.asarray(o3.vertices),
                               faces=np.asarray(o3.triangles),
                               process=True)
    except Exception as e:
        print("smooth_mesh_for_ham failed:", e)
        return mesh

def smooth_mesh_for_ham_o3d(o3: o3d.geometry.TriangleMesh):
    try:
        o3 = o3.filter_smooth_simple(3)
        o3 = o3.filter_smooth_laplacian(4)
        o3 = o3.filter_smooth_taubin(3)
        o3.compute_vertex_normals()
    except Exception as e:
        print("smooth_mesh_for_ham_o3d failed:", e)
    return o3

# -----------------------------
# Topology optimization
# -----------------------------
def optimize_mesh_topology(mesh):
    """
    Deduplicate / remove degenerate geometry. Returns trimesh.Trimesh.
    """
    try:
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                                  faces=np.asarray(mesh.triangles),
                                  process=False)
        else:
            tri = mesh
        if hasattr(tri, "remove_unreferenced_vertices"):
            tri.remove_unreferenced_vertices()
        if hasattr(tri, "remove_duplicated_vertices"):
            tri.remove_duplicated_vertices()
        if hasattr(tri, "remove_duplicated_triangles"):
            tri.remove_duplicated_triangles()
        if hasattr(tri, "remove_degenerate_triangles"):
            tri.remove_degenerate_triangles()
        if hasattr(tri, "remove_infinite_values"):
            tri.remove_infinite_values()
        if hasattr(tri, "rezero"):
            tri.rezero()
        tri.process(validate=True)
        return tri
    except Exception as e:
        print("optimize_mesh_topology failed:", e)
        return mesh

# -----------------------------
# Distance-based pruning
# -----------------------------
def prune_mesh_by_point_distance(tri_mesh,
                                 pcd,
                                 distance_factor=4.0,
                                 absolute_max=None,
                                 keep_ratio=0.98):
    """
    Remove vertices whose distance to nearest point cloud point is too large.
    Strategy:
      - Compute nearest neighbor distance for each vertex.
      - Base threshold = median_distance * distance_factor.
      - Optionally clamp with absolute_max.
      - Also enforce that at least keep_ratio of vertices remain (quantile fallback).
    """
    try:
        if isinstance(tri_mesh, o3d.geometry.TriangleMesh):
            verts = np.asarray(tri_mesh.vertices)
            faces = np.asarray(tri_mesh.triangles)
            to_o3d = True
        else:
            verts = np.asarray(tri_mesh.vertices)
            faces = np.asarray(tri_mesh.faces)
            to_o3d = False

        if len(verts) == 0 or len(pcd.points) == 0:
            return tri_mesh

        pts = np.asarray(pcd.points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        dists = np.zeros(len(verts), dtype=np.float32)
        for i, v in enumerate(verts):
            _, idx, dist = kdtree.search_knn_vector_3d(v, 1)
            dists[i] = np.sqrt(dist[0]) if idx else np.inf

        finite = np.isfinite(dists)
        if not finite.any():
            return tri_mesh

        med = np.median(dists[finite])
        thresh = med * distance_factor
        if absolute_max is not None:
            thresh = min(thresh, absolute_max)

        # Ensure keep_ratio
        q_keep = np.quantile(dists[finite], keep_ratio)
        thresh = max(thresh, q_keep)

        keep_mask = dists <= thresh
        if keep_mask.sum() < 10:
            # Abort if would delete almost everything
            return tri_mesh

        index_map = -np.ones(len(verts), dtype=np.int64)
        index_map[keep_mask] = np.cumsum(keep_mask) - 1

        new_faces = []
        for f in faces:
            if keep_mask[f].all():
                new_faces.append(index_map[f])
        new_faces = np.asarray(new_faces, dtype=np.int32)

        new_verts = verts[keep_mask]

        if to_o3d:
            out = o3d.geometry.TriangleMesh()
            out.vertices = o3d.utility.Vector3dVector(new_verts)
            out.triangles = o3d.utility.Vector3iVector(new_faces)
            out.remove_degenerate_triangles()
            out.remove_duplicated_triangles()
            out.remove_unreferenced_vertices()
            out.remove_duplicated_vertices()
            out.compute_vertex_normals()
            return out
        else:
            return trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=True)
    except Exception as e:
        print("prune_mesh_by_point_distance failed:", e)
    return tri_mesh

# -----------------------------
# Remove flat components
# -----------------------------
def remove_flat_mesh_components(tri_mesh,
                                thickness_tol=0.006,
                                area_ratio_threshold=0.03):
    """
    Remove connected components that are (a) thin (smallest bbox axis < thickness_tol)
    AND (b) represent at least area_ratio_threshold of total area (large flat plates).
    """
    try:
        if isinstance(tri_mesh, o3d.geometry.TriangleMesh):
            tri = trimesh.Trimesh(vertices=np.asarray(tri_mesh.vertices),
                                  faces=np.asarray(tri_mesh.triangles),
                                  process=False)
            to_o3d = True
        else:
            tri = tri_mesh
            to_o3d = False

        if tri.faces.shape[0] == 0:
            return tri_mesh

        components = tri.split(only_watertight=False)
        if len(components) <= 1:
            return tri_mesh

        total_area = sum(c.area for c in components if c.faces.shape[0] > 0)
        keep = []
        for c in components:
            if c.faces.shape[0] == 0:
                continue
            # Bounding box extents
            ext = c.extents  # lengths along bbox axes
            if ext.size != 3:
                keep.append(c)
                continue
            min_axis = np.min(ext)
            area_ratio = c.area / total_area if total_area > 0 else 0
            # Removal condition
            if (min_axis < thickness_tol) and (area_ratio >= area_ratio_threshold):
                # Drop it (do not add)
                continue
            keep.append(c)

        if not keep:
            # If everything removed, revert
            return tri_mesh

        merged = trimesh.util.concatenate(keep)
        if to_o3d:
            out = o3d.geometry.TriangleMesh()
            out.vertices = o3d.utility.Vector3dVector(merged.vertices)
            out.triangles = o3d.utility.Vector3iVector(merged.faces)
            out.remove_degenerate_triangles()
            out.remove_duplicated_triangles()
            out.remove_unreferenced_vertices()
            out.remove_duplicated_vertices()
            out.compute_vertex_normals()
            return out
        else:
            return merged
    except Exception as e:
        print("remove_flat_mesh_components failed:", e)
    return tri_mesh
