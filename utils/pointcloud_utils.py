import numpy as np, open3d as o3d

def remove_invalid_points_and_colors(pcd):
    pts = np.asarray(pcd.points)
    mask = np.isfinite(pts).all(axis=1)
    if mask.all():
        return pcd
    new = o3d.geometry.PointCloud()
    new.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        new.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        new.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
    return new

def keep_largest_cluster(pcd, eps=0.02, min_points=10):
    try:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        if labels.size == 0:
            return pcd
        valid = labels >= 0
        if not valid.any():
            return pcd
        u, c = np.unique(labels[valid], return_counts=True)
        largest = u[np.argmax(c)]
        idx = np.where(labels == largest)[0]
        return pcd.select_by_index(idx)
    except Exception as e:
        print("keep_largest_cluster skipped:", e)
        return pcd

def remove_planes(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000,
                  max_planes=3, min_ratio=0.01):
    try:
        remaining = pcd
        for _ in range(max_planes):
            if len(remaining.points) < 50:
                break
            plane_model, inliers = remaining.segment_plane(distance_threshold, ransac_n, num_iterations)
            if not inliers:
                break
            frac = len(inliers) / len(remaining.points)
            if frac < min_ratio:
                break
            remaining = remaining.select_by_index(inliers, invert=True)
        return remaining
    except Exception as e:
        print("remove_planes skipped:", e)
        return pcd

def prepare_point_cloud_for_reconstruction(pcd, keep_all_clusters=True,
                                           plane_removal=True,
                                           cluster_eps=0.02,
                                           cluster_min_points=10):
    pcd = remove_invalid_points_and_colors(pcd)
    if not keep_all_clusters:
        pcd = keep_largest_cluster(pcd, eps=cluster_eps, min_points=cluster_min_points)
    if plane_removal:
        pcd = remove_planes(pcd)
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=60))
        try:
            pcd.orient_normals_consistent_tangent_plane(120)
        except:
            pcd.normalize_normals()
    return pcd

def normalize_point_cloud_for_reconstruction(pcd):
    """
    Return (normalized_pcd, center, scale)
    Center: mean of original points.
    Scale: maximum axis extent after centering (so largest dimension becomes 1).
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        empty = o3d.geometry.PointCloud()
        return empty, np.zeros(3), 1.0
    center = pts.mean(axis=0)
    shifted = pts - center
    extents = shifted.ptp(axis=0)
    scale = extents.max() if extents.max() > 0 else 1.0
    norm_pts = shifted / scale
    norm_pcd = o3d.geometry.PointCloud()
    norm_pcd.points = o3d.utility.Vector3dVector(norm_pts)
    if pcd.has_colors():
        norm_pcd.colors = pcd.colors
    if pcd.has_normals():
        norm_pcd.normals = pcd.normals
    return norm_pcd, center, scale

def advanced_point_cloud_preprocessing(pcd,
                                       voxel_size=None,
                                       target_points=250000,
                                       keep_all_clusters=True,
                                       remove_planes_flag=False):
    """
    Optional richer preprocessing pipeline.
    - Optional voxel downsample (either by explicit voxel_size or adaptive to target_points)
    - Remove invalid points
    - (Optional) plane removal
    - (Optional) keep largest cluster
    - Ensure normals
    """
    pcd = remove_invalid_points_and_colors(pcd)

    # Adaptive voxel downsample if needed
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)
    elif target_points and len(pcd.points) > target_points:
        # heuristic voxel size based on bounding box
        pts = np.asarray(pcd.points)
        bbox = pts.ptp(axis=0)
        mean_extent = bbox.mean()
        # smaller voxel for smaller objects
        adaptive_voxel = mean_extent * 0.005
        pcd = pcd.voxel_down_sample(adaptive_voxel)

    if remove_planes_flag:
        pcd = remove_planes(pcd)

    if not keep_all_clusters:
        pcd = keep_largest_cluster(pcd)

    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=60))
        try:
            pcd.orient_normals_consistent_tangent_plane(100)
        except:
            pcd.normalize_normals()

    return pcd