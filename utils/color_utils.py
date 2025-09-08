import numpy as np, open3d as o3d

def enhance_colors_vectorized(rgb_colors):
    rgb = rgb_colors[:, :3]
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb,0,1)

def apply_ham_color_enhancement(colors):
    return colors  # placeholder (identity)

def adjust_scales_for_ham(scales):
    return scales  # identity

def transfer_colors_to_mesh(pcd, mesh):
    try:
        pts = np.asarray(pcd.points); cols = np.asarray(pcd.colors)
        verts = np.asarray(mesh.vertices)
        tree = o3d.geometry.KDTreeFlann(pcd)
        out = np.zeros((len(verts),3),dtype=np.float32)
        for i,v in enumerate(verts):
            _,idx,_ = tree.search_knn_vector_3d(v,1)
            if idx: out[i]=cols[idx[0]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(out)
    except Exception as e:
        print("transfer_colors_to_mesh failed:",e)
    return mesh

def transfer_colors_to_mesh_enhanced(pcd, mesh, original_colors):
    try:
        verts = np.asarray(mesh.vertices)
        tree = o3d.geometry.KDTreeFlann(pcd)
        out = np.zeros((len(verts),3),dtype=np.float32)
        for i,v in enumerate(verts):
            k, idx, dist = tree.search_knn_vector_3d(v,5)
            if idx:
                w = 1.0/(np.array(dist)+1e-6); w /= w.sum()
                out[i] = (original_colors[idx][:,:3]*w[:,None]).sum(0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(out,0,1))
    except Exception as e:
        print("transfer_colors_to_mesh_enhanced failed:",e)
    return mesh

def transfer_colors_to_surface_mesh(pcd, mesh, colors_rgba):
    try:
        verts = np.asarray(mesh.vertices)
        tree = o3d.geometry.KDTreeFlann(pcd)
        out = np.zeros((len(verts),3),dtype=np.float32)
        for i,v in enumerate(verts):
            _,idx,_ = tree.search_knn_vector_3d(v,1)
            if idx: out[i]=colors_rgba[idx[0]][:3]
        mesh.vertex_colors = o3d.utility.Vector3dVector(out)
    except Exception as e:
        print("transfer_colors_to_surface_mesh failed:",e)
    return mesh