"""
The code for point-to-plane ICP is modified from the respository https://github.com/pglira/simpleICP/tree/master/python
"""
import time
import torch
import numpy as np
from datetime import datetime
from scipy import spatial, stats

def depth_to_pointcloud(depth, K):
    if not isinstance(depth, torch.Tensor):
        depth = torch.tensor(depth, dtype=torch.float32)
    K = K.squeeze().to(depth.device)
    depth = depth.squeeze()
    
    vs, us = depth.nonzero(as_tuple=True)
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = torch.stack([xs, ys, zs], dim=1)
    return pts


def torch_batch_cov(X):
    """
    calculate covariance
    """
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    cov = X @ X.transpose(-1, -2) / (X.shape[-1] - 1) 
    return cov
    

class PointCloud:
    def __init__(self, pts):
        self.xyz_pts = pts
        self.normals = None
        self.planarity = None
        self.no_points = len(pts)
        self.sel = None
        self.device=pts.device
        self.dtype = pts.dtype

    def select_n_points(self, n):
        if self.no_points > n:
            self.sel = torch.linspace(0, self.no_points-1, n).round().type(torch.int64).to(self.device)
        else:
            self.sel = torch.arange(self.no_points).to(self.device)

    def estimate_normals(self, neighbors):
        self.normals = torch.full((self.no_points, 3), float('nan'), dtype=self.dtype, device=self.device)
        self.planarity = torch.full((self.no_points, ), float('nan'), dtype=self.dtype, device=self.device)
        
        knn_dists = -(self.xyz_pts[self.sel].unsqueeze(1) - self.xyz_pts.unsqueeze(0)).norm(dim=2, p=2) # QxN
        _, idxNN_all_qp = torch.topk(knn_dists, k=neighbors, dim=1)
        
        selected_points = self.xyz_pts[idxNN_all_qp]
        batch_C = torch_batch_cov(selected_points.transpose(-2, -1))
        
        eig_vals, eig_vecs = np.linalg.eig(batch_C.detach().cpu().numpy())
        eig_vals = torch.tensor(eig_vals).to(self.device)
        eig_vecs = torch.tensor(eig_vecs).to(self.device)
        
        _, idx_sort_vals = eig_vals.topk(k=eig_vals.shape[-1], dim=-1) # descending orders, Qx3
        idx_sort_vecs = idx_sort_vals[:, 2:3][..., None].repeat(1, 3, 1) # Qx3x3
        new_eig_vals = torch.gather(eig_vals, dim=1, index=idx_sort_vals).squeeze() # sorted eigen values by descending order
        new_eig_vecs = torch.gather(eig_vecs, dim=2, index=idx_sort_vecs).squeeze() # the vector whose corresponds to the smallest eigen value 
        
        self.normals[self.sel] = new_eig_vecs
        self.planarity[self.sel] = (new_eig_vals[:, 1] - new_eig_vals[:, 2]) / new_eig_vals[:, 0]

    def transform(self, H):
        XInH = PointCloud.euler_coord_to_homogeneous_coord(self.xyz_pts)
        XOutH = (H @ XInH.T).T
        self.xyz_pts = PointCloud.homogeneous_coord_to_euler_coord(XOutH)


    @staticmethod
    def euler_coord_to_homogeneous_coord(XE):
        no_points = XE.shape[0]
        XH = torch.cat([XE, torch.ones(no_points, 1, device=XE.device)], dim=-1)
        return XH

    @staticmethod
    def homogeneous_coord_to_euler_coord(XH):
        XE = torch.stack([XH[:,0]/XH[:,3], XH[:,1]/XH[:,3], XH[:,2]/XH[:,3]], dim=-1)

        return XE

def matching(pcfix, pcmov):
    knn_dists = -(pcfix.xyz_pts[pcfix.sel].unsqueeze(1) - pcmov.xyz_pts.unsqueeze(0)).norm(dim=2, p=2) # QxN
    pcmov.sel = torch.topk(knn_dists, k=1, dim=1)[1].squeeze()
    dxdyxdz = pcmov.xyz_pts[pcmov.sel] - pcfix.xyz_pts[pcfix.sel]
    nxnynz = pcfix.normals[pcfix.sel] # Qx3
    distances = (dxdyxdz * nxnynz).sum(dim=1)

    return distances


def reject(pcfix, pcmov, min_planarity, distances):
    planarity = pcfix.planarity[pcfix.sel]
    med = distances.median()
    sigmad = (distances - torch.median(distances)).abs().median() * 1.4826 # normal

    keep_distance = abs(distances-med) <= 3 * sigmad
    keep_planarity = planarity > min_planarity
    keep = keep_distance & keep_planarity
    
    pcfix.sel = pcfix.sel[keep]
    pcmov.sel = pcmov.sel[keep]
    distances = distances[keep]

    return distances


def estimate_rigid_body_transformation(pcfix, pcmov):
    fix_pts = pcfix.xyz_pts[pcfix.sel]
    dst_normals = pcfix.normals[pcfix.sel]

    mov_pts = pcmov.xyz_pts[pcmov.sel]
    x_mov = mov_pts[:, 0]
    y_mov = mov_pts[:, 1]
    z_mov = mov_pts[:, 2]

    nx_fix = dst_normals[:, 0]
    ny_fix = dst_normals[:, 1]
    nz_fix = dst_normals[:, 2]
   
    A = torch.stack([-z_mov*ny_fix + y_mov*nz_fix,
                     z_mov*nx_fix - x_mov*nz_fix,
                     -y_mov*nx_fix + x_mov*ny_fix,
                     nx_fix,  ny_fix, nz_fix], dim=-1).detach().cpu().numpy()
    
    b = (dst_normals * (fix_pts - mov_pts)).sum(dim=1).detach().cpu().numpy() # Sx3 -> S
    
    x, _, _, _ = np.linalg.lstsq(A, b)

    A = torch.tensor(A).to(pcfix.device)
    b = torch.tensor(b).to(pcfix.device)
    x = torch.tensor(x).to(pcfix.device)
    
    x = torch.clamp(x, torch.tensor(-0.5, device=pcfix.device),  torch.tensor(0.5, device=pcfix.device))

    residuals = A @ x - b
    
    R =  euler_angles_to_linearized_rotation_matrix(x[0], x[1], x[2])
    t = x[3:6]
    H = create_homogeneous_transformation_matrix(R, t)

    return H, residuals


def euler_angles_to_linearized_rotation_matrix(alpha1, alpha2, alpha3):
    dR = torch.tensor([[      1, -alpha3,  alpha2],
                   [ alpha3,       1, -alpha1],
                   [-alpha2,  alpha1,       1]]).to(alpha1.device)

    return dR


def create_homogeneous_transformation_matrix(R, t):
    H = torch.tensor([[R[0,0], R[0,1], R[0,2], t[0]],
                      [R[1,0], R[1,1], R[1,2], t[1]],
                      [R[2,0], R[2,1], R[2,2], t[2]],
                      [     0,      0,      0,    1]]).to(R.device)

    return H

def check_convergence_criteria(distances_new, distances_old, min_change):
    def change(new, old):
        return torch.abs((new - old) / old * 100)
    
    change_of_mean = change(torch.mean(distances_new), torch.mean(distances_old))
    change_of_std = change(torch.std(distances_new), torch.std(distances_old))

    return True if change_of_mean < min_change and change_of_std < min_change else False


def sim_icp(X_fix, X_mov, correspondences=1000, neighbors=10, min_planarity=0.3, min_change=1, max_iterations=100, verbose=False):
    if len(X_fix) < neighbors:
        return torch.eye(4, dtype=X_fix.dtype).to(X_fix.device)
    pcfix = PointCloud(X_fix)
    pcmov = PointCloud(X_mov)
    
    pcfix.select_n_points(correspondences)
    sel_orig = pcfix.sel

    pcfix.estimate_normals(neighbors)  # 500ms

    H = torch.eye(4, dtype=X_fix.dtype).to(X_fix.device)
    residual_distances = []
    
    for i in range(0, max_iterations):
        initial_distances = matching(pcfix, pcmov) # 146ms
        # Todo Change initial_distances without return argument
        initial_distances = reject(pcfix, pcmov, min_planarity, initial_distances) # 3.3ms
        dH, residuals = estimate_rigid_body_transformation(pcfix, pcmov)
        residual_distances.append(residuals)
        pcmov.transform(dH)

        H = dH @ H
        pcfix.sel = sel_orig

        if i > 0:
            if check_convergence_criteria(residual_distances[i], residual_distances[i-1], min_change):
                break
    return H