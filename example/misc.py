import torch
import numpy as np
from scipy import spatial

def str2dict(ss):
    obj_score = dict()
    for obj_str in ss.split(','):
        obj_s = obj_str.strip()
        if len(obj_s) > 0:
            obj_id = obj_s.split(':')[0].strip()
            obj_s = obj_s.split(':')[1].strip()
            if len(obj_s) > 0:
                obj_score[int(obj_id)] = float(obj_s)
    return obj_score

def cal_score(adi_str, add_str):
    adi_score = str2dict(adi_str)
    add_score = str2dict(add_str)
    add_score[10] = adi_score[10]
    add_score[11] = adi_score[11]
    if 3 in add_score:
        add_score.pop(3)
    if 7 in add_score:
        add_score.pop(7)
    return np.mean(list(add_score.values()))

def printAD(add, adi, name='RAW'):
    print("{}: ADD:{:.5f}, ADI:{:.5f}, ADD(-S):{:.5f}".format(
        name,
    np.sum(list(str2dict(add).values()))/len(str2dict(add)),
    np.sum(list(str2dict(adi).values()))/len(str2dict(adi)),
    cal_score(adi_str=adi, add_str=add)))
    


def box_2D_shape(points, pose, K):
    canonical_homo_pts = torch.tensor(vert2_to_bbox8(points).T, dtype=torch.float32)
    trans_homo = pose @ canonical_homo_pts
    homo_K = torch.zeros((3, 4), dtype=torch.float32)
    homo_K[:3, :3] = torch.tensor(K, dtype=torch.float32)
    bbox_2D = (homo_K @ trans_homo)
    bbox_2D = (bbox_2D[:2] / bbox_2D[2]).T.type(torch.int32)#.tolist()
    return bbox_2D


def vert2_to_bbox8(corner_pts, homo=True):
    pts = list()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if homo:
                    pt = [corner_pts[i, 0], corner_pts[j, 1], corner_pts[k, 2], 1.0]
                else:
                    pt = [corner_pts[i, 0], corner_pts[j, 1], corner_pts[k, 2]]
                pts.append(pt)   
    return np.asarray(pts)

def bbox_to_shape(bbox_2D):
    connect_points = [[0, 2, 3, 1, 0], [0, 4, 6, 2], [2, 3, 7, 6], [6, 4, 5, 7], [7, 3, 1, 5]]
    shape = list()
    for plane in connect_points:
        for idx in plane:
            point = (bbox_2D[idx][0], bbox_2D[idx][1])
            shape.append(point)
    return shape

# def calc_ADDS(gt_pose, pd_pose, obj_model):

def transform_pts_Rt(pts, R, t):
  """Applies a rigid transformation to 3D points.

  :param pts: nx3 ndarray with 3D points.
  :param R: 3x3 ndarray with a rotation matrix.
  :param t: 3x1 ndarray with a translation vector.
  :return: nx3 ndarray with transformed 3D points.
  """
  assert (pts.shape[1] == 3)
  pts_t = R.dot(pts.T) + t.reshape((3, 1))
  return pts_t.T


def add(R_est, t_est, R_gt, t_gt, pts):
  """Average Distance of Model Points for objects with no indistinguishable
  views - by Hinterstoisser et al. (ACCV'12).

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param pts: nx3 ndarray with 3D model points.
  :return: The calculated error.
  """
  pts_est = transform_pts_Rt(pts, R_est, t_est)
  pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
  e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
  return e

def adi(R_est, t_est, R_gt, t_gt, pts):
  """Average Distance of Model Points for objects with indistinguishable views
  - by Hinterstoisser et al. (ACCV'12).

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param t_est: 3x1 ndarray with the estimated translation vector.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :param t_gt: 3x1 ndarray with the ground-truth translation vector.
  :param pts: nx3 ndarray with 3D model points.
  :return: The calculated error.
  """
  pts_est = transform_pts_Rt(pts, R_est, t_est)
  pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

  # Calculate distances to the nearest neighbors from vertices in the
  # ground-truth pose to vertices in the estimated pose.
  nn_index = spatial.cKDTree(pts_est)
  nn_dists, _ = nn_index.query(pts_gt, k=1)

  e = nn_dists.mean()
  return e