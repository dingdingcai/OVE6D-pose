
import math
import torch
from pytorch3d.transforms import euler_angles_to_matrix

RANDOM_SEED = 2021       # for reproduce the results of evaluation

RENDER_WIDTH = 640       # the width of rendered images
RENDER_HEIGHT = 480      # the height of rendered images
RENDER_DIST = 5          # the radius distance factor of uniform sampling relative to object diameter.
RENDER_NUM_VIEWS = 4000  # the number of uniform sampling views from a sphere
MODEL_SCALING = 1.0/1000 # TLESS object model scale from millimeter to meter

ZOOM_SIZE = 128          # the target zooming size
ZOOM_MODE = 'bilinear'   # the target zooming mode (bilinear or nearest)
ZOOM_DIST_FACTOR = 0.01  # the distance factor of zooming (relative to object diameter)
DATASET_NAME = ''
SAVE_FTMAP = True        # store the latent feature maps of viewpoint (for rotation regression)

# FUSE_RCNN_VPSIM = False # whether fusing the rcnn detection confidence with the retrieved viewpoint cosine similarity score

HEMI_ONLY = True
USE_ICP = True
ICP_neighbors = 10         
ICP_min_planarity = 0.2     
ICP_max_iterations = 20     # max iterations for ICP
ICP_correspondences = 1000  # the number of points selected for iteration

# VP_NMS_DIST = 10 # the distance of viewpoint hypothesis in degrees
# VP_MAX_NN = -1   # the maximum number of nearest neighors for each viewpoint (-1 for all viewpoints)


# USE_VPNMS = False
VP_NUM_TOPK = 50   # the number of viewpoint retrievals
POSE_NUM_TOPK = 5  # the number of pose hypotheses


DATA_PATH = '/home/hdd/Dingding/Dataspace/BOP_Dataset'


def BOP_REF_POSE(ref_R):
    unsqueeze = False
    if not isinstance(ref_R, torch.Tensor):
        ref_R = torch.tensor(ref_R, dtype=torch.float32)
    if ref_R.dim() == 2:
        ref_R = ref_R.unsqueeze(0)
        unsqueeze = True
    assert ref_R.dim() == 3 and ref_R.shape[-1] == 3, "rotation R dim must be B x 3 x 3"
    CAM_REF_POSE = torch.tensor((
                (-1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
            ), dtype=torch.float32)

    XR = euler_angles_to_matrix(torch.tensor([180/180*math.pi, 0, 0]), "XYZ")
    R = (XR[None, ...] @ ref_R.clone())
    R = CAM_REF_POSE.T[None, ...] @ R @ CAM_REF_POSE[None, ...]
    if unsqueeze:
        R = R.squeeze(0)
    return R

def POSE_TO_BOP(ref_R):
    unsqueeze = False
    if not isinstance(ref_R, torch.Tensor):
        ref_R = torch.tensor(ref_R, dtype=torch.float32)
    if ref_R.dim() == 2:
        ref_R = ref_R.unsqueeze(0)
        unsqueeze = True
    assert ref_R.dim() == 3 and ref_R.shape[-1] == 3, "rotation R dim must be B x 3 x 3"
    CAM_REF_POSE = torch.tensor((
                (-1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
            ), dtype=torch.float32)
    XR = euler_angles_to_matrix(torch.tensor([180/180*math.pi, 0, 0]), "XYZ")
    R = XR[None, ...] @ ref_R
    
    R = CAM_REF_POSE[None, ...] @ R @ CAM_REF_POSE.T[None, ...]
    if unsqueeze:
        R = R.squeeze(0)
    return R

