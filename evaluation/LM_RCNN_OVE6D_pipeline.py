
import os
import cv2
import sys
import json
# import yaml
import time
import torch
import warnings
import numpy as np
from PIL import Image
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
# from detectron2.structures import BoxMode, BitMasks
# from detectron2.utils.visualizer import Visualizer
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from pycocotools import mask as mask_util
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor



from os.path import join as pjoin
from bop_toolkit_lib import inout
warnings.filterwarnings("ignore")


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from lib import rendering, network

from dataset import LineMOD_Dataset
from evaluation import utils
from evaluation import config as cfg

gpu_id = 0
# gpu_id = 1

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ['EGL_DEVICE_ID'] = str(gpu_id)
DEVICE = torch.device('cuda')


datapath = Path(cfg.DATA_PATH)

eval_dataset = LineMOD_Dataset.Dataset(datapath / 'lm')

################################################# MASK-RCNN Segmentation ##################################################################
rcnnIdx_to_lmIds_dict = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:11, 11:12, 12:13, 13:14, 14:15}
rcnnIdx_to_lmCats_dict ={0:'Ape', 1:'Benchvice', 3:'Bowl', 4:'Camera', 5:'Can', 6:'Cat', 7:'Cup', 8:'Driller', 
                        9:'Duck', 10:'Eggbox', 11:'Glue', 12:'Holepunch', 13:'Iron', 14:'Lamp', 15:'Phone'}
rcnn_cfg = get_cfg()
# rcnn_cfg.INPUT.MASK_FORMAT = 'bitmask'
rcnn_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
rcnn_cfg.MODEL.WEIGHTS = pjoin(base_path, 
                            'checkpoints', 
                            'LMO-maskrcnn', 
                            'lm_model_final.pth')
rcnn_cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(rcnnIdx_to_lmCats_dict)
rcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001 # the predicted category scores
predictor = DefaultPredictor(rcnn_cfg)
################################################# MASK-RCNN Segmentation ##################################################################




cfg.DATASET_NAME = 'lm'        # dataset name
# cfg.ZOOM_DIST_FACTOR  = 0.01   # zooming distance factor relative to object diameter (i.e. zoom_dist = 8 * diameter)
# cfg.VP_MAX_NN = 500            # only consider the top 500 neighbors at most for each viewpoint (-1 for all neighbors)
cfg.RENDER_WIDTH = eval_dataset.cam_width    # the width of rendered images
cfg.RENDER_HEIGHT = eval_dataset.cam_height  # the height of rendered images
# cfg.ZOOM_MODE = 'bilinear'
# cfg.USE_ICP = False
# cfg.USE_VPNMS = False
# cfg.RENDER_NUM_VIEWS = 4000

# cfg.VP_NUM_TOPK = 50  # the retrieval number of viewpoint 
# cfg.RANK_NUM_TOPK = 5  # the ranking number of full 3D orientation 


cfg.HEMI_ONLY = True

ckpt_file = pjoin(base_path, 
                'checkpoints', 
                'OVE6D-weight',
                "pose_model_50_121526_11_02-05:39:36_0.0046_0.0198_5.3.pth"
                )
model_net = network.OVE6D().to(DEVICE)

model_net.load_state_dict(torch.load(ckpt_file), strict=True)
model_net.eval()

codebook_saving_dir = pjoin(base_path,'evaluation/object_codebooks',
                            cfg.DATASET_NAME, 
                            'zoom_{}'.format(cfg.ZOOM_DIST_FACTOR), 
                            'views_{}'.format(str(cfg.RENDER_NUM_VIEWS)))


object_codebooks = utils.OVE6D_codebook_generation(codebook_dir=codebook_saving_dir, 
                                                    model_func=model_net,
                                                    dataset=eval_dataset, 
                                                    config=cfg, 
                                                    device=DEVICE)
raw_pred_results = list()
icp1_pred_results = list()
icpk_pred_results = list()
raw_pred_runtime = list()
icp1_pred_runtime = list()
icpk_pred_runtime = list()

rcnn_gt_results = dict()
rcnn_pd_results = dict()

test_data_dir = datapath / 'lm' / 'test'          # path to the test dataset of BOP
eval_dir = pjoin(base_path, 'evaluation/bop_pred_results/LM')

raw_file_mode = "raw-sampleN{}-viewpointK{}-poseP{}-rcnn_lm-test.csv"
if cfg.USE_ICP:
    icp1_file_mode = "icp1-sampleN{}-viewpointK{}-poseP{}-nbr{}-itr{}-pts{}-pla{}-rcnn_lm-test.csv"
    icpk_file_mode = "icpk-sampleN{}-viewpointK{}-poseP{}-nbr{}-itr{}-pts{}-pla{}-rcnn_lm-test.csv"


obj_renderer = rendering.Renderer(width=cfg.RENDER_WIDTH, height=cfg.RENDER_HEIGHT)

if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

# single_proposal_icp_cost = list()
# single_proposal_raw_cost = list()

img_read_cost = list()
bg_cost = list()
zoom_cost = list()
rot_cost = list()
tsl_cost = list()
# single_proposal_icp_cost = list()
# single_proposal_sum_raw_cost = list()
# single_proposal_sum_icp_cost = list()

raw_syn_render_cost = list()
raw_selection_cost = list()
raw_postprocess_cost = list()

icp1_refinement_cost = list()
icpk_refinement_cost = list()

icpk_syn_render_cost = list()
icpk_selection_cost = list()
icpk_postprocess_cost = list()

for scene_id in sorted(os.listdir(test_data_dir)):
    tar_obj_id = int(scene_id)
    # if tar_obj_id not in [3, 7]:  # skip these two objects
    #     continue

    scene_dir = pjoin(test_data_dir, scene_id)
    if not os.path.isdir(scene_dir):
        continue        
    cam_info_file = pjoin(scene_dir, 'scene_camera.json')
    with open(cam_info_file, 'r') as cam_f:
        scene_camera_info = json.load(cam_f)
    
    gt_pose_file = os.path.join(scene_dir, 'scene_gt.json')
    with open(gt_pose_file, 'r') as pose_f:
        pose_anno = json.load(pose_f)

    rgb_dir = pjoin(scene_dir, 'rgb')
    depth_dir = pjoin(scene_dir, 'depth')
    mask_dir = os.path.join(scene_dir, 'mask_visib')
    rcnn_runtime = list()
    view_runtime = list()
    for rgb_png in sorted(os.listdir(rgb_dir)):
        if not rgb_png.endswith('.png'):
            continue
        view_id_str = rgb_png.split('.')[0]
        view_id = int(view_id_str)
        view_timer = time.time()
        
        ###################### read gt mask ##########################
        # target_gt_masks = dict()
        # view_gt_poses = pose_anno[str(view_id)]
        # for ix, gt_obj in enumerate(view_gt_poses):
        #     gt_obj_id = gt_obj['obj_id']
        #     mask_file = os.path.join(mask_dir, "{:06d}_{:06d}.png".format(view_id, ix))
        #     gt_msk = torch.tensor(cv2.imread(mask_file, 0)).type(torch.bool)
        #     target_gt_masks[gt_obj_id] = gt_msk
        #     if gt_obj_id not in rcnn_gt_results:
        #         rcnn_gt_results[gt_obj_id] = 0
        #     rcnn_gt_results[gt_obj_id] += 1
        ###################### read gt mask ##########################    
        
        ###################### object segmentation ######################
        img_name = "{:06d}.png".format(view_id)
        rgb_file = os.path.join(rgb_dir, img_name)
        rgb_img = cv2.imread(rgb_file)
        imread_cost = time.time() - view_timer
        img_read_cost.append(imread_cost)

        rcnn_timer = time.time()
        output = predictor(rgb_img)
        rcnn_pred_ids = output["instances"].pred_classes
        rcnn_pred_masks = output["instances"].pred_masks
        rcnn_pred_scores = output["instances"].scores
        # rcnn_pred_bboxes = output["instances"].pred_boxes
        rcnn_cost = time.time() - rcnn_timer
        rcnn_runtime.append(rcnn_cost)
        ###################### object segmentation ######################

        obj_masks = rcnn_pred_masks # NxHxW

        view_cam_info = scene_camera_info[str(view_id)]  # scene camera information        
        depth_file = pjoin(depth_dir, "{:06d}.png".format(view_id))
        view_depth = torch.tensor(np.array(Image.open(depth_file)), dtype=torch.float32) # HxW
        view_depth *= view_cam_info['depth_scale']
        view_depth *= cfg.MODEL_SCALING # convert to meter scale from millimeter scale
        view_camK = torch.tensor(view_cam_info['cam_K'], dtype=torch.float32).view(3, 3)[None, ...] # 1x3x3
        
        cam_K = view_camK.to(DEVICE)
        view_depth = view_depth.to(DEVICE)
        obj_depths = view_depth[None, ...] * obj_masks
        
        tar_obj_codebook = object_codebooks[tar_obj_id]
        tar_rcnn_d = tar_obj_id - 1
        tar_obj_depths = obj_depths[tar_rcnn_d==rcnn_pred_ids]
        tar_obj_masks = rcnn_pred_masks[tar_rcnn_d==rcnn_pred_ids]
        tar_obj_scores = rcnn_pred_scores[tar_rcnn_d==rcnn_pred_ids]
        
        if len(tar_obj_scores) > 0:
            mask_pixel_count = tar_obj_masks.view(tar_obj_masks.size(0), -1).sum(dim=1)
            valid_idx = (mask_pixel_count >= 100)
            if valid_idx.sum() == 0:
                mask_visib_ratio = mask_pixel_count / mask_pixel_count.max()
                valid_idx = mask_visib_ratio >= 0.05

            tar_obj_masks = tar_obj_masks[valid_idx]
            tar_obj_depths = tar_obj_depths[valid_idx]
            tar_obj_scores = tar_obj_scores[valid_idx]
        
            pose_ret = utils.OVE6D_rcnn_full_pose(model_func=model_net, 
                                                obj_depths=tar_obj_depths,
                                                obj_masks=tar_obj_masks,
                                                obj_rcnn_scores=tar_obj_scores,
                                                obj_codebook=tar_obj_codebook, 
                                                cam_K=cam_K,
                                                config=cfg, 
                                                device=DEVICE,
                                                obj_renderer=obj_renderer)
            select_rcnn_idx = pose_ret['rcnn_idx']
            rcnn_pd_mask = tar_obj_masks[select_rcnn_idx].cpu()
            rcnn_pd_score = tar_obj_scores[select_rcnn_idx].cpu()
            raw_pred_results.append({'time': pose_ret['raw_time'],
                                        'scene_id': int(scene_id),
                                        'im_id': int(view_id),
                                        'obj_id': int(tar_obj_id),
                                        'score': pose_ret['raw_score'].squeeze().numpy(), 
                                        'R': cfg.POSE_TO_BOP(pose_ret['raw_R']).squeeze().numpy(),
                                        't': pose_ret['raw_t'].squeeze().numpy() * 1000.0}) # convert estimated pose to BOP format
            
            bg_cost.append(pose_ret['bg_time'])
            zoom_cost.append(pose_ret['zoom_time'])
            rot_cost.append(pose_ret['rot_time'])
            tsl_cost.append(pose_ret['tsl_time'])

            raw_pred_runtime.append(pose_ret['raw_time'])
            raw_syn_render_cost.append(pose_ret['raw_syn_time'])
            raw_selection_cost.append(pose_ret['raw_select_time'])
            raw_postprocess_cost.append(pose_ret['raw_postp_time'])

            # single_proposal_raw_cost.append(pose_ret['top1_raw_time'])
            if cfg.USE_ICP:
                icp1_refinement_cost.append(pose_ret['icp1_ref_time'])
                icp1_pred_runtime.append(pose_ret['icp1_rawicp_time'])

                icpk_syn_render_cost.append(pose_ret['icpk_syn_time'])
                icpk_selection_cost.append(pose_ret['icpk_select_time'])
                icpk_postprocess_cost.append(pose_ret['icpk_postp_time'])

                icpk_refinement_cost.append(pose_ret['icpk_ref_time'])
                icpk_pred_runtime.append(pose_ret['icpk_rawicp_time'])
        
                icp1_pred_results.append({'time': pose_ret['icp1_rawicp_time'],
                                            'scene_id': int(scene_id),
                                            'im_id': int(view_id),
                                            'obj_id': int(tar_obj_id),
                                            'score': pose_ret['icp1_score'].squeeze().numpy(), 
                                            'R': cfg.POSE_TO_BOP(pose_ret['icp1_R']).squeeze().numpy(),
                                            't': pose_ret['icp1_t'].squeeze().numpy() * 1000.0})
                
                icpk_pred_results.append({'time': pose_ret['icpk_rawicp_time'],
                                            'scene_id': int(scene_id),
                                            'im_id': int(view_id),
                                            'obj_id': int(tar_obj_id),
                                            'score': pose_ret['icpk_score'].squeeze().numpy(), 
                                            'R': cfg.POSE_TO_BOP(pose_ret['icpk_R']).squeeze().numpy(),
                                            't': pose_ret['icpk_t'].squeeze().numpy() * 1000.0})
                
                
                
        view_runtime.append(time.time() - view_timer)
        if (view_id+1) % 100 == 0:
            print('scene:{}, image: {}, rcnn:{:.3f}, image_cost:{:.3f}, raw_t:{:.3f}, icp1_t:{:.3f}, icpk_t:{:.3f}'.format(
                    int(scene_id), view_id+1, np.mean(rcnn_runtime), np.mean(view_runtime), 
                    np.mean(raw_pred_runtime), np.mean(icp1_pred_runtime), np.mean(icpk_pred_runtime)))

            # print(('[{}/{}] \n' + 
            #         'view:{:.3f}, read:{:.3f}, rcnn:{:.3f}, ' +
            #         'bg:{:.3f}, zoom:{:.3f}, rot:{:.3f}, tsl:{:.3f}, \n' +
            #         ' raw_sum:{:.3f}, raw_syn:{:.3f}, raw_sele:{:.3f}, raw_post:{:.3f}, \n' + 
            #         'icp1_sum:{:.3f}, \n' + 
            #         'icpk_sum:{:.3f}, icpk_syn:{:.3f}, icpk_sele:{:.3f}, icpk_post:{:.3f}, \n' +
            #         'icp1_ref:{:.3f}, raw_icp1:{:.3f}, \n' +  
            #         'icpk_ref:{:.3f}, raw_icpk:{:.3f}'
            #     ).format(int(scene_id), view_id+1, 
            #             np.mean(view_runtime), np.mean(img_read_cost), np.mean(rcnn_runtime),  
            #             np.mean(bg_cost), np.mean(zoom_cost), np.mean(rot_cost), np.mean(tsl_cost), 
            #             np.mean(raw_pred_runtime), np.mean(raw_syn_render_cost), np.mean(raw_selection_cost), np.mean(raw_postprocess_cost), 
            #             np.mean(icp1_pred_runtime), np.mean(icpk_pred_runtime), np.mean(icpk_syn_render_cost), np.mean(icpk_selection_cost), np.mean(icpk_postprocess_cost), 
            #             np.mean(icp1_refinement_cost), np.mean(icp1_pred_runtime), 
            #             np.mean(icpk_refinement_cost), np.mean(icpk_pred_runtime),     
            # ))
    print('{}, {}'.format(scene_id, time.strftime('%m_%d-%H:%M:%S', time.localtime())))

rawk_eval_file = pjoin(eval_dir, raw_file_mode.format(
                        cfg.RENDER_NUM_VIEWS, cfg.VP_NUM_TOPK, cfg.POSE_NUM_TOPK))
inout.save_bop_results(rawk_eval_file, raw_pred_results)

mean_raw_time = np.mean(raw_pred_runtime)
print('raw_mean_runtime: {:.4f}, saving to {}'.format(mean_raw_time, rawk_eval_file))
    
if cfg.USE_ICP:
    icp1_eval_file = pjoin(eval_dir, icp1_file_mode.format(
        cfg.RENDER_NUM_VIEWS, cfg.VP_NUM_TOPK, cfg.POSE_NUM_TOPK,
        cfg.ICP_neighbors, cfg.ICP_max_iterations, cfg.ICP_correspondences, cfg.ICP_min_planarity,
        ))
    icpk_eval_file = pjoin(eval_dir, icpk_file_mode.format(
        cfg.RENDER_NUM_VIEWS, cfg.VP_NUM_TOPK, cfg.POSE_NUM_TOPK,
        cfg.ICP_neighbors, cfg.ICP_max_iterations, cfg.ICP_correspondences, cfg.ICP_min_planarity,
        ))
    inout.save_bop_results(icp1_eval_file, icp1_pred_results)
    inout.save_bop_results(icpk_eval_file, icpk_pred_results)

    mean_icp1_time = np.mean(icp1_pred_runtime)
    mean_icpk_time = np.mean(icpk_pred_runtime)
    print('icp1_mean_runtime: {:.4f}, saving to {}'.format(mean_icp1_time, icp1_eval_file))
    print('icpk_mean_runtime: {:.4f}, saving to {}'.format(mean_icpk_time, icpk_eval_file))

del obj_renderer
