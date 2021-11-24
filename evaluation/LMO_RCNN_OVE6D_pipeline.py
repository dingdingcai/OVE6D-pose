
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
rcnnIdx_to_lmoIds_dict = {0:1, 1:5, 2:6, 3:8, 4:9, 5:10, 6:11, 7:12}
rcnnIdx_to_lmoCats_dict = {0:'Ape', 1:'Can', 2:'Cat', 3:'Driller', 4:'Duck', 5:'Eggbox', 6:'Glue', 7:'Holepunch'}
catId_to_catName_dict = {1:'Ape', 5:'Can', 6:'Cat', 8:'Driller', 9:'Duck', 10:'Eggbox', 11:'Glue', 12:'Holepunch'}
rcnn_cfg = get_cfg()
rcnn_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
rcnn_cfg.MODEL.WEIGHTS = pjoin(base_path, 'checkpoints', 'lmo_maskrcnn_model.pth')

rcnn_cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(rcnnIdx_to_lmoCats_dict)
rcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001 # the predicted category scores
predictor = DefaultPredictor(rcnn_cfg)
################################################# MASK-RCNN Segmentation ##################################################################

cfg.DATASET_NAME = 'lm'                      # dataset name
cfg.RENDER_WIDTH = eval_dataset.cam_width    # the width of rendered images
cfg.RENDER_HEIGHT = eval_dataset.cam_height  # the height of rendered images
cfg.HEMI_ONLY = True

ckpt_file = pjoin(base_path, 
                'checkpoints', 
                "OVE6D_pose_model.pth"
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

test_data_dir = datapath / 'lmo' / 'test'          # path to the test dataset of BOP
eval_dir = pjoin(base_path, 'evaluation/pred_results/LMO')


raw_file_mode = "raw-sampleN{}-viewpointK{}-poseP{}-rcnn_lmo-test.csv"
if cfg.USE_ICP:
    icp1_file_mode = "icp1-sampleN{}-viewpointK{}-poseP{}-nbr{}-itr{}-pts{}-pla{}-rcnn_lmo-test.csv"
    icpk_file_mode = "icpk-sampleN{}-viewpointK{}-poseP{}-nbr{}-itr{}-pts{}-pla{}-rcnn_lmo-test.csv"

obj_renderer = rendering.Renderer(width=cfg.RENDER_WIDTH, height=cfg.RENDER_HEIGHT)

if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

for scene_id in sorted(os.listdir(test_data_dir)):
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
        target_gt_masks = dict()
        view_gt_poses = pose_anno[str(view_id)]
        for ix, gt_obj in enumerate(view_gt_poses):
            gt_obj_id = gt_obj['obj_id']
            mask_file = os.path.join(mask_dir, "{:06d}_{:06d}.png".format(view_id, ix))
            gt_msk = torch.tensor(cv2.imread(mask_file, 0)).type(torch.bool)
            target_gt_masks[gt_obj_id] = gt_msk
            if gt_obj_id not in rcnn_gt_results:
                rcnn_gt_results[gt_obj_id] = 0
            rcnn_gt_results[gt_obj_id] += 1
        ###################### read gt mask ##########################    
        
        ###################### object segmentation ######################
        img_name = "{:06d}.png".format(view_id)
        rgb_file = os.path.join(rgb_dir, img_name)
        rgb_img = cv2.imread(rgb_file)
        output = predictor(rgb_img)
        rcnn_pred_ids = output["instances"].pred_classes # cat_idx: 0 - 7
        rcnn_pred_masks = output["instances"].pred_masks
        # rcnn_pred_bboxes = output["instances"].pred_boxes
        rcnn_pred_scores = output["instances"].scores
        rcnn_cost = time.time() - view_timer
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
        
        unique_rcnn_obj_ids = torch.unique(rcnn_pred_ids)    
        for uniq_rcnn_id in unique_rcnn_obj_ids:
            uniq_lmo_id = rcnnIdx_to_lmoIds_dict[uniq_rcnn_id.item()]
            uniq_obj_codebook = object_codebooks[uniq_lmo_id]

            uniq_obj_mask = obj_masks[rcnn_pred_ids==uniq_rcnn_id]
            uniq_obj_depth = obj_depths[rcnn_pred_ids==uniq_rcnn_id]
            uniq_obj_score = rcnn_pred_scores[rcnn_pred_ids==uniq_rcnn_id]

            mask_pixel_count = uniq_obj_mask.view(uniq_obj_mask.size(0), -1).sum(dim=1)

            valid_idx = (mask_pixel_count >= 100)
            if valid_idx.sum() == 0:
                mask_visib_ratio = mask_pixel_count / mask_pixel_count.max()
                valid_idx = mask_visib_ratio >= 0.05

            uniq_obj_mask = uniq_obj_mask[valid_idx]
            uniq_obj_depth = uniq_obj_depth[valid_idx]
            uniq_obj_score = uniq_obj_score[valid_idx]

            pose_ret = utils.OVE6D_rcnn_full_pose(model_func=model_net, 
                                            obj_depths=uniq_obj_depth,
                                            obj_masks=uniq_obj_mask,
                                            obj_rcnn_scores=uniq_obj_score,
                                            obj_codebook=uniq_obj_codebook, 
                                            cam_K=cam_K,
                                            config=cfg, 
                                            device=DEVICE,
                                            obj_renderer=obj_renderer)
            select_rcnn_idx = pose_ret['rcnn_idx']
            rcnn_pd_mask = uniq_obj_mask[select_rcnn_idx].cpu()
            rcnn_pd_score = uniq_obj_score[select_rcnn_idx].cpu()
            
            if uniq_lmo_id not in rcnn_pd_results:
                rcnn_pd_results[uniq_lmo_id] = list()
            
            if uniq_lmo_id in target_gt_masks:
                obj_gt_mask = target_gt_masks[uniq_lmo_id]
                inter_area = obj_gt_mask & rcnn_pd_mask
                outer_area = obj_gt_mask | rcnn_pd_mask
                iou = inter_area.sum() / outer_area.sum()
                rcnn_pd_results[uniq_lmo_id].append(iou.item())
            else:
                rcnn_pd_results[uniq_lmo_id].append(0.0)

            raw_pred_results.append({'time': pose_ret['raw_time'],
                                    'scene_id': int(scene_id),
                                    'im_id': int(view_id),
                                    'obj_id': int(uniq_lmo_id),
                                    'score': pose_ret['raw_score'].squeeze().numpy(), 
                                    'R': cfg.POSE_TO_BOP(pose_ret['raw_R']).squeeze().numpy(),
                                    't': pose_ret['raw_t'].squeeze().numpy() * 1000.0}) # convert estimated pose to BOP format
            raw_pred_runtime.append(pose_ret['raw_time'])
            if cfg.USE_ICP:
                icp1_pred_results.append({'time': pose_ret['icp1_rawicp_time'],
                                         'scene_id': int(scene_id),
                                         'im_id': int(view_id),
                                         'obj_id': int(uniq_lmo_id),
                                         'score': pose_ret['icp1_score'].squeeze().numpy(), 
                                         'R': cfg.POSE_TO_BOP(pose_ret['icp1_R']).squeeze().numpy(),
                                         't': pose_ret['icp1_t'].squeeze().numpy() * 1000.0})
                icp1_pred_runtime.append(pose_ret['icp1_rawicp_time'])

                icpk_pred_results.append({'time': pose_ret['icpk_rawicp_time'],
                                         'scene_id': int(scene_id),
                                         'im_id': int(view_id),
                                         'obj_id': int(uniq_lmo_id),
                                         'score': pose_ret['icpk_score'].squeeze().numpy(), 
                                         'R': cfg.POSE_TO_BOP(pose_ret['icpk_R']).squeeze().numpy(),
                                         't': pose_ret['icpk_t'].squeeze().numpy() * 1000.0})
                icpk_pred_runtime.append(pose_ret['icpk_rawicp_time'])
        
        view_runtime.append(time.time() - view_timer)
        if (view_id) % 100 == 0:
            print('scene:{}, image: {}, rcnn:{:.3f}, image_cost:{:.3f}, raw_t:{:.3f}, icp1_t:{:.3f}, icpk_t:{:.3f}'.format(
                    int(scene_id), view_id+1, np.mean(rcnn_runtime), np.mean(view_runtime), 
                    np.mean(raw_pred_runtime), np.mean(icp1_pred_runtime), np.mean(icpk_pred_runtime)))

    print('{}, {}'.format(scene_id, time.strftime('%m_%d-%H:%M:%S', time.localtime())))

rawk_eval_file = pjoin(eval_dir, raw_file_mode.format(
                        cfg.RENDER_NUM_VIEWS, cfg.VP_NUM_TOPK, cfg.POSE_NUM_TOPK
                        ))
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


##################### evaluate rcnn detection and segmentation performance ####################
iou_T = 0.5
rcnn_obj_ARs = list()
rcnn_obj_APs = list()
print(' #################################### IOU_Threshold = {:.2f} #################################### '.format(iou_T))
for obj_abs_id, obj_iou in rcnn_pd_results.items():
    obj_name = catId_to_catName_dict[obj_abs_id]
    obj_rcnn_iou = np.array(obj_iou)
    
    all_pd_count = len(obj_rcnn_iou)
    all_gt_count = rcnn_gt_results[obj_abs_id]
    true_pd_count = sum(obj_rcnn_iou >= iou_T)
    
    obj_AP = true_pd_count / all_pd_count # True_PD / ALL_PD
    obj_AR = true_pd_count / all_gt_count # True_PD / ALL_GT
    
    rcnn_obj_APs.append(obj_AP)
    rcnn_obj_ARs.append(obj_AR)
    
    print('obj_id: {:02d}, obj_AR: {:.5f}, obj_AP: {:.5f}, All_GT:{}, All_PD:{}, True_PD:{}, obj_name: {}'.format(
        obj_abs_id, obj_AR, obj_AP, all_gt_count, all_pd_count, true_pd_count, obj_name))

mAR = np.mean(rcnn_obj_ARs)
mAP = np.mean(rcnn_obj_APs)
print('IOU_T:{:.5f}, mean_recall:{:.5f},  mean_precision: {:.5f}'.format(iou_T, mAR, mAP))