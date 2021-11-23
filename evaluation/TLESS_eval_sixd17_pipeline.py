
import os
import sys
# import glob
import json
import yaml
import time
import torch
import warnings
import numpy as np
from PIL import Image
from pathlib import Path
from os.path import join as pjoin

warnings.filterwarnings("ignore")

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

# from bop_toolkit_lib import inout
# from sixd_toolkit.pysixd import inout

from dataset import TLESS_Dataset
from lib import network, rendering
from evaluation import utils
from evaluation import config as cfg

# this function is borrowed from https://github.com/thodan/sixd_toolkit/blob/master/pysixd/inout.py
def save_results_sixd17(path, res, run_time=-1):

    txt = 'run_time: ' + str(run_time) + '\n' # The first line contains run time
    txt += 'ests:\n'
    line_tpl = '- {{score: {:.8f}, ' \
                   'R: [' + ', '.join(['{:.8f}'] * 9) + '], ' \
                   't: [' + ', '.join(['{:.8f}'] * 3) + ']}}\n'
    for e in res['ests']:
        Rt = e['R'].flatten().tolist() + e['t'].flatten().tolist()
        txt += line_tpl.format(e['score'], *Rt)
    with open(path, 'w') as f:
        f.write(txt)

# gpu_id = 0
gpu_id = 1

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ['EGL_DEVICE_ID'] = str(gpu_id)
DEVICE = torch.device('cuda')

datapath = Path(cfg.DATA_PATH)

cfg.DATASET_NAME = 'tless'   # dataset name

eval_dataset = TLESS_Dataset.Dataset(datapath / cfg.DATASET_NAME)
cfg.RENDER_WIDTH = eval_dataset.cam_width        # the width of rendered images
cfg.RENDER_HEIGHT = eval_dataset.cam_height      # the height of rendered imagescd

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

test_data_dir = datapath / 'tless' / 'test_primesense'
rcnn_mask_dir = datapath / 'tless' / 'mask_RCNN_50'


eval_dir = pjoin(base_path, 'evaluation/bop_pred_results/TLESS')

raw_file_mode = "raw-sampleN{}-viewpointK{}-poseP{}-mpmask_tless_primesense"
if cfg.USE_ICP:
    icp1_file_mode = "icp1-sampleN{}-viewpointK{}-poseP{}-nbr{}-itr{}-pts{}-pla{}-css-mpmask_tless_primesense"
    icpk_file_mode = "icpk-sampleN{}-viewpointK{}-poseP{}-nbr{}-itr{}-pts{}-pla{}-css-mpmask_tless_primesense"

obj_renderer = rendering.Renderer(width=cfg.RENDER_WIDTH, height=cfg.RENDER_HEIGHT)

for scene_id in sorted(os.listdir(test_data_dir)):
    raw_eval_dir = pjoin(eval_dir, raw_file_mode.format(
                    cfg.RENDER_NUM_VIEWS, cfg.VP_NUM_TOPK, cfg.POSE_NUM_TOPK))
    scene_raw_eval_dir = pjoin(raw_eval_dir, scene_id)
    if not os.path.exists(scene_raw_eval_dir):
        os.makedirs(scene_raw_eval_dir)

    if cfg.USE_ICP:
        icp1_eval_dir = pjoin(eval_dir, icp1_file_mode.format(
                        cfg.RENDER_NUM_VIEWS, cfg.VP_NUM_TOPK, cfg.POSE_NUM_TOPK,
                        cfg.ICP_neighbors, cfg.ICP_max_iterations, cfg.ICP_correspondences, cfg.ICP_min_planarity,
                        ))
        icpk_eval_dir = pjoin(eval_dir, icpk_file_mode.format(
                        cfg.RENDER_NUM_VIEWS, cfg.VP_NUM_TOPK, cfg.POSE_NUM_TOPK,
                        cfg.ICP_neighbors, cfg.ICP_max_iterations, cfg.ICP_correspondences, cfg.ICP_min_planarity,
                        ))
        scene_icp1_eval_dir = pjoin(icp1_eval_dir, scene_id)
        if not os.path.exists(scene_icp1_eval_dir):
            os.makedirs(scene_icp1_eval_dir)
        scene_icpk_eval_dir = pjoin(icpk_eval_dir, scene_id)
        if not os.path.exists(scene_icpk_eval_dir):
            os.makedirs(scene_icpk_eval_dir)

    scene_dir = pjoin(test_data_dir, scene_id)
    if not os.path.isdir(scene_dir):
        continue

    cam_info_file = pjoin(scene_dir, 'scene_camera.json')
    with open(cam_info_file, 'r') as cam_f:
        scene_camera_info = json.load(cam_f)
    
    scene_mask_dir = pjoin(rcnn_mask_dir, "{:02d}".format(int(scene_id)))
    scene_rcnn_file = pjoin(scene_mask_dir, 'mask_rcnn_predict.yml')
    with open(scene_rcnn_file, 'r') as rcnn_f:
        scene_detect_info = yaml.load(rcnn_f, Loader=yaml.FullLoader)

    depth_dir = pjoin(scene_dir, 'depth')
    view_runtime = list()
    for depth_png in sorted(os.listdir(depth_dir)):
        if not depth_png.endswith('.png'):
            continue
        view_id = int(depth_png.split('.')[0])           # 000000.png
        view_rcnn_ret = scene_detect_info[view_id]       # scene detection results
        view_cam_info = scene_camera_info[str(view_id)]  # scene camera information

        depth_file = pjoin(depth_dir, depth_png)
        mask_file = pjoin(scene_mask_dir, 'masks', '{}.npy'.format(view_id))  # 0000001.npy
        view_masks = torch.tensor(np.load(mask_file), dtype=torch.float32)
        view_depth = torch.from_numpy(np.array(Image.open(depth_file), dtype=np.float32))
        
        view_depth *= view_cam_info['depth_scale']
        view_camK = torch.tensor(view_cam_info['cam_K'], dtype=torch.float32).view(3, 3)[None, ...] # 1x3x3
        view_timer = time.time()
        for obj_rcnn in view_rcnn_ret: # estimate the detected objects
            obj_timer = time.time()
            chan = obj_rcnn['np_channel_id']
            obj_id = obj_rcnn['obj_id']
            obj_conf = obj_rcnn['score']
            if obj_conf < 0:  # only consider the valid detected objects
                continue
            if len(view_masks.shape) == 2:
                obj_mask = view_masks
            else:
                obj_mask = view_masks[:, :, chan] # 1xHxW            
            
            obj_depth = view_depth * obj_mask
            obj_depth = obj_depth * cfg.MODEL_SCALING # from mm to meter
            obj_codebook = object_codebooks[obj_id]
            obj_depth = obj_depth.unsqueeze(0)
            obj_mask = obj_mask.unsqueeze(0)
            pose_ret = utils.OVE6D_mask_full_pose(model_func=model_net, 
                                                obj_depth=obj_depth,
                                                obj_mask=obj_mask,
                                                obj_codebook=obj_codebook, 
                                                cam_K=view_camK,
                                                config=cfg, 
                                                device=DEVICE,
                                                obj_renderer=obj_renderer)

            raw_preds = dict()
            raw_preds.setdefault('ests',[]).append({'score': pose_ret['raw_score'].squeeze().numpy(), 
                                                    'R': cfg.POSE_TO_BOP(pose_ret['raw_R']).numpy().squeeze(),
                                                    't': pose_ret['raw_t'].squeeze().numpy() * 1000.0})
            
            raw_ret_path = os.path.join(scene_raw_eval_dir, '%04d_%02d.yml' % (view_id, obj_id))
            save_results_sixd17(raw_ret_path, raw_preds, run_time=pose_ret['raw_time'])
            raw_pred_runtime.append(pose_ret['raw_time'])

            if cfg.USE_ICP:
                icp1_preds = dict()
                icp1_preds.setdefault('ests',[]).append({'score': pose_ret['icp1_score'].squeeze().numpy(), 
                                                         'R': cfg.POSE_TO_BOP(pose_ret['icp1_R']).numpy().squeeze(),
                                                         't': pose_ret['icp1_t'].squeeze().numpy() * 1000.0})
                
                icp1_ret_path = os.path.join(scene_icp1_eval_dir, '%04d_%02d.yml' % (view_id, obj_id))
                save_results_sixd17(icp1_ret_path, icp1_preds, run_time=pose_ret['icp1_time']) 
                icp1_pred_runtime.append(pose_ret['icp1_time'])                 

                icpk_preds = dict()
                icpk_preds.setdefault('ests',[]).append({'score': pose_ret['icpk_score'].squeeze().numpy(), 
                                                         'R': cfg.POSE_TO_BOP(pose_ret['icpk_R']).numpy().squeeze(),
                                                         't': pose_ret['icpk_t'].squeeze().numpy() * 1000.0})
                
                icpk_ret_path = os.path.join(scene_icpk_eval_dir, '%04d_%02d.yml' % (view_id, obj_id))
                save_results_sixd17(icpk_ret_path, icpk_preds, run_time=pose_ret['icpk_time']) 
                icpk_pred_runtime.append(pose_ret['icpk_time'])  
        
        view_runtime.append(time.time() - view_timer)
        if (view_id+1) % 100 == 0:
            print('scene:{}, image: {}, image_cost:{:.3f}, raw_t:{:.3f}, icp1_t:{:.3f}, icpk_t:{:.3f}'.format(
                    int(scene_id), view_id+1, np.mean(view_runtime), 
                    np.mean(raw_pred_runtime), np.mean(icp1_pred_runtime), np.mean(icpk_pred_runtime)))


    print('{}, {}'.format(scene_id, time.strftime('%m_%d-%H:%M:%S', time.localtime())))

mean_raw_time = np.mean(raw_pred_runtime)
print('raw_mean_runtime: {:.4f}'.format(mean_raw_time))

if cfg.USE_ICP:
    mean_icp1_time = np.mean(icp1_pred_runtime)
    mean_icpk_time = np.mean(icpk_pred_runtime)
    print('icp1_mean_runtime: {:.4f}'.format(mean_icp1_time))
    print('icpk_mean_runtime: {:.4f}'.format(mean_icpk_time))

del obj_renderer


