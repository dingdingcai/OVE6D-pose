import os
from pickle import load
import sys
import time
import math
import matplotlib
import torch
import random
import shutil
import structlog
import torchvision

import copy


from pytorch3d.transforms import matrix_to_euler_angles


import numpy as np
from pathlib import Path
from torch import optim
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from lib import network
from lib.three import batchview

from training import shapenet
from training import train_utils as preprocess
from training import config as cfg
from tools import visualization as viz

# torch.autograd.detect_anomaly()

logger = structlog.get_logger(__name__)

gpu_id = 0
# gpu_id = 1

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ['EGL_DEVICE_ID'] = str(gpu_id)
DEVICE = torch.device('cuda')

ShapeNet_PATH = "/path/to/shapenet"

batchsize = 8
img_wid = cfg.RENDER_WIDTH
img_hei = cfg.RENDER_HEIGHT

dataset = shapenet.ShapeNetV2(data_dir=ShapeNet_PATH,        # dir to shapenet dataset
                              config=cfg,                    # config for renderer
                              x_bound=(-0.02, 0.02),         # x-axis offset from origin
                              y_bound=(-0.02, 0.02),         # y-axis offset from origin
                              scale_jitter=(0.05, 0.25),     # object scaling ranges relative to unit cube(1m x 1m x 1m)
                              dist_jitter=(0.5, 1.5),        # z-axis offset scale range
                              aug_rescale_jitter=(0.2, 0.8), # rescale factor range for augmenting the training images
                              aug_patch_area_ratio=0.2,      # the blocking area ratio for augmenting the training images
                              aug_patch_max_num=1,           # the number of blocks for augmenting the training images
                              aug_guassian_std=0.02,         # the guassian noise std for augmenting the training images
                             )

worker_init_fn = dataset.worker_init_fn if hasattr(dataset, 'worker_init_fn') else None
data_loader = torch.utils.data.DataLoader(dataset,
                                          shuffle=True,
                                          batch_size=batchsize,
                                          num_workers=8,
                                          drop_last=False,
                                          pin_memory=True,
                                          worker_init_fn=worker_init_fn,
                                         )

logger.info('models:{}, batches:{}'.format(len(dataset), len(dataset)//batchsize))

eps = 1e-8
iter_steps = 0
start_lr = cfg.BASE_LR
end_lr = 1e-5
max_epochs = cfg.MAX_EPOCHS
warmup_epochs = cfg.WARMUP_EPOCHS  # paper set to 1
triple_margin = cfg.RANKING_MARGIN # Rz - Rxy >= 0.1

max_angle_margin = math.pi / 2  # 90˚


variant_name = "OVE6D_{}x{}_{}epochs".format(img_hei, img_wid, max_epochs)
ckpt_dir = os.path.join(os.path.abspath('.'), 'checkpoints', variant_name)
tensorboard_dir = os.path.join(ckpt_dir, 'tensorboard')

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if os.path.exists(tensorboard_dir):
    shutil.rmtree(tensorboard_dir)

writer = SummaryWriter(tensorboard_dir)
model_net = network.OVE6D().to(DEVICE)


total_iters = max_epochs * len(dataset) // batchsize
optimizer = optim.Adam(model_net.parameters(), lr=start_lr, weight_decay=1e-5)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=end_lr, last_epoch=-1)

viewpoint_BCE = torch.nn.BCELoss(reduction='mean')
Cosine_Sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
SmoothL1_sum = torch.nn.SmoothL1Loss(reduction='sum', beta=0.001)   # beta=0, smoothL1 = L1
SmoothL1_mean = torch.nn.SmoothL1Loss(reduction='mean', beta=0.001) # beta=0, smoothL1 = L1

Triple_CosineDist = torch.nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
    margin=triple_margin, # Rxy + 0.5 < Rz
    swap=True,
    reduction='mean')

loss_turn_on = False

iterval_loss = list()
iterval_dz_loss = list()

iterval_rot_loss = list()
iterval_tsl_loss = list()

iterval_rank_loss = list()
iterval_stn_L1_loss = list()
iterval_stn_sim_loss = list()
iterval_wgt_rot_loss = list()
iterval_vp_conf_loss = list()

iterval_img_L1_loss = list()
iterval_img_sim_loss = list()

model_net.train()

for epoch in range(max_epochs):
    for train_data in data_loader:
        optimizer.zero_grad()
        render_timer = time.time()
        trainer_timer = time.time()

        valid_anc_idx = batchview.bv2b(train_data['anchor']['valid_idx'])
        valid_inp_idx = batchview.bv2b(train_data['inplane']['valid_idx'])
        valid_out_idx = batchview.bv2b(train_data['outplane']['valid_idx'])
        valid_depth_idx = (valid_anc_idx * valid_inp_idx * valid_out_idx).bool()
        
        anc_masks_GT = batchview.bv2b(train_data['anchor']['mask'])
        inp_masks_GT = batchview.bv2b(train_data['inplane']['mask'])
        out_masks_GT = batchview.bv2b(train_data['outplane']['mask'])
        
        inp_depths_aug = batchview.bv2b(train_data['inplane']['aug_depth'])
        out_depths_aug = batchview.bv2b(train_data['outplane']['aug_depth'])
        inp_masks_aug = torch.zeros_like(inp_depths_aug)
        inp_masks_aug[inp_depths_aug>0] = 1
        out_masks_aug = torch.zeros_like(out_depths_aug)
        out_masks_aug[out_depths_aug>0] = 1
        
        anc_mask_pixels = anc_masks_GT.view(anc_masks_GT.shape[0], -1).sum(1)
        inp_gt_mask_pixels = inp_masks_GT.view(inp_masks_GT.shape[0], -1).sum(1)
        inp_aug_mask_pixels = inp_masks_aug.view(inp_masks_aug.shape[0], -1).sum(1)
        inp_aug_visib_frac = inp_aug_mask_pixels / inp_gt_mask_pixels

        out_gt_mask_pixels = out_masks_GT.view(out_masks_GT.shape[0], -1).sum(1)
        out_aug_mask_pixels = out_masks_aug.view(out_masks_aug.shape[0], -1).sum(1)
        oup_aug_visib_frac = out_aug_mask_pixels / out_gt_mask_pixels

                
        valid_mask_idx = ((anc_mask_pixels > cfg.MIN_DEPTH_PIXELS).bool() & 
                          (inp_aug_mask_pixels > cfg.MIN_DEPTH_PIXELS).bool() &
                          (out_aug_mask_pixels > cfg.MIN_DEPTH_PIXELS).bool() & 
                          (inp_aug_visib_frac > cfg.VISIB_FRAC).bool() & 
                          (oup_aug_visib_frac > cfg.VISIB_FRAC).bool())


        valid_idx = valid_mask_idx & valid_depth_idx
        num_samples = valid_idx.sum()
        if num_samples < 4:  # skip the batch with few samples
            continue
        
        # del (inp_masks_GT, out_masks_GT, inp_masks_aug, out_masks_aug, inp_depths_aug, out_depths_aug)

        anc_depths_GT = batchview.bv2b(train_data['anchor']['depth'])[valid_idx]
        inp_depths_GT = batchview.bv2b(train_data['inplane']['depth'])[valid_idx]
        out_depths_GT = batchview.bv2b(train_data['outplane']['depth'])[valid_idx]
        
        inp_depths_aug = batchview.bv2b(train_data['inplane']['aug_depth'])[valid_idx]
        out_depths_aug = batchview.bv2b(train_data['outplane']['aug_depth'])[valid_idx]
        
        anc_extrinsic_GT = batchview.bv2b(train_data['anchor']['extrinsic'])[valid_idx]
        inp_extrinsic_GT = batchview.bv2b(train_data['inplane']['extrinsic'])[valid_idx]
        out_extrinsic_GT = batchview.bv2b(train_data['outplane']['extrinsic'])[valid_idx]
        
        inp_Rxyz_GT = batchview.bv2b(train_data['inplane']['rotation_to_anchor'])[valid_idx]
        obj_diameters = batchview.bv2b(train_data['anchor']['obj_diameter'])[valid_idx]

        target_zoom_dists = cfg.ZOOM_DIST_FACTOR * obj_diameters * cfg.INTRINSIC.squeeze()[0, 0]
        try:
            zoom_anchor_GT, _, _ = preprocess.input_zoom_preprocess(
                                                images=anc_depths_GT.to(DEVICE),
                                                target_dist=target_zoom_dists,
                                                target_size=cfg.ZOOM_SIZE,
                                                scale_mode=cfg.ZOOM_MODE,
                                                intrinsic=cfg.INTRINSIC,
                                                extrinsic=anc_extrinsic_GT,
                                                normalize=True)
            zoom_inplane_T = inp_extrinsic_GT[:, :3, 3]
            zoom_outplane_GT, _, _ = preprocess.input_zoom_preprocess(
                                                images=out_depths_GT.to(DEVICE),
                                                target_dist=target_zoom_dists,
                                                target_size=cfg.ZOOM_SIZE,
                                                scale_mode=cfg.ZOOM_MODE,
                                                intrinsic=cfg.INTRINSIC,
                                                extrinsic=out_extrinsic_GT,
                                                normalize=True)
        
            zoom_inplane_aug, _, zoom_inplane_T_aug = preprocess.input_zoom_preprocess(
                                                    images=inp_depths_aug.to(DEVICE),
                                                    target_dist=target_zoom_dists,
                                                    target_size=cfg.ZOOM_SIZE,
                                                    scale_mode=cfg.ZOOM_MODE,
                                                    intrinsic=cfg.INTRINSIC,
                                                    extrinsic=None,
                                                    normalize=True)

            inp_extrinsic_aug = inp_extrinsic_GT.clone()
            inp_extrinsic_aug[:, :2, 3] = zoom_inplane_T_aug[:, :2] # only replace tx, ty

            zoom_outplane_aug, _, zoom_outplane_T_aug = preprocess.input_zoom_preprocess(
                                                    images=out_depths_aug.to(DEVICE),
                                                    target_dist=target_zoom_dists,
                                                    target_size=cfg.ZOOM_SIZE,
                                                    scale_mode=cfg.ZOOM_MODE,
                                                    intrinsic=cfg.INTRINSIC,
                                                    extrinsic=None,
                                                    normalize=True)
                                                    
            inp_gt_transform = preprocess.residual_inplane_transform3(gt_t=zoom_inplane_T,
                                                                    init_t=zoom_inplane_T_aug, 
                                                                    gt_Rz=inp_Rxyz_GT, 
                                                                    config=cfg,
                                                                    target_dist=target_zoom_dists,
                                                                    device=DEVICE)
            del (anc_depths_GT, inp_depths_GT, out_depths_GT, inp_depths_aug, out_depths_aug)
        except:
            logger.warning('skip zooming preprocess ...')
            continue
        
        gt_full_T = zoom_inplane_T.to(DEVICE)               # ground truth translation
        gt_theta = inp_gt_transform[:, :2, :3].to(DEVICE)   # Bx2x3, ground truth in-plane 2D transformation (rotation + translation)
        gt_dz_offset = inp_gt_transform[:, 2:3, 2].to(DEVICE) # Bx1, ground truth depth shift (gt_dz = gt_tz - pd_tz)

        pd_init_T = zoom_inplane_T_aug.to(DEVICE)
        
        (pd_theta,
        gt_inp_cls, pd_inp_cls, pd_oup_cls, pd_mix_cls,
        z_anc_gt_vec, z_inp_aug_vec, z_out_aug_vec) = model_net(x_anc_gt=zoom_anchor_GT.to(DEVICE), 
                                                                x_oup_gt=zoom_outplane_GT.to(DEVICE),
                                                                x_inp_aug=zoom_inplane_aug.to(DEVICE),
                                                                x_oup_aug=zoom_outplane_aug.to(DEVICE),
                                                                inp_gt_theta=gt_theta.to(DEVICE))
        
        pd_stn_depths = preprocess.spatial_transform_2D(x=zoom_anchor_GT.to(DEVICE),
                                                        theta=pd_theta.to(DEVICE), 
                                                        mode=cfg.ZOOM_MODE, 
                                                        padding_mode="zeros", 
                                                        align_corners=False)
        gt_stn_depths = preprocess.spatial_transform_2D(x=zoom_anchor_GT.to(DEVICE), 
                                                        theta=gt_theta.to(DEVICE), 
                                                        mode=cfg.ZOOM_MODE, 
                                                        padding_mode="zeros", 
                                                        align_corners=False)

        gt_dxdy_offset = gt_theta[:, :2, 2] # Bx2
        pd_dxdy_offset = pd_theta[:, :2, 2] # Bx2 

        tsl_loss = SmoothL1_mean(pd_dxdy_offset, gt_dxdy_offset)

        pd_T_offset = torch.cat([pd_dxdy_offset, gt_dz_offset.to(pd_dxdy_offset.device)], dim=1) # Bx3
        pd_init_T = pd_init_T.type(pd_T_offset.dtype) # Bx3
                
        pd_full_T = preprocess.recover_residual_translation3(init_t=pd_init_T, 
                                                             offset_t=pd_T_offset,
                                                             config=cfg,
                                                             target_dist=target_zoom_dists,
                                                             device=DEVICE)    

        img_L1_loss = SmoothL1_sum(pd_stn_depths, gt_stn_depths) / num_samples
        # img_L1_loss = SmoothL1_mean(pd_stn_depths, gt_stn_depths)
        view_cosim = Cosine_Sim(pd_stn_depths.view(num_samples, -1), gt_stn_depths.view(num_samples, -1)) # cosine similarity [-1, 1.0]
        view_cosim = torch.clamp((view_cosim + 1.0) / 2.0, min=0.0+eps, max=1.0-eps) # from [-1, 1.0] to (0, 1)
        view_dissim = -torch.log(view_cosim) # negative logarithm 
        img_sim_loss = view_dissim.mean()    # minimize img_sim_loss -> maximize view cosine similarity

        gt_Rz_rot = gt_theta[:, :2, :2] # Bx2x2, object_space to camera_space
        pd_Rz_rot = pd_theta[:, :2, :2] # BV x 2 x 2
        rot_cosine = torch.einsum('bii->b', gt_Rz_rot.to(DEVICE) @ pd_Rz_rot.transpose(-1, -2)) / 2.0
        rot_loss = torch.arccos(torch.clamp(rot_cosine, min=-1.0+eps, max=1.0-eps))
        rot_loss = rot_loss.mean()
        
        inp_vp = anc_extrinsic_GT[:, 2, :3].to(DEVICE)     # Nx3       
        out_vp = out_extrinsic_GT[:, 2, :3].to(DEVICE)     # Nx3        
        
        rank_loss = Triple_CosineDist(z_anc_gt_vec, z_inp_aug_vec, z_out_aug_vec)

        vp_conf_loss = torch.maximum(pd_oup_cls - gt_inp_cls + triple_margin, torch.zeros_like(pd_oup_cls)).mean()

        loss = 100 * rank_loss + 10 * vp_conf_loss + img_sim_loss
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()  

        iter_steps += 1
        iterval_loss.append(loss.mean().item())
        iterval_rot_loss.append(rot_loss.mean().item())
        iterval_tsl_loss.append(tsl_loss.mean().item())

        iterval_rank_loss.append(rank_loss.mean().item())
        iterval_vp_conf_loss.append(vp_conf_loss.mean().item())

        iterval_img_L1_loss.append(img_L1_loss.mean().item())
        iterval_img_sim_loss.append(img_sim_loss.mean().item())
        
        if iter_steps <= 10 or iter_steps % 100 == 0:
            if iter_steps >= 1000:
                iter_str = '{:.1f}K'.format(iter_steps / 1000)
            else:
                iter_str = str(iter_steps)
            mean_loss = np.mean(iterval_loss)
            mean_rot_loss = np.mean(iterval_rot_loss)
            mean_tsl_loss = np.mean(iterval_tsl_loss)

            mean_rank_loss = np.mean(iterval_rank_loss)

            mean_vp_conf_loss = np.mean(iterval_vp_conf_loss)

            mean_img_L1_loss = np.mean(iterval_img_L1_loss)
            mean_img_sim_loss = np.mean(iterval_img_sim_loss)

            outplane_sim = Cosine_Sim(z_anc_gt_vec, z_out_aug_vec)
            inplane_sim = Cosine_Sim(z_anc_gt_vec, z_inp_aug_vec)
            rank_margin = inplane_sim - outplane_sim
            
            topK = np.minimum(10, inplane_sim.shape[0])
            inplane_loss = -(-inplane_sim).topk(k=topK)[0].mean()
            outplane_loss = outplane_sim.topk(k=topK)[0].mean()
            top_rank_margin = (rank_margin).topk(k=topK)[0].mean()

            rot_err_in_degree = mean_rot_loss / math.pi * 180
            current_lr = optimizer.param_groups[0]['lr']

 
            model_net.eval()
            eval_anc_maps = model_net.vipri_encoder(zoom_anchor_GT[:4], return_maps=True)[0]
            eval_inp_maps = model_net.vipri_encoder(zoom_inplane_aug[:4], return_maps=True)[0]
            eval_theta = model_net.regression_head(x=eval_anc_maps, y=eval_inp_maps).detach().cpu()
            model_net.train()

            eval_pd_grid = torchvision.utils.make_grid(viz.colorize_tanh_depth(
                                                        preprocess.spatial_transform_2D(x=zoom_anchor_GT[:4], 
                                                        theta=eval_theta, 
                                                        mode='nearest')), nrow=2)

            anc_gt_grid = torchvision.utils.make_grid(viz.colorize_tanh_depth(zoom_anchor_GT[:4]), nrow=2)
            # inp_gt_grid = torchvision.utils.make_grid(viz.colorize_tanh_depth(zoom_inplane_GT_augE[:4]), nrow=2)
            oup_gt_grid = torchvision.utils.make_grid(viz.colorize_tanh_depth(zoom_outplane_GT[:4]), nrow=2)

            inp_aug_grid = torchvision.utils.make_grid(viz.colorize_tanh_depth(zoom_inplane_aug[:4]), nrow=2)
            oup_aug_grid = torchvision.utils.make_grid(viz.colorize_tanh_depth(zoom_outplane_aug[:4]), nrow=2)

            stn_pd_grid = torchvision.utils.make_grid(viz.colorize_tanh_depth(pd_stn_depths[:4]), nrow=2)
            stn_gt_grid = torchvision.utils.make_grid(viz.colorize_tanh_depth(gt_stn_depths[:4]), nrow=2)

            
            gt_Rz_mat = F.pad(gt_theta[:4, :2, :2], (0, 1, 0, 1))
            gt_Rz_mat[:, -1, -1] = 0.0
            gt_Rz_angles = matrix_to_euler_angles(gt_Rz_mat, "XYZ") / math.pi * 180

            pd_Rz_mat = F.pad(pd_theta[:4, :2, :2], (0, 1, 0, 1))
            pd_Rz_mat[:, -1, -1] = 0.0
            pd_Rz_angles = matrix_to_euler_angles(pd_Rz_mat, "XYZ") / math.pi * 180

            pd_eval_mat = F.pad(eval_theta[:4, :2, :2], (0, 1, 0, 1))
            pd_eval_mat[:, -1, -1] = 0.0
            pd_eval_angles = matrix_to_euler_angles(pd_eval_mat, "XYZ") / math.pi * 180

            writer.add_image("Images/A_source_view", anc_gt_grid, iter_steps)
            writer.add_image("Images/C_pd_stn_view", stn_pd_grid, iter_steps)
            writer.add_image("Images/D_gt_stn_view", stn_gt_grid, iter_steps)
            writer.add_image("Images/E_pd_eval_view", eval_pd_grid, iter_steps)

            writer.add_image("Training/A_anchor_gt", anc_gt_grid, iter_steps)
            writer.add_image("Training/C_outplane_gt", oup_gt_grid, iter_steps)
            writer.add_image("Training/D_inplane_aug", inp_aug_grid, iter_steps)
            writer.add_image("Training/E_outplane_aug", oup_aug_grid, iter_steps)
        
            for i in range(4):
                viz_cosim_inp = inplane_sim[i].cpu().item()
                viz_cosim_oup = outplane_sim[i].cpu().item()
                viz_cosim_diff = viz_cosim_inp - viz_cosim_oup

                pd_pos_vp_conf = pd_inp_cls[i].cpu().item()
                pd_neg_vp_conf = pd_oup_cls[i].cpu().item()
                pd_mix_vp_conf = pd_mix_cls[i].cpu().item()


                init_t = pd_init_T[i].cpu() * 1000
                init_tsl = "[{:+06.1f}, {:+06.1f}, {:+06.1f}]".format(init_t[0].item(), 
                                                                        init_t[1].item(), 
                                                                        init_t[2].item())

                pd_t = pd_full_T[i].cpu() * 1000
                pd_tsl = "[{:+06.1f}, {:+06.1f}, {:+06.1f}]".format(pd_t[0].item(), 
                                                                    pd_t[1].item(), 
                                                                    pd_t[2].item())                                                
                gt_t = gt_full_T[i].cpu() * 1000
                gt_tsl = "[{:+06.1f}, {:+06.1f}, {:+06.1f}]".format(gt_t[0].item(), 
                                                                    gt_t[1].item(), 
                                                                    gt_t[2].item())
                pd_err_t = gt_t - pd_t
                pd_err_tsl = "[{:+06.1f}, {:+06.1f}, {:+06.1f}]".format(pd_err_t[0].item(), 
                                                                        pd_err_t[1].item(), 
                                                                        pd_err_t[2].item())  

                init_err_t = gt_t - init_t
                init_err_tsl = "[{:+06.1f}, {:+06.1f}, {:+06.1f}]".format(init_err_t[0].item(), 
                                                                            init_err_t[1].item(), 
                                                                            init_err_t[2].item())  

                gt_Rz_ang = gt_Rz_angles[i].cpu().squeeze()      
                pd_Rz_ang = pd_Rz_angles[i].cpu().squeeze()      
                pd_eval_ang = pd_eval_angles[i].cpu().squeeze()      

                writer.add_hparams(hparam_dict={"name":int(iter_steps*10+i), 
                                                'pd_err(mm)':pd_err_tsl,
                                                'init_err(mm)':init_err_tsl,
                                                }, 
                                    metric_dict={
                                                'hparam/pos_vp': pd_pos_vp_conf, 
                                                'hparam/neg_vp': pd_neg_vp_conf, 
                                                'hparam/mix_vp': pd_mix_vp_conf, 

                                                })
            if iter_steps > 200: # skip the statistics at the beginning
                writer.add_scalar("Loss/total_loss", mean_loss, iter_steps)
                writer.add_scalar("Loss/tsl_loss", mean_tsl_loss, iter_steps)
                writer.add_scalar("Loss/rank_loss", mean_rank_loss, iter_steps)
                writer.add_scalar("Loss/mean_vp_conf_loss", np.minimum(mean_vp_conf_loss, 0.5), iter_steps)
                
                writer.add_scalar("WarmUp/img_L1_loss", mean_img_L1_loss, iter_steps)
                writer.add_scalar("WarmUp/img_sim_loss", mean_img_sim_loss, iter_steps) 
                
                writer.add_scalar("Other/LR", current_lr, iter_steps)
                writer.add_scalar("Other/rot_loss", mean_rot_loss, iter_steps)
                writer.add_scalar("Other/rot_deg", rot_err_in_degree, iter_steps)

            del (anc_gt_grid, oup_gt_grid, 
                inp_aug_grid, oup_aug_grid, eval_pd_grid,
                stn_gt_grid, stn_pd_grid)  
                                        

            printstr = ('[{}/{:.1f}K], ls:{:.3f}, '
                        + 'sim:{:.4f}, tsl:{:.3f}, img_L1:{:.1f}, img_sim:{:.3f}, vp:{:.4f}, conf:[{:.3f}, {:.3f}, {:.3f}], deg:{:.1f}, '
                        + '{}, lr(*k):{:.3f}').format(iter_str, total_iters/1000, 
                            mean_loss, 
                            mean_rank_loss, 
                            mean_tsl_loss,
                            mean_img_L1_loss,
                            mean_img_sim_loss,
                            mean_vp_conf_loss,
                            gt_inp_cls.cpu().squeeze()[0].item(),
                            pd_inp_cls.cpu().squeeze()[0].item(),
                            pd_oup_cls.cpu().squeeze()[0].item(),
                            rot_err_in_degree,
                            len(z_anc_gt_vec), 
                            # inplane_loss, outplane_loss, 
                            current_lr*1000  # for better printing
                            )

            logger.info(printstr)
            iterval_loss = list()
            iterval_dz_loss = list()
            iterval_rot_loss = list()
            iterval_tsl_loss = list()
            iterval_rank_loss = list()
            iterval_stn_L1_loss = list()
            iterval_stn_sim_loss = list()
            iterval_wgt_rot_loss = list()
            iterval_img_L1_loss = list()
            iterval_img_sim_loss = list()
            iterval_vp_conf_loss = list()
        
    time_stamp = time.strftime('%m_%d-%H:%M:%S', time.localtime())
    ckpt_file = os.path.join(ckpt_dir, 'pose_model_{}_{}_{}_{:.4f}_{:.4f}_{:.1f}.pth'.format(
                                        epoch+1, iter_steps, time_stamp, mean_rank_loss, mean_tsl_loss, rot_err_in_degree))
    torch.save(model_net.state_dict(), ckpt_file)
    logger.info('{} epoch, saving model to {}'.format(epoch+1, ckpt_file))

