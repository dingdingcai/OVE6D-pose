import os
import time
import glob
import math
import torch
import numpy as np
import torch.nn.functional as F
from lib import geometry, rendering, three, preprocess
from evaluation import pplane_ICP
from lib import preprocess


def rotation_to_position(R):
    t = torch.tensor([0, 0, 1], dtype=torch.float32, device=R.device)[None, ..., None]
    pos = (-R.squeeze().transpose(-2, -1) @ t).squeeze()
    return pos


def rotation_error(R0, R1):
    cos = (torch.trace(R0.squeeze().clone() @ R1.squeeze().T) - 1.0) / 2.0
    if cos < -1:
        cos = -2 - cos
    elif cos > 1:
        cos = 2 - cos
    return torch.arccos(cos)/math.pi*180


def background_filter(depths, diameters, dist_factor=0.5):
    """
    filter out the outilers beyond the object diameter
    """
    new_depths = list()
    unsqueeze = False
    if not isinstance(diameters, torch.Tensor):
        diameters = torch.tensor(diameters)
    
    if depths.dim() == 2:
        depths = depths[None, ...]
    if depths.dim() > 3:
        depths = depths.view(-1, depths.shape[-2], depths.shape[-1])
        # diameters = diameters.view(-1)
        unsqueeze = True
    
    if diameters.dim() == 0:
        diameters = diameters[None, ...].repeat(len(depths), 1)
        
    diameters = diameters.to(depths.device)
    assert len(depths) == len(diameters)
    for ix, dep in enumerate(depths):
        hei, wid = dep.shape
        diameter = diameters[ix]
        if (dep>0).sum() < 10:
            new_depths.append(dep)
            continue
            
        dep_vec = dep.view(-1)
        dep_val = dep_vec[dep_vec>0].clone()
        med_val = dep_val.median()
        
        dep_dist = (dep_val - med_val).abs()
        dist, indx = torch.topk(dep_dist, k=len(dep_dist))
        invalid_idx = indx[dist > dist_factor * diameter]
        dep_val[invalid_idx] = 0
        dep_vec[dep_vec>0] = dep_val
        new_dep = dep_vec.view(hei, wid)
        if (new_dep>0).sum() < 100: # the number of valid depth values is too small, then return old one
            new_depths.append(dep)
        else:
            new_depths.append(new_dep)
    
    new_depths = torch.stack(new_depths, dim=0).to(depths.device)
    if unsqueeze:
        new_depths = new_depths.unsqueeze(1) 
    return new_depths


def input_zoom_preprocess(input_depth, input_mask, intrinsic, device, target_zoom_dist, zoom_scale_mode, zoom_size, extrinsic=None):
    input_depth, input_mask = input_depth.squeeze(), input_mask.squeeze()
    if input_depth.dim() == 2:
        input_depth = input_depth[None, None, ...]
    elif input_depth.dim() == 3:
        input_depth = input_depth.unsqueeze(1)
    if input_mask.dim() == 2:
        input_mask = input_mask[None, None, ...]
    elif input_mask.dim() == 3:
        input_mask = input_mask.unsqueeze(1)
    input_depth, input_mask = input_depth.to(device), input_mask.to(device)
    intrinsic = intrinsic.to(device)
    input_centroids = geometry.masks_to_centroids(input_mask)
    input_translations = torch.stack(geometry.estimate_translation(depth=input_depth, 
                                                                   mask=input_mask, 
                                                                   intrinsic=intrinsic), dim=1)
    input_zs = input_translations[:, 2]
    input_mean_depths = input_mask * input_zs[..., None, None, None]
    input_mean_depths[input_depth<=0] = 0    
    input_norm_depths = input_depth - input_mean_depths

    im_hei, im_wid = input_depth.shape[-2:]

    if extrinsic is None:
        input_cameras = geometry.Camera(intrinsic=intrinsic, height=im_hei, width=im_wid)

        input_zoom_depths, _ = input_cameras.zoom(image=input_norm_depths,
                                                target_dist=target_zoom_dist, 
                                                target_size=zoom_size, 
                                                zs=input_zs, 
                                                centroid_uvs=input_centroids,
                                                scale_mode=zoom_scale_mode)
    else:
        input_cameras = geometry.Camera(intrinsic=intrinsic, extrinsic=extrinsic, width=im_wid, height=im_hei)
        input_zoom_depths, _ = input_cameras.zoom(image=input_norm_depths,
                                                 target_dist=target_zoom_dist, 
                                                 target_size=zoom_size, 
                                                 scale_mode=zoom_scale_mode)
    return input_zoom_depths, input_translations


def viewpoint_sampling_and_encoding(model_func, obj_model_file, obj_diameter, config, intrinsic, device):
    render_width = config.RENDER_WIDTH
    render_height = config.RENDER_HEIGHT
    render_num_views = config.RENDER_NUM_VIEWS
    render_obj_scale = config.MODEL_SCALING
    store_featmap = config.SAVE_FTMAP
    zoom_size = config.ZOOM_SIZE
    zoom_scale_mode = config.ZOOM_MODE
    target_zoom_dist = config.ZOOM_DIST_FACTOR * obj_diameter * intrinsic.squeeze().cpu()[0, 0]
     
    obj_Rs = rendering.evenly_distributed_rotation(n=render_num_views, random_seed=config.RANDOM_SEED)

    if config.HEMI_ONLY:
        upper_hemi = obj_Rs[:, 2, 2] > 0 # the upper hemisphere only
        hemi_obj_Rs = obj_Rs[upper_hemi]
        obj_Rs = hemi_obj_Rs

    obj_Ts = torch.zeros((len(obj_Rs), 3), dtype=torch.float32)
    obj_Ts[:, 2] = config.RENDER_DIST * obj_diameter
    
    obj_codebook_Z_vec = list()
    obj_codebook_Z_map = list()
    obj_codebook_delta_Ts = list()
    Rs_chunks = torch.split(obj_Rs, split_size_or_sections=200, dim=0)
    Ts_chunks = torch.split(obj_Ts, split_size_or_sections=200, dim=0)

    render_costs = list()
    infer_costs = list()
    intrinsic = intrinsic.to(device)

    obj_mesh, _ = rendering.load_object(obj_model_file, resize=False, recenter=False)
    obj_mesh.rescale(scale=render_obj_scale) # from millimeter normalize to meter

    for cbk_Rs, cbk_Ts in zip(Rs_chunks, Ts_chunks):
        render_timer = time.time()
        cbk_depths, cbk_masks = rendering.rendering_views(obj_mesh=obj_mesh,
                                                        intrinsic=intrinsic.cpu(),
                                                        R=cbk_Rs,
                                                        T=cbk_Ts,   
                                                        width=render_width,
                                                        height=render_height)
        render_cost = time.time() - render_timer
        render_costs.append(render_cost)
        
        encoder_timer = time.time()
        
        cbk_depths, cbk_masks = cbk_depths.to(device), cbk_masks.to(device)

        extrinsic = torch.ones((cbk_Rs.size(0), 4, 4), device=cbk_Rs.device)
        extrinsic[:, :3, :3] = cbk_Rs
        extrinsic[:, :3, 3] = cbk_Ts

        cbk_zoom_depths, cbk_init_ts = input_zoom_preprocess(input_depth=cbk_depths, 
                                                            input_mask=cbk_masks, 
                                                            intrinsic=intrinsic, 
                                                            target_zoom_dist=target_zoom_dist, 
                                                            zoom_scale_mode=zoom_scale_mode, 
                                                            zoom_size=zoom_size, 
                                                            device=device)
                
        cbk_delta_Ts = cbk_Ts - cbk_init_ts.cpu().squeeze()
        with torch.no_grad():
            z_map, z_vec = model_func.vipri_encoder(cbk_zoom_depths, return_maps=True)
        
        obj_codebook_Z_vec.append(z_vec.detach().cpu())

        obj_codebook_delta_Ts.append(cbk_delta_Ts)
        if store_featmap:
            obj_codebook_Z_map.append(z_map.detach().cpu())

    del cbk_depths, cbk_masks, z_map, z_vec
    encoder_cost = time.time() - encoder_timer
    infer_costs.append(encoder_cost)

    obj_codebook_delta_Ts = torch.cat(obj_codebook_delta_Ts, dim=0)
    obj_codebook_Z_vec = torch.cat(obj_codebook_Z_vec, dim=0)
    if store_featmap:
        obj_codebook_Z_map = torch.cat(obj_codebook_Z_map, dim=0)
    else:
        obj_codebook_Z_map = None
    print('render_time:{:.3f}, encoding_time:{:.3f}'.format(np.sum(render_costs), np.sum(infer_costs)))
    return {"Rs":obj_Rs, 
            "Ts":obj_codebook_delta_Ts, 
            "diameter":obj_diameter,
            "Z_vec":obj_codebook_Z_vec,
            "Z_map":obj_codebook_Z_map,
            "obj_mesh":obj_mesh,
           }


def OVE6D_codebook_generation(model_func, codebook_dir, dataset, config, device):
    object_codebooks = dict()

    codebook_files = sorted(glob.glob(os.path.join(codebook_dir, 
                            '*_views_{}.npy'.format(config.RENDER_NUM_VIEWS))))
    intrinsic = dataset.cam_K
    obj_model_files = dataset.obj_model_file
    obj_diameter_info = dataset.obj_diameter

    num_objects = len(obj_model_files)

    if len(codebook_files) == num_objects:  # pre-built codebooks exist, load it
        for obj_cbk_file in codebook_files:
            cbk_name = obj_cbk_file.split('/')[-1]
            obj_id = int(cbk_name.split('_')[-3])    # _obj_{:02d}_views_{}.npy
            print('Loading ', obj_cbk_file)
            with open(obj_cbk_file, 'rb') as f:
                object_codebooks[obj_id] = np.load(f, allow_pickle=True).item()
    else:
        print('generating codebook for {} viewpoints ...'.format(config.RENDER_NUM_VIEWS))
        for obj_id, obj_model_file in obj_model_files.items():
            obj_diameter = obj_diameter_info[obj_id] * config.MODEL_SCALING # from milimeter to meter
            obj_cbk = viewpoint_sampling_and_encoding(model_func=model_func,
                                                            obj_model_file=obj_model_file,
                                                            obj_diameter=obj_diameter,
                                                            config=config,
                                                            intrinsic=intrinsic,
                                                            device=device)                                              
            if not os.path.exists(codebook_dir):
                os.makedirs(codebook_dir)
            codebook_file = os.path.join(codebook_dir, 
                            '{}_obj_{:02d}_views_{}.npy'.format(
                            config.DATASET_NAME, obj_id, config.RENDER_NUM_VIEWS))
                           
            with open(codebook_file, 'wb') as f:
                np.save(f, obj_cbk)
            object_codebooks[obj_id] = obj_cbk
            
            print('obj_id: ', obj_id, time.strftime('%m_%d-%H:%M:%S', time.localtime()))
    return object_codebooks


def OVE6D_translation_estimation(est_R, est_t, intrinsic, obj_scene, obj_render):
    device = est_R.device
    est_t = est_t.to(device)
    obj_scene.set_pose(rotation=est_R, translation=est_t)
    est_depth, est_mask = obj_render.render(obj_scene)[1:]
    est_syn_t = torch.stack(geometry.estimate_translation(depth=est_depth[None, ...].to(device), 
                                                         mask=est_mask[None, ...].to(device), 
                                                         intrinsic=intrinsic.squeeze().to(device)), 
                            dim=-1).squeeze().numpy()
    refined_est_t = 2 * est_t - est_syn_t
    return refined_est_t


def OVE6D_mask_full_pose(model_func, obj_depth, obj_mask, obj_codebook, cam_K, config, obj_renderer, device):
    """
    Perform OVE6D with given single mask 
    """
    pose_ret = dict()
    cam_K = cam_K.to(device)
    obj_mask = obj_mask.to(device)
    obj_depth = obj_depth.to(device)

    obj_mesh = obj_codebook['obj_mesh']
    obj_diameter = obj_codebook['diameter']
    
    bg_timer = time.time()
    obj_depth = background_filter(obj_depth, obj_diameter) # filter out outliers
    bg_cost = time.time() - bg_timer
    prep_timer = time.time()
    obj_mask[obj_depth<0] = 0
    obj_zoom_dist = config.ZOOM_DIST_FACTOR * obj_diameter * cam_K.squeeze().cpu()[0, 0]
    zoom_test_depth, init_t = input_zoom_preprocess(input_depth=obj_depth, 
                                                input_mask=obj_mask, 
                                                intrinsic=cam_K, 
                                                device=device, 
                                                target_zoom_dist=obj_zoom_dist, 
                                                zoom_scale_mode=config.ZOOM_MODE, 
                                                zoom_size=config.ZOOM_SIZE) # 1xHxW
    prep_cost = time.time() - prep_timer # the pre-processing runtime
    rot_timer = time.time()

    # if obj_codebook.get('gt_R', None) is not None:
    #     estimated_R = obj_codebook['gt_R'].expand(config.RANK_NUM_TOPK, -1, -1)
    #     estimated_scores = torch.zeros((config.RANK_NUM_TOPK,))
    # else:
    estimated_R, estimated_scores = OVE6D_mask_rotation_estimation(input_depth=zoom_test_depth,
                                                                model_net=model_func,
                                                                object_codebook=obj_codebook, 
                                                                cfg=config)    
    rot_cost = time.time() - rot_timer # rotation estimation runtime

    raw_est_Rs = []
    raw_est_ts = []
    raw_masks = []
    raw_depths = []

    icp_est_Rs = []
    icp_est_ts = []
    icp_masks = []
    icp_depths = []

    icp_timer = time.time()
    dst_pts = pplane_ICP.depth_to_pointcloud(obj_depth.squeeze(), cam_K)
    icp_cost = time.time() - icp_timer
    postp_cost = 0
    tsl_cost = 0
    obj_context = rendering.SceneContext(obj=obj_mesh, intrinsic=cam_K.cpu()) # define a scene
    for idx, (obj_est_R, obj_est_score) in enumerate(zip(estimated_R, estimated_scores)):
        tsl_timer = time.time()
        obj_est_t = OVE6D_translation_estimation(est_R=obj_est_R, 
                                                est_t=init_t, 
                                                intrinsic=cam_K, 
                                                obj_scene=obj_context, 
                                                obj_render=obj_renderer)
        tsl_cost += time.time() - tsl_timer
        raw_est_Rs.append(obj_est_R)
        raw_est_ts.append(obj_est_t)

        postp_timer = time.time() # synthesize depth image for post-process (hypotheses selection)
        obj_context.set_pose(rotation=obj_est_R, translation=obj_est_t)
        refined_est_depth, refined_est_mask = obj_renderer.render(obj_context)[1:]
        raw_depths.append(refined_est_depth)
        raw_masks.append(refined_est_mask)
        postp_cost += (time.time() - postp_timer)

        if config.USE_ICP:
            icp_timer = time.time()
            H_est = torch.eye(4, dtype=torch.float32)
            H_est[:3, :3] = obj_est_R.squeeze()
            H_est[:3, 3] = obj_est_t.squeeze()
            icp_H = torch.eye(4, dtype=torch.float32)
            if len(dst_pts) > 10:
                src_pts = pplane_ICP.depth_to_pointcloud(refined_est_depth.squeeze(), cam_K)
                if len(src_pts) > 10:
                    icp_H = pplane_ICP.sim_icp(dst_pts.to(device), 
                                                src_pts.to(device), 
                                                correspondences=config.ICP_correspondences, 
                                                max_iterations=config.ICP_max_iterations,
                                                neighbors=config.ICP_neighbors,
                                                min_planarity=config.ICP_min_planarity
                                                ).cpu().squeeze()
            H_est_refined = icp_H @ H_est
            icp_refined_R = H_est_refined[:3, :3]
            icp_refined_t = H_est_refined[:3, 3]
            icp_cost += time.time() - icp_timer
            icp_est_Rs.append(icp_refined_R)
            icp_est_ts.append(icp_refined_t)

            obj_context.set_pose(rotation=icp_refined_R, translation=icp_refined_t)
            icp_refined_depth, icp_refined_mask = obj_renderer.render(obj_context)[1:]
            icp_masks.append(icp_refined_mask)
            icp_depths.append(icp_refined_depth)

    #   pose hypotheses selection
    postp_timer = time.time() 
    num_poses = len(raw_depths)
    topk_raw_depths = torch.stack(raw_depths, dim=0).to(device)
    topk_raw_masks = torch.stack(raw_masks, dim=0).to(device)  
    rawk_err_depths = (topk_raw_depths - obj_depth.to(device)).abs() > obj_diameter * 0.1 # outliers
    topk_raw_errors = rawk_err_depths.view(num_poses, -1).sum(-1) / topk_raw_masks.view(num_poses, -1).sum(-1)
    min_raw_err, best_topk_raw_idx = torch.topk(-topk_raw_errors, k=1)
    obj_raw_R = raw_est_Rs[best_topk_raw_idx]
    obj_raw_t = raw_est_ts[best_topk_raw_idx]
    obj_rcnn_score = estimated_scores[best_topk_raw_idx]
    postp_cost += time.time() - postp_timer
    raw_topk_cost = bg_cost + prep_cost + rot_cost + tsl_cost + postp_cost

    pose_ret['raw_R'] = obj_raw_R
    pose_ret['raw_t'] = obj_raw_t
    pose_ret['raw_score'] = obj_rcnn_score
    pose_ret['raw_time'] = raw_topk_cost

    if config.USE_ICP:
        # the selected pose refined with ICP (single ICP)
        obj_icp1_R = icp_est_Rs[best_topk_raw_idx]
        obj_icp1_t = icp_est_ts[best_topk_raw_idx]
        icp_top1_cost = raw_topk_cost + icp_cost / num_poses
        pose_ret['icp1_R'] = obj_icp1_R
        pose_ret['icp1_t'] = obj_icp1_t
        pose_ret['icp1_score'] = obj_rcnn_score
        pose_ret['icp1_time'] = icp_top1_cost

        # select the final pose from all hypotheses refined with ICP  (multiple ICPs)
        topk_icp_depths = torch.stack(icp_depths, dim=0).to(device)
        topk_icp_masks = torch.stack(icp_masks, dim=0).to(device)
        icpk_err_depths = (topk_icp_depths - obj_depth.to(device)).abs() > obj_diameter * 0.1 # outliers
        topk_icp_errors = icpk_err_depths.view(num_poses, -1).sum(-1) / topk_icp_masks.view(num_poses, -1).sum(-1)
        min_icp_err, best_topk_icp_idx = torch.topk(-topk_icp_errors, k=1)
        obj_icpk_R = icp_est_Rs[best_topk_icp_idx]
        obj_icpk_t = icp_est_ts[best_topk_icp_idx]
        obj_icpk_score = estimated_scores[best_topk_icp_idx]
        icp_topk_cost = raw_topk_cost + icp_cost
        
        pose_ret['icpk_R'] = obj_icpk_R
        pose_ret['icpk_t'] = obj_icpk_t
        pose_ret['icpk_score'] = obj_icpk_score
        pose_ret['icpk_time'] = icp_topk_cost
    
    return pose_ret

def OVE6D_mask_rotation_estimation(input_depth, model_net, object_codebook, cfg):
    obj_codebook_Rs = object_codebook['Rs']       # 4000 x 3 x 3
    obj_codebook_Z_vec = object_codebook['Z_vec'] # 4000 x 64
    obj_codebook_Z_map = object_codebook['Z_map'] # 4000 x 128 x 8 x 8
    device = input_depth.device
    
    with torch.no_grad():
        obj_query_z_map, obj_query_z_vec = model_net.vipri_encoder(input_depth[None, ...], return_maps=True)
    obj_cosim_cores = F.cosine_similarity(obj_codebook_Z_vec.to(obj_query_z_vec.device), obj_query_z_vec)

    topK_cosim_idxes = obj_cosim_cores.topk(k=cfg.VP_NUM_TOPK)[1]

    estimated_scores = obj_cosim_cores[topK_cosim_idxes].detach().cpu()
    retrieved_codebook_R = obj_codebook_Rs[topK_cosim_idxes].clone()    # K x 3 x 3
    top_codebook_z_maps = obj_codebook_Z_map[topK_cosim_idxes, ...]
    
    with torch.no_grad():
        query_theta, pd_conf = model_net.inference(top_codebook_z_maps.to(device), obj_query_z_map.expand_as(top_codebook_z_maps).to(device))
    stn_theta = F.pad(query_theta[:, :2, :2].clone(), (0, 1))
    
    homo_z_R = F.pad(stn_theta, (0, 0, 0, 1))
    homo_z_R[:, -1, -1] = 1.0
    homo_z_R = homo_z_R.to(retrieved_codebook_R.device)
    estimated_xyz_R = homo_z_R @ retrieved_codebook_R
    pd_conf = pd_conf.squeeze(1)
    sorted_idxes = pd_conf.topk(k=len(pd_conf))[1]

    final_R = estimated_xyz_R[sorted_idxes][:cfg.POSE_NUM_TOPK]
    final_S = estimated_scores[sorted_idxes][:cfg.POSE_NUM_TOPK]
    
    # final_vp = retrieved_codebook_R[sorted_idxes][:cfg.POSE_NUM_TOPK]
    # print(final_vp)
   
    return final_R, final_S
           

def OVE6D_rcnn_full_pose(model_func, obj_depths, obj_masks, obj_rcnn_scores, obj_codebook, cam_K, config, obj_renderer, device, return_rcnn_idx=False):
    """
    Perform OVE6D with multiple masks predicted by Mask-RCNN
    Full pipeline (Mask-RCNN + OVE6D)
    take advantage of the detection confidence scores
    """
    pose_ret = dict()
    cam_K = cam_K.to(device)
    obj_masks = obj_masks.to(device)
    obj_depths = obj_depths.to(device)

    obj_mesh = obj_codebook['obj_mesh']
    obj_diameter = obj_codebook['diameter']
    
    bg_timer = time.time()
    obj_depths = background_filter(obj_depths, obj_diameter) # filter out outliers
    bg_cost = time.time() - bg_timer
    obj_masks[obj_depths<0] = 0
    obj_zoom_dist = config.ZOOM_DIST_FACTOR * obj_diameter * cam_K.squeeze().cpu()[0, 0]
    zoom_timer = time.time()
    zoom_test_depths, init_ts = input_zoom_preprocess(input_depth=obj_depths, 
                                                        input_mask=obj_masks, 
                                                        intrinsic=cam_K, 
                                                        device=device, 
                                                        target_zoom_dist=obj_zoom_dist,
                                                        zoom_scale_mode=config.ZOOM_MODE, 
                                                        zoom_size=config.ZOOM_SIZE) # 1xHxW
    zoom_cost = time.time() - zoom_timer # the pre-processing runtime
    rot_timer = time.time()

    (estimated_R, estimated_scores, 
    estimated_rcnn_idx) = OVE6D_rcnn_rotation_estimation(input_depth=zoom_test_depths,
                                                        rcnn_score=obj_rcnn_scores,
                                                        model_net=model_func,
                                                        object_codebook=obj_codebook, 
                                                        cfg=config)  
    num_proposals = len(estimated_R)
    init_t = init_ts[estimated_rcnn_idx] 
    obj_depth = obj_depths[estimated_rcnn_idx] 
    rot_cost = time.time() - rot_timer # rotation estimation runtime

    raw_est_Rs = []
    raw_est_ts = []
    raw_masks = []
    raw_depths = []

    icp_est_Rs = []
    icp_est_ts = []
    icp_masks = []
    icp_depths = []

    icp_timer = time.time()
    dst_pts = pplane_ICP.depth_to_pointcloud(obj_depth.squeeze(), cam_K)
    cnt_dst_pts = len(dst_pts)
    icp_dst_cost = time.time() - icp_timer
    icp_src_cost = 0
    
    # tsl_top1_cost = list()
    icp_topk_cost = list()

    tsl_cost = 0
    raw_syn_cost = 0
    icpk_syn_cost = 0
    obj_context = rendering.SceneContext(obj=obj_mesh, intrinsic=cam_K.cpu()) # define a scene
    for idx, (obj_est_R, obj_est_score) in enumerate(zip(estimated_R, estimated_scores)):
        tsl_timer = time.time()
        obj_est_t = OVE6D_translation_estimation(est_R=obj_est_R, 
                                                est_t=init_t, 
                                                intrinsic=cam_K, 
                                                obj_scene=obj_context, 
                                                obj_render=obj_renderer)
        tsl_cost += (time.time() - tsl_timer)
        # tsl_top1_cost.append(time.time() - tsl_timer)
        raw_est_Rs.append(obj_est_R)
        raw_est_ts.append(obj_est_t)
        
        if num_proposals > 1 or config.USE_ICP:
            syn_timer = time.time() # synthesize depth image for post-process (hypotheses selection)if multiple hypotheses provided
            obj_context.set_pose(rotation=obj_est_R, translation=obj_est_t)
            refined_est_depth, refined_est_mask = obj_renderer.render(obj_context)[1:]
            raw_depths.append(refined_est_depth)
            raw_masks.append(refined_est_mask)
            raw_syn_cost += (time.time() - syn_timer)

        if config.USE_ICP:
            icp_timer = time.time()
            src_pts = pplane_ICP.depth_to_pointcloud(refined_est_depth.squeeze(), cam_K)
            cnt_src_pts = len(src_pts)
            icp_src_cost += (time.time() - icp_timer)
            H_est = torch.eye(4, dtype=torch.float32)
            H_est[:3, :3] = obj_est_R.squeeze()
            H_est[:3, 3] = obj_est_t.squeeze()
            icp_H = torch.eye(4, dtype=torch.float32)
            if np.minimum(cnt_src_pts, cnt_dst_pts) > 10:
                icp_H = pplane_ICP.sim_icp(dst_pts.to(device), 
                                            src_pts.to(device), 
                                            correspondences=config.ICP_correspondences, 
                                            max_iterations=config.ICP_max_iterations,
                                            neighbors=config.ICP_neighbors,
                                            min_planarity=config.ICP_min_planarity
                                            ).cpu().squeeze()
            H_est_refined = icp_H @ H_est
            icp_refined_R = H_est_refined[:3, :3]
            icp_refined_t = H_est_refined[:3, 3]
            icp_topk_cost.append(time.time() - icp_timer)
            icp_est_Rs.append(icp_refined_R)
            icp_est_ts.append(icp_refined_t)
            if num_proposals > 1:
                syn_timer = time.time() # synthesize depth image for post-process 
                obj_context.set_pose(rotation=icp_refined_R, translation=icp_refined_t)
                icp_refined_depth, icp_refined_mask = obj_renderer.render(obj_context)[1:]
                icp_masks.append(icp_refined_mask)
                icp_depths.append(icp_refined_depth)
                icpk_syn_cost += (time.time() - syn_timer)
    raw_cost = bg_cost + zoom_cost + rot_cost + tsl_cost
    raw_select_cost = 0
    best_topk_raw_idx = 0
    obj_raw_R = raw_est_Rs[0]
    obj_raw_t = raw_est_ts[0]
    obj_score = estimated_scores[0]
    if num_proposals > 1: # pose hypotheses selection if multiple hypotheses provided
        postp_timer = time.time() 
        topk_raw_depths = torch.stack(raw_depths, dim=0).to(obj_depth.device)
        topk_raw_masks = torch.stack(raw_masks, dim=0).to(obj_depth.device)
        rawk_err_depths = (topk_raw_depths - obj_depth).abs() > obj_diameter * 0.1 # outliers
        topk_raw_errors = rawk_err_depths.view(num_proposals, -1).sum(-1) / topk_raw_masks.view(num_proposals, -1).sum(-1)
        _, best_topk_raw_idx = torch.topk(-topk_raw_errors, k=1)
        obj_raw_R = raw_est_Rs[best_topk_raw_idx]
        obj_raw_t = raw_est_ts[best_topk_raw_idx]
        obj_score = estimated_scores[best_topk_raw_idx]
        raw_select_cost = time.time() - postp_timer
    raw_postp_cost = raw_syn_cost + raw_select_cost 
    raw_cost += raw_postp_cost

    pose_ret['rcnn_idx'] = estimated_rcnn_idx
    pose_ret['raw_R'] = obj_raw_R
    pose_ret['raw_t'] = obj_raw_t
    pose_ret['raw_score'] = obj_score
    pose_ret['raw_time'] = raw_cost

    pose_ret['bg_time'] = bg_cost 
    pose_ret['zoom_time'] = zoom_cost 
    pose_ret['rot_time'] = rot_cost
    pose_ret['tsl_time'] = tsl_cost
    
    pose_ret['raw_syn_time'] = raw_syn_cost
    pose_ret['raw_select_time'] = raw_select_cost
    pose_ret['raw_postp_time'] = raw_postp_cost

    if config.USE_ICP:
        # the selected pose refined with ICP (single ICP)

        top1_icp_cost = raw_cost + icp_dst_cost + icp_src_cost/num_proposals + np.mean(icp_topk_cost)

        obj_icp1_R = icp_est_Rs[best_topk_raw_idx]
        obj_icp1_t = icp_est_ts[best_topk_raw_idx]
        
        pose_ret['icp1_R'] = obj_icp1_R
        pose_ret['icp1_t'] = obj_icp1_t
        pose_ret['icp1_score'] = obj_score
        pose_ret['icp1_ref_time'] = icp_dst_cost + icp_src_cost/num_proposals + np.mean(icp_topk_cost)
        pose_ret['icp1_rawicp_time'] = top1_icp_cost

        icpk_select_cost = 0
        best_topk_icp_idx = 0
        obj_icpk_R = obj_icp1_R
        obj_icpk_t = obj_icp1_t
        obj_icpk_score = obj_score
        if num_proposals > 1:
            # select the final pose from all hypotheses refined with ICP  (multiple ICPs)
            postp_timer = time.time() 
            topk_icp_depths = torch.stack(icp_depths, dim=0).to(device)
            topk_icp_masks = torch.stack(icp_masks, dim=0).to(device)
            icpk_err_depths = (topk_icp_depths - obj_depth.to(device)).abs() > obj_diameter * 0.1 # outliers
            topk_icp_errors = icpk_err_depths.view(num_proposals, -1).sum(-1) / topk_icp_masks.view(num_proposals, -1).sum(-1)
            _, best_topk_icp_idx = torch.topk(-topk_icp_errors, k=1)
            obj_icpk_R = icp_est_Rs[best_topk_icp_idx]
            obj_icpk_t = icp_est_ts[best_topk_icp_idx]
            obj_icpk_score = estimated_scores[best_topk_icp_idx]
            icpk_select_cost = time.time() - postp_timer
        icpk_postp_cost = icpk_syn_cost + icpk_select_cost 
        
        topk_icp_cost = raw_cost + icp_dst_cost + icp_src_cost + np.sum(icp_topk_cost) + icpk_postp_cost
        pose_ret['icpk_R'] = obj_icpk_R
        pose_ret['icpk_t'] = obj_icpk_t
        pose_ret['icpk_score'] = obj_icpk_score
        pose_ret['icpk_ref_time'] = icp_dst_cost + icp_src_cost + np.sum(icp_topk_cost) + icpk_postp_cost # icp cost
        
        pose_ret['icpk_syn_time'] = icpk_syn_cost
        pose_ret['icpk_select_time'] = icpk_select_cost
        pose_ret['icpk_postp_time'] = icpk_postp_cost
        pose_ret['icpk_rawicp_time'] = topk_icp_cost # total cost
    if return_rcnn_idx: # return the index of the final selected segmentation mask predicted by RCNN
        return pose_ret, estimated_rcnn_idx
    return pose_ret

def OVE6D_rcnn_rotation_estimation(input_depth, rcnn_score, model_net, object_codebook, cfg):
    obj_codebook_Rs = object_codebook['Rs']       # 4000 x 3 x 3
    obj_codebook_Z_vec = object_codebook['Z_vec'] # 4000 x 64
    obj_codebook_Z_map = object_codebook['Z_map'] # 4000 x 128 x 8 x 8
    
    with torch.no_grad():
        obj_query_z_map, obj_query_z_vec = model_net.vipri_encoder(input_depth, return_maps=True)
    vp_obj_cosim_scores = F.cosine_similarity(obj_codebook_Z_vec.unsqueeze(0).to(obj_query_z_vec.device), # 1x4000x64
                                                obj_query_z_vec.unsqueeze(1), dim=2) # Mx1x64 => Mx4000
    # multiple instances of a single object exist and we select the best instance for the object
    mean_vp_obj_cosim_scores = vp_obj_cosim_scores.topk(k=1, dim=1)[0].mean(dim=1).squeeze() # M
    fused_obj_scores = mean_vp_obj_cosim_scores * rcnn_score.to(mean_vp_obj_cosim_scores.device)

    best_rcnn_idx = fused_obj_scores.max(dim=0)[1] # 
    best_obj_cosim_scores = vp_obj_cosim_scores[best_rcnn_idx] # (4000,)
    best_obj_query_z_map = obj_query_z_map[best_rcnn_idx][None, ...] # 
    topK_cosim_idxes = best_obj_cosim_scores.topk(k=cfg.VP_NUM_TOPK)[1]

    estimated_scores = best_obj_cosim_scores[topK_cosim_idxes]#.detach().cpu()
    retrieved_codebook_R = obj_codebook_Rs[topK_cosim_idxes].clone()    # K x 3 x 3
    top_codebook_z_maps = obj_codebook_Z_map[topK_cosim_idxes, ...]
    
    with torch.no_grad():
        query_theta, pd_conf = model_net.inference(top_codebook_z_maps.to(best_obj_query_z_map.device), 
                                                    best_obj_query_z_map.expand_as(top_codebook_z_maps))
    stn_theta = F.pad(query_theta[:, :2, :2].clone(), (0, 1))
    
    homo_z_R = F.pad(stn_theta, (0, 0, 0, 1))
    homo_z_R[:, -1, -1] = 1.0
    estimated_xyz_R = homo_z_R @ retrieved_codebook_R.to(homo_z_R.device)
    pd_conf = pd_conf.squeeze(1)
    sorted_idxes = pd_conf.topk(k=len(pd_conf))[1]

    final_R = estimated_xyz_R[sorted_idxes][:cfg.POSE_NUM_TOPK]
    final_S = estimated_scores[sorted_idxes][:cfg.POSE_NUM_TOPK]
   
    return final_R.cpu(), final_S.cpu(), best_rcnn_idx
 

