import torch
import math
import torch.nn.functional as F
from lib import geometry

def background_filter(depths, diameters, dist_factor=0.5):
    """
    filter out the outilers beyond the object diameter
    """
    new_depths = list()
    unsqueeze = False
    if not isinstance(diameters, torch.Tensor):
        diameters = torch.tensor(diameters)
    if diameters.dim() == 0:
        diameters = diameters[None, ...]
    if depths.dim() == 2:
        depths = depths[None, ...]
    if depths.dim() > 3:
        depths = depths.view(-1, depths.shape[-2], depths.shape[-1])
        diameters = diameters.view(-1)
        unsqueeze = True
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


def convert_3Dcoord_to_2Dpixel(obj_t, intrinsic):
    """
    convert the 3D space coordinates (dx, dy, dz) to 2D pixel coordinates (px, py, dz)
    """
    obj_t = obj_t.squeeze()
    K = intrinsic.squeeze().to(obj_t.device)
    
    assert(obj_t.dim() <= 2), 'the input dimension must be 3 or Nx3'
    assert(K.dim() <= 3), 'the input dimension must be 3x3 or Nx3x3'

    if obj_t.dim() == 1:
        obj_t = obj_t[None, ...]
    if K.dim() == 2:
        K = K.unsqueeze(0).expand(obj_t.size(0), 1, 1)

    assert obj_t.size(0) == K.size(0), 'batch size must be equal'
    dz = obj_t[:, 2]
    px = obj_t[:, 0] / dz * K[:, 0, 0] + K[:, 0, 2]
    py = obj_t[:, 1] / dz * K[:, 1, 1] + K[:, 1, 2]
    new_t = torch.stack([px, py, dz], dim=1)
    return new_t


def input_zoom_preprocess(images, target_dist, intrinsic, extrinsic=None, 
                          images_mask=None, normalize=True, dz=None,
                          target_size=128, scale_mode='nearest'):
    device = images.device
    intrinsic = intrinsic.to(device)
    height, width = images.shape[-2:]

    assert(images.dim()==3 or images.dim()==4)
    if images.dim() == 3:
        images = images[None, ...]

    if images_mask is None:
        images_mask = torch.zeros_like(images)
        images_mask[images>0] = 1.0  
    
    images_mask = images_mask.to(device)

    assert(images_mask.dim()==3 or images_mask.dim()==4)
    if images_mask.dim() == 3:
        images_mask = images_mask[None, ...]
    
    if not isinstance(target_dist, torch.Tensor):
        target_dist = torch.tensor(target_dist)

    target_dist = target_dist.to(device)

    if extrinsic is None:
        obj_translations = torch.stack(geometry.estimate_translation(depth=images, 
                                                                     mask=images_mask, 
                                                                     intrinsic=intrinsic), dim=1).to(device)
        if dz is not None:
            obj_translations[:, 2] = dz.to(device)                                                           
    else:
        extrinsic = extrinsic.to(device)
        obj_translations = extrinsic[:, :3, 3] # N x 3
    
    obj_zs = obj_translations[:, 2]
    
    if normalize:
        images -= images_mask * obj_zs[..., None, None, None].to(device)
        
    if extrinsic is None:
        cameras = geometry.Camera(intrinsic=intrinsic, height=height, width=width)
        obj_centroids = geometry.masks_to_centroids(images_mask)
        zoom_images, zoom_camera = cameras.zoom(image=images,
                                    target_dist=target_dist, 
                                    target_size=target_size, 
                                    zs=obj_zs, 
                                    centroid_uvs=obj_centroids,
                                    scale_mode=scale_mode)
        # zoom_masks, _ = cameras.zoom(image=images_mask,
        #                             target_dist=target_dist, 
        #                             target_size=target_size, 
        #                             zs=obj_zs, 
        #                             centroid_uvs=obj_centroids,
        #                             scale_mode=scale_mode)
    else:
        cameras = geometry.Camera(intrinsic=intrinsic, extrinsic=extrinsic, width=width, height=height)
        zoom_images, zoom_camera = cameras.zoom(images,
                                    target_dist=target_dist, 
                                    target_size=target_size, 
                                    scale_mode=scale_mode)
        # zoom_masks, _ = cameras.zoom(images_mask,
        #                             target_dist=target_dist, 
        #                             target_size=target_size, 
        #                             scale_mode=scale_mode)    
    return zoom_images, zoom_camera, obj_translations


def inplane_residual_theta(gt_t, init_t, gt_Rz, config, target_dist, device):
    """
    gt_t(Nx3): the ground truth translation
    est_t(Nx3: the initial translation (directly estimated from depth)
    gt_Rz(Nx3x3): the ground truth relative in-plane rotation along camera optical axis
    
    return: the relative transformation between the anchor image and the query image
    
    """    
    W = config.RENDER_WIDTH
    H = config.RENDER_HEIGHT
    fx = config.INTRINSIC[0, 0]
    fy = config.INTRINSIC[1, 1]
    cx = config.INTRINSIC[0, 2]
    cy = config.INTRINSIC[1, 2]
    
    gt_t = gt_t.clone().to(device) # Nx3
    init_t = init_t.clone().to(device)       # Nx3
    Rz_rot = gt_Rz[:, :2, :2].clone().to(device)     # Nx2x2
    
    gt_tx = gt_t[:, 0:1]
    gt_ty = gt_t[:, 1:2]
    gt_tz = gt_t[:, 2:3]
    
    init_tx = init_t[:, 0:1]
    init_ty = init_t[:, 1:2]
    init_tz = init_t[:, 2:3]
    
    if not isinstance(target_dist, torch.Tensor):
        target_dist = torch.tensor(target_dist)
    if target_dist.dim() == 1:
        target_dist = target_dist[..., None] # Nx1
    if target_dist.dim() != 0:
        assert(target_dist.dim() == init_tz.dim()), "shape must be same, however, {}, {}".format(target_dist.shape, init_tz.shape)

    init_scale = target_dist.to(device) / init_tz # Nx1 / config.ZOOM_CROP_SIZE
    
    gt_t[:, 0:1] = (gt_tx / gt_tz * fx  + cx) / W # Nx1 * gt_scale # projection to 2D image plane
    gt_t[:, 1:2] = (gt_ty / gt_tz * fy  + cy) / H # Nx1 * gt_scale
    
    init_t[:, 0:1] = (init_tx / init_tz * fx + cx) / W # Nx1 * init_scale
    init_t[:, 1:2] = (init_ty / init_tz * fy + cy) / H # Nx1 * init_scale

    offset_t = gt_t - init_t     # N x 3 [dx, dy, dz] unit with (pixel, pixel, meter)
    offset_t[:, :2] = offset_t[:, :2] * init_scale

    res_T = torch.zeros((gt_t.size(0), 3, 3), device=device) # Nx3x3
    res_T[:, :2, :2] = Rz_rot
    res_T[:, :3, 2] = offset_t

    return res_T


def spatial_transform_2D(x, theta, mode='nearest', padding_mode='border', align_corners=False):    
    assert(x.dim()==3 or x.dim()==4)
    assert(theta.dim()==2 or theta.dim()==3)
    assert(theta.shape[-2]==2 and theta.shape[-1]==3), "theta must be Nx2x3"
    if x.dim() == 3:
        x = x[None, ...]
    if theta.dim() == 2:
        theta = theta[None, ...].repeat(x.size(0), 1, 1)
    
    stn_theta = theta.clone()
    stn_theta[:, :2, :2] = theta[:, :2, :2].transpose(-1, -2)
    stn_theta[:, :2, 2:3] = -(stn_theta[:, :2, :2] @ stn_theta[:, :2, 2:3])
    
    grid = F.affine_grid(stn_theta.to(x.device), x.shape, align_corners=align_corners)
    new_x = F.grid_sample(x.type(grid.dtype), grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return new_x


def recover_full_translation(init_t, offset_t, config, target_dist, device):
    W = config.RENDER_WIDTH
    H = config.RENDER_HEIGHT  
    fx = config.INTRINSIC[0, 0]
    fy = config.INTRINSIC[1, 1]

    dx = offset_t[:, 0:1].to(device) # Bx1
    dy = offset_t[:, 1:2].to(device) # Bx1
    dz = offset_t[:, 2:3].to(device) # Bx1
        
    init_tx = init_t[:, 0:1].to(device) # Bx1
    init_ty = init_t[:, 1:2].to(device) # Bx1
    init_tz = init_t[:, 2:3].to(device) # Bx1
    
    if not isinstance(target_dist, torch.Tensor):
        target_dist = torch.tensor(target_dist)
    if target_dist.dim() == 1:
        target_dist = target_dist[..., None] # Nx1
    if target_dist.dim() != 0:
        assert(target_dist.dim() == init_tz.dim()), "shape must be same, however, {}, {}".format(target_dist.shape, init_tz.shape)

    init_scale = target_dist.to(device) / init_tz #/ config.ZOOM_CROP_SIZE
    
    est_tz = init_tz + dz.to(device)
    est_tx = est_tz * (W / init_scale / fx * dx + init_tx/init_tz) # Nx1
    est_ty = est_tz * (H / init_scale / fy * dy + init_ty/init_tz)
    
    # print(est_tx.shape, est_ty.shape, est_tz.shape)

    est_full_t = torch.cat([est_tx, est_ty, est_tz], dim=1) # Nx3
    
    return est_full_t


def residual_inplane_transform(gt_t, init_t, gt_Rz, config, target_dist, device):
    """
    gt_t(Nx3): the ground truth translation
    est_t(Nx3: the initial translation (directly estimated from depth)
    gt_Rz(Nx3x3): the ground truth relative in-plane rotation along camera optical axis
    return: the relative transformation between the anchor image and the query image
    """    
    # W = config.RENDER_WIDTH
    # H = config.RENDER_HEIGHT
    fx = config.INTRINSIC[0, 0]
    fy = config.INTRINSIC[1, 1]
    cx = config.INTRINSIC[0, 2]
    cy = config.INTRINSIC[1, 2]
    
    gt_t = gt_t.clone().to(device) # Nx3
    init_t = init_t.clone().to(device)       # Nx3
    Rz_rot = gt_Rz[:, :2, :2].clone().to(device)     # Nx2x2
    
    gt_tx = gt_t[:, 0:1]
    gt_ty = gt_t[:, 1:2]
    gt_tz = gt_t[:, 2:3]
    
    init_tx = init_t[:, 0:1]
    init_ty = init_t[:, 1:2]
    init_tz = init_t[:, 2:3]

    if not isinstance(target_dist, torch.Tensor):
        target_dist = torch.tensor(target_dist)
    if target_dist.dim() == 1:
        target_dist = target_dist[..., None] # Nx1
    if target_dist.dim() != 0:
        assert(target_dist.dim() == init_tz.dim()), "shape must be same, however, {}, {}".format(target_dist.shape, init_tz.shape)
    target_dist = target_dist.to(device)
    
    tz_offset_frac = gt_tz / init_tz  # gt_tz = tz_factor * init_tz, the ratio bwteen the ground truth distance and initial distance
    
    gt_t[:, 0:1] = (gt_tx / gt_tz * fx  + cx) # Nx1, pixel coordinate projected on 2D image plane
    gt_t[:, 1:2] = (gt_ty / gt_tz * fy  + cy) # Nx1 
    
    init_t[:, 0:1] = (init_tx / init_tz * fx + cx) # Nx1 
    init_t[:, 1:2] = (init_ty / init_tz * fy + cy) # Nx1 

    gt_crop_scaling = target_dist / gt_tz       # the scaling factor for the cropped object patch
    # init_crop_scaling = target_dist / init_tz

    gt_bbox_size = gt_crop_scaling * config.ZOOM_SIZE  # the bbox size of the cropped object with gt distance
    # init_bbox_size = gt_bbox_size * tz_offset_frac
    
    delta_px = gt_tx - init_tx  # from source image center to target image center
    delta_py = gt_ty  - init_ty # from source image center to target image center

    px_offset_frac = delta_px / gt_bbox_size # convert the offset relative to the target image size
    py_offset_frac = delta_py / gt_bbox_size # convert the offset relative to the target image size

    offset_t = torch.cat([px_offset_frac, py_offset_frac, tz_offset_frac], dim=1)
    
    res_T = torch.zeros((gt_t.size(0), 3, 3), device=device) # Nx3x3
    res_T[:, :2, :2] = Rz_rot
    res_T[:, :3, 2] = offset_t

    return res_T


def recover_residual_translation(init_t, offset_t, config, target_dist, device):
    # W = config.RENDER_WIDTH
    # H = config.RENDER_HEIGHT  
    fx = config.INTRINSIC[0, 0]
    fy = config.INTRINSIC[1, 1]
    cx = config.INTRINSIC[0, 2]
    cy = config.INTRINSIC[1, 2]

    init_t = init_t.clone().to(device)     # Nx3
    offset_t = offset_t.clone().to(device) # Nx3
    
    init_tx = init_t[:, 0:1] # Bx1
    init_ty = init_t[:, 1:2] # Bx1
    init_tz = init_t[:, 2:3] # Bx1

    px_offset_frac = offset_t[:, 0:1] # Bx1
    py_offset_frac = offset_t[:, 1:2] # Bx1
    tz_offset_frac = offset_t[:, 2:3] # Bx1
        
    init_t[:, 0:1] = (init_tx / init_tz * fx + cx) # Nx1 * init_scale
    init_t[:, 1:2] = (init_ty / init_tz * fy + cy) # Nx1 * init_scale
    
    if not isinstance(target_dist, torch.Tensor):
        target_dist = torch.tensor(target_dist)
    if target_dist.dim() == 1:
        target_dist = target_dist[..., None] # Nx1
    if target_dist.dim() != 0:
        assert(target_dist.dim() == init_tz.dim()), "shape must be same, however, {}, {}".format(target_dist.shape, init_tz.shape)
    target_dist = target_dist.to(device)
    
    init_crop_scaling = target_dist / init_tz
    init_bbox_size = init_crop_scaling * config.ZOOM_SIZE
    pd_bbox_size = init_bbox_size / tz_offset_frac

    pd_delta_px = px_offset_frac * pd_bbox_size
    pd_delta_py = py_offset_frac * pd_bbox_size

    pd_px = init_t[:, 0:1] + pd_delta_px
    pd_py = init_t[:, 1:2] + pd_delta_py

    est_tz = tz_offset_frac * init_tz

    # est_tz = init_tz + tz_offset_frac * init_tz

    
    est_tx = (pd_px - cx) / fx * est_tz
    est_ty = (pd_py - cy) / fy * est_tz
    
    est_full_t = torch.cat([est_tx, est_ty, est_tz], dim=1) # Nx3
    
    return est_full_t


def residual_inplane_transform3(gt_t, init_t, gt_Rz, config, target_dist, device):
    """
    gt_t(Nx3): the ground truth translation
    est_t(Nx3: the initial translation (directly estimated from depth)
    gt_Rz(Nx3x3): the ground truth relative in-plane rotation along camera optical axis
    return: the relative transformation between the anchor image and the query image
    """    
    # W = config.RENDER_WIDTH
    # H = config.RENDER_HEIGHT
    fx = config.INTRINSIC[0, 0]
    fy = config.INTRINSIC[1, 1]
    cx = config.INTRINSIC[0, 2]
    cy = config.INTRINSIC[1, 2]
    
    gt_t = gt_t.clone().to(device) # Nx3
    init_t = init_t.clone().to(device)       # Nx3
    Rz_rot = gt_Rz[:, :2, :2].clone().to(device)     # Nx2x2
    
    gt_tx = gt_t[:, 0:1]
    gt_ty = gt_t[:, 1:2]
    gt_tz = gt_t[:, 2:3]

    init_tx = init_t[:, 0:1]
    init_ty = init_t[:, 1:2]
    init_tz = init_t[:, 2:3]

    # tz_offset_frac = (gt_tz - init_tz) / init_tz  # gt_tz = init_tz + tz_offset_frac * init_tz

    tz_offset_frac = (gt_tz - init_tz)# / init_tz  # gt_tz = init_tz + tz_offset_frac * init_tz

    if not isinstance(target_dist, torch.Tensor):
        target_dist = torch.tensor(target_dist)
    if target_dist.dim() == 1:
        target_dist = target_dist[..., None] # Nx1
    if target_dist.dim() != 0:
        assert(target_dist.dim() == init_tz.dim()), "shape must be same, however, {}, {}".format(target_dist.shape, init_tz.shape)
    target_dist = target_dist.to(device)    
    
    # object GT 2D center in image
    gt_px = (gt_tx / gt_tz * fx + cx)      # Nx1, pixel x-coordinate of the object gt_center
    gt_py = (gt_ty / gt_tz * fy + cy)      # Nx1, pixel y-coordinate of the object gt_center

    # object initial 2D center in image
    init_px = (init_tx / init_tz * fx + cx) # Nx1 
    init_py = (init_ty / init_tz * fy + cy) # Nx1 

    offset_px = gt_px - init_px  # from source image center to target image center
    offset_py = gt_py - init_py # from source image center to target image center

    # gt_box_size = 1.0 * target_dist / gt_tz * config.ZOOM_SIZE     # cropped patch size with the gt depth
    init_box_size = 1.0 * target_dist / init_tz * config.ZOOM_SIZE # cropped patch size with the estimated depth

    px_offset_frac = offset_px / (init_box_size / 2.0)
    py_offset_frac = offset_py / (init_box_size / 2.0)

    offset_t = torch.cat([px_offset_frac, py_offset_frac, tz_offset_frac], dim=1)
    
    res_T = torch.zeros((gt_t.size(0), 3, 3), device=device) # Nx3x3
    res_T[:, :2, :2] = Rz_rot
    res_T[:, :3, 2] = offset_t

    return res_T


def recover_residual_translation3(init_t, offset_t, config, target_dist, device):
    # W = config.RENDER_WIDTH
    # H = config.RENDER_HEIGHT  
    fx = config.INTRINSIC[0, 0]
    fy = config.INTRINSIC[1, 1]
    cx = config.INTRINSIC[0, 2]
    cy = config.INTRINSIC[1, 2]

    init_t = init_t.clone().to(device)     # Nx3
    offset_t = offset_t.clone().to(device) # Nx3
    
    init_tx = init_t[:, 0:1] # Bx1
    init_ty = init_t[:, 1:2] # Bx1
    init_tz = init_t[:, 2:3] # Bx1

    px_offset_frac = offset_t[:, 0:1] # Bx1
    py_offset_frac = offset_t[:, 1:2] # Bx1
    tz_offset_frac = offset_t[:, 2:3] # Bx1
        
    init_px = (init_tx / init_tz * fx + cx) # Nx1 * init_scale
    init_py = (init_ty / init_tz * fy + cy) # Nx1 * init_scale
    
    if not isinstance(target_dist, torch.Tensor):
        target_dist = torch.tensor(target_dist)
    if target_dist.dim() == 1:
        target_dist = target_dist[..., None] # Nx1
    if target_dist.dim() != 0:
        assert(target_dist.dim() == init_tz.dim()), "shape must be same, however, {}, {}".format(target_dist.shape, init_tz.shape)
    target_dist = target_dist.to(device)
    
    init_box_size = 1.0 * target_dist / init_tz * config.ZOOM_SIZE  # cropped patch size with the estimated depth

    est_px = init_px + px_offset_frac / 2.0 * init_box_size
    est_py = init_py + py_offset_frac / 2.0 * init_box_size

    est_tz = init_tz + tz_offset_frac # * init_tz
    est_tx = (est_px - cx) / fx * est_tz
    est_ty = (est_py - cy) / fy * est_tz
        
    est_full_t = torch.cat([est_tx, est_ty, est_tz], dim=1) # Nx3
    
    return est_full_t


def dynamic_margin(x_vp, y_vp, max_margin=0.5, threshold_angle=math.pi/2):
    """
    given two viewpoint vector (Nx3), calcuate the dynamic margin for the triplet loss
    """
    assert(max_margin>=0 and max_margin<=1), "maximum margin must be between (0, 1)"
    vp_cosim = (x_vp * y_vp).sum(dim=1, keepdim=True) # Nx1
    vp_angle = torch.arccos(vp_cosim)
    threshold = torch.ones_like(vp_cosim) * threshold_angle
    vp_cosim[vp_angle>threshold] = 0.0
    dynamic_margin = max_margin * (1 - vp_cosim) # smaller margin for more similar viewpoint pairs
    return dynamic_margin

