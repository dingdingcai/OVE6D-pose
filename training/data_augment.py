import cv2
import torch
import numpy as np
import imgaug.augmenters as iaa
import torchvision.transforms.functional as tf

from lib import geometry


def divergence_depth(anchor_depth, query_depth, min_dep_pixels=100, bins_num=100):
    hist_diff = 0
    anc_val_idx = anchor_depth>0
    que_val_idx = query_depth>0
    if anc_val_idx.sum() > min_dep_pixels and que_val_idx.sum() > min_dep_pixels:
        anc_vals = anchor_depth[anc_val_idx]
        que_vals = query_depth[que_val_idx]
        min_val = torch.minimum(anc_vals.min(), que_vals.min())
        max_val = torch.maximum(anc_vals.max(), que_vals.max())
        anc_hist = torch.histc(anc_vals, bins=bins_num, min=min_val, max=max_val)
        que_hist = torch.histc(que_vals, bins=bins_num, min=min_val, max=max_val)
        hist_diff = (que_hist - anc_hist).abs().mean()
    return hist_diff


def batch_data_morph(depths, min_dep_pixels=None, hole_size=5, edge_size=5):
    new_depths = list()
    unsqueeze = False
    use_filter = False
    if min_dep_pixels is not None and isinstance(min_dep_pixels, int):
        use_filter = True

    if depths.dim() == 2:
        depths = depths[None, ...]
    if depths.dim() > 3:
        depths = depths.view(-1, depths.shape[-2], depths.shape[-1])
        unsqueeze = True
        
    valid_idxes = torch.zeros(len(depths), dtype=torch.uint8)
    for ix, dep in enumerate(depths):
        dep = torch.tensor(
            cv2.morphologyEx(
                cv2.morphologyEx(dep.detach().cpu().numpy(), 
                                 cv2.MORPH_CLOSE, 
                                 np.ones((hole_size, hole_size), np.uint8)
                                ), 
                cv2.MORPH_OPEN, np.ones((edge_size, edge_size), np.uint8)
            )
        )
        new_depths.append(dep)
        if use_filter and (dep>0).sum() > min_dep_pixels:        
                valid_idxes[ix] = 1
    new_depths = torch.stack(new_depths, dim=0).to(depths.device)
    if unsqueeze:
        new_depths = new_depths.unsqueeze(1) 
    if use_filter:
        return new_depths, valid_idxes
    return new_depths


def random_block_patches(tensor, max_area_cov=0.2, max_patch_nb=5):
    assert tensor.dim() == 4, "input must be BxCxHxW {}".format(tensor.shape)
    def square_patch(tensor, max_coverage=0.05):
        data_tensor = tensor.clone()
        batchsize, channel, height, width = data_tensor.shape
        coverage = torch.rand(len(data_tensor)) * (max_coverage - 0.01) + 0.01
        patches_size = (coverage.sqrt() * np.minimum(height, width)).type(torch.int64)
        square_mask = torch.zeros_like(data_tensor, dtype=torch.float32)
        x_offset = ((width - patches_size) * torch.rand(len(patches_size))).type(torch.int64)
        y_offset = ((height - patches_size) * torch.rand(len(patches_size))).type(torch.int64)
        for ix in range(batchsize):
            square_mask[ix, :, :patches_size[ix], :patches_size[ix]] = 1
            t_mask = tf.affine(img=square_mask[ix], angle=0, translate=(x_offset[ix], y_offset[ix]), scale=1.0, shear=0)
            data_tensor[ix] *= (1 - t_mask.to(data_tensor.device))
        return data_tensor
    def circle_patch(tensor, max_coverage=0.05):
        data_tensor = tensor.clone()
        batchsize, channel, height, width = data_tensor.shape
        coverage = torch.rand(len(data_tensor)) * (max_coverage - 0.01) + 0.01
        patches_size = (coverage.sqrt() * np.minimum(height, width)).type(torch.int64)
        circle_mask = torch.zeros_like(data_tensor, dtype=torch.float32)
        radius = (patches_size / 2.0 - 0.5)[..., None, None, None]
        grid_map = torch.stack(
            torch.meshgrid(torch.linspace(0, height, height+1)[:-1], 
            torch.linspace(0, width, width+1)[:-1]), dim=0
        ).expand(batchsize, -1, -1, -1)
        distance = ((grid_map[:, 0:1, :, :] - radius)**2 + (grid_map[:, 1:2, :, :] - radius)**2).sqrt()
        circle_mask[distance<radius] = 1.0
        x_offset = ((width - patches_size) * torch.rand(len(patches_size))).type(torch.int64)
        y_offset = ((height - patches_size) * torch.rand(len(patches_size))).type(torch.int64)
        for ix in range(batchsize):
            t_mask = tf.affine(img=circle_mask[ix], angle=0, translate=(x_offset[ix], y_offset[ix]), scale=1.0, shear=0)
            data_tensor[ix] *= (1 - t_mask.to(data_tensor.device))
        return data_tensor
            
    new_tensor = tensor
    choices = 3
    max_coverage = max_area_cov / max_patch_nb
    for i in range(max_patch_nb):
        prob = torch.rand([])
        if prob <= 1.0/choices:
            new_tensor = square_patch(new_tensor, max_coverage)
        elif prob <= 2.0/choices:
            new_tensor = circle_patch(new_tensor, max_coverage)
    return new_tensor


def custom_aug(data, scale_jitter=(0.1, 0.4), nb_patch=2, area_patch=0.2, noise_level=0.01):  
    if data.dim() == 3:
        data = data.unsqueeze(1)
    assert scale_jitter[0] <= scale_jitter[1] and scale_jitter[1]<=1.0 and scale_jitter[0] >=0
    scaler = list(np.random.random(len(data))*(scale_jitter[1] - scale_jitter[0]) + scale_jitter[0])

    aug = iaa.KeepSizeByResize(
        [
            iaa.Resize(scaler),
            iaa.AdditiveLaplaceNoise(loc=0, scale=(0, 0.01), per_channel=True),
            # iaa.CoarseDropout(p=(0.01, 0.05),  
            #                   size_percent=(0.1, 0.2),
            #                  ),
            iaa.Cutout(nb_iterations=(1, 5), 
                       position='normal',
                       size=(0.01, 0.1), 
                       cval=0.0, 
                       fill_mode='constant', 
                       squared=0.1),
            iaa.GaussianBlur(sigma=(0.0, 1.5),),
            # iaa.AverageBlur(k=(2, 5)),
        ],
        interpolation=["nearest", "linear"],
    )
    aug_depths = aug(images=data.detach().cpu().permute(0, 2, 3, 1).numpy())
    aug_depths = torch.tensor(aug_depths).permute(0, 3, 1, 2).to(data.device)  # B x C x H x W
    aug_depths[data<=0] = 0

    if nb_patch > 0:
            aug_depths = random_block_patches(aug_depths.clone().to(data.device), max_area_cov=area_patch, max_patch_nb=nb_patch)
    return aug_depths


def zoom_and_crop(images, extrinsic, obj_diameter, cam_config, normalize=True, nan_check=False):
    device = images.device
    extrinsic = extrinsic.to(device)
    obj_diameter = obj_diameter.to(device)

    target_zoom_dist = cam_config.ZOOM_DIST_FACTOR * obj_diameter
    
    height, width = images.shape[-2:]
    cameras = geometry.Camera(intrinsic=cam_config.INTRINSIC.to(device), extrinsic=extrinsic.to(device), width=width, height=height)
    images_mask = torch.zeros_like(images)
    images_mask[images>0] = 1.0  

    # substract mean depth value
    obj_dist = extrinsic[:, 2, 3]
    images -= images_mask * obj_dist[..., None, None, None].to(device)  # substract the mean value
    
    # add noise based on object diameter
    random_noise = obj_diameter * (torch.rand_like(obj_diameter) - 0.5) # add noise to the depth image
    images += images_mask * random_noise[..., None, None, None] 
    
    zoom_images, _ = cameras.zoom(images, target_size=cam_config.ZOOM_CROP_SIZE, target_dist=target_zoom_dist, scale_mode=cam_config.ZOOM_MODE)

    if nan_check:
        nan_cnt = torch.isnan(zoom_images.view(len(zoom_images), -1)).sum(dim=1)  # calculate the amount of images containing NaN values
        val_idx = nan_cnt < 1   # return batch indexes of non-NaN images
        return zoom_images, val_idx
    return zoom_images
