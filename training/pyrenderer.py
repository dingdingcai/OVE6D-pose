import os
import torch
import random
import structlog
from torch.utils.data import Dataset

from lib import rendering
from lib.three import rigid

from training import data_augment
from training import train_utils

os.environ['PYOPENGL_PLATFORM'] = 'egl'
logger = structlog.get_logger(__name__)


class PyrenderDataset(Dataset):
    def __init__(self, shape_paths, config,
                 x_bound=(-0.04,0.04),
                 y_bound=(-0.02,0.02),
                 scale_jitter=(0.5, 1.0),
                 dist_jitter=(0.5, 1.5), 
                 aug_guassian_std=0.01,
                 aug_rescale_jitter=(0.2, 0.8),
                 aug_patch_area_ratio=0.2,
                 aug_patch_max_num=1
                 ):
        super().__init__()
        self.shape_paths = shape_paths
        self.width = config.RENDER_WIDTH
        self.height = config.RENDER_HEIGHT
        self.num_inputs = config.NUM_VIEWS
        self.intrinsic = config.INTRINSIC
        self.dist_base = config.RENDER_DIST     
        self.data_augment = config.USE_DATA_AUG   
        self.hist_bin_num = config.HIST_BIN_NUMS
        self.min_hist_filter_threshold = config.MIN_HIST_STAT
        self.min_dep_pixel_threshold = config.MIN_DEPTH_PIXELS
        
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.scale_jitter = scale_jitter
        self.dist_jitter = torch.tensor(dist_jitter)
        
        self.aug_guassian_std = aug_guassian_std
        self.aug_rescale_jitter = aug_rescale_jitter
        self.aug_patch_area_ratio = aug_patch_area_ratio
        self.aug_patch_max_num = aug_patch_max_num
        
        self._renderer = None
        self._worker_id = None
        self._log = None

    def __len__(self):
        return len(self.shape_paths)
    
    def worker_init_fn(self, worker_id):
        self._worker_id = worker_id
        self._log = logger.bind(worker_id=worker_id)
        self._renderer = rendering.Renderer(width=self.width, height=self.height)
        # self._log.info('renderer initialized')
        
    def random_rotation(self, n):
        random_R = rendering.random_xyz_rotation(n)
        anchor_R = random_R @ rendering.evenly_distributed_rotation(n)
        outplane_R = rendering.random_xy_rotation(n)
        inplane_R = rendering.random_z_rotation(n)
        jitter_R = rendering.random_xy_rotation(n, rang_degree=3)
        return anchor_R, inplane_R, outplane_R, jitter_R
        
    def __getitem__(self, idx):
        
        anchor_R, inplane_R, outplane_R, jitter_R = self.random_rotation(self.num_inputs)
        
        scale_jitter = random.uniform(*self.scale_jitter)
        
        while True:
            model_path = random.choice(self.shape_paths)
            file_size = model_path.stat().st_size
            max_size = 2e7
            if file_size > max_size:
                # self._log.warning('skipping large model', path=model_path, max_size=max_size, file_size=file_size)
                continue
            try:
                obj, obj_diameter = rendering.load_object(model_path, scale=scale_jitter)
                
                obj_dist = self.dist_base * obj_diameter
                z_bound = (obj_dist * min(self.dist_jitter), obj_dist * max(self.dist_jitter)) # camera distance is set to be relative to object diameter

                anchor_T = rigid.random_translation(self.num_inputs, self.x_bound, self.y_bound, z_bound)
                inplane_T = rigid.random_translation(self.num_inputs, self.x_bound, self.y_bound, z_bound)
                outplane_T = rigid.random_translation(self.num_inputs, self.x_bound, self.y_bound, z_bound)

                context = rendering.SceneContext(obj, self.intrinsic)
                break
            except ValueError as e:
                continue
                # self._log.error('exception while loading mesh', exc_info=e)
        obj_diameters = obj_diameter.repeat(self.num_inputs)
        
        anchor_masks = list()
        anchor_depths = list()

        inplane_masks = list()
        inplane_depths = list()
        
        outplane_masks = list()
        outplane_depths = list()

        jitter_inplane_depths = list()

        valid_rot_idexes = list() # the discrepancy error count between anchor camera and its out-of-plane rotation 

        # for R, T in zip(anchor_R, anchor_T):
        #     context.set_pose(rotation=R, translation=T)
        #     depth, mask = self._renderer.render(context)[1:]
        #     anchor_masks.append(mask)
        #     anchor_depths.append(depth)
        
        in_Rxyz = inplane_R @ anchor_R  # object-space rotation
        for R, T in zip(in_Rxyz, inplane_T):
            context.set_pose(rotation=R, translation=T)
            depth, mask = self._renderer.render(context)[1:]
            inplane_masks.append(mask)
            inplane_depths.append(depth)
        

        jitter_in_Rxyz = jitter_R @ in_Rxyz  # jittering the object-space rotation
        for R, T in zip(jitter_in_Rxyz, inplane_T):
            context.set_pose(rotation=R, translation=T)
            depth, mask = self._renderer.render(context)[1:]
            jitter_inplane_depths.append(depth)
        

        out_Rxy = outplane_R @ anchor_R  # object-space rotation
        for R, T in zip(out_Rxy, outplane_T):
            context.set_pose(rotation=R, translation=T)
            depth, mask = self._renderer.render(context)[1:]
            outplane_masks.append(mask)
            outplane_depths.append(depth)
            
        
        # constant_T = torch.zeros_like(anchor_T)
        # constant_T[:, -1] = obj_dist          # centerizing object with constant distance
        for anc_R, oup_R, inp_R, const_T in zip(anchor_R, out_Rxy, jitter_in_Rxyz, anchor_T):
            context.set_pose(rotation=anc_R, translation=const_T)
            anc_depth, anc_mask = self._renderer.render(context)[1:]
            context.set_pose(rotation=oup_R, translation=const_T)
            oup_depth = self._renderer.render(context)[1]

            anchor_masks.append(anc_mask)
            anchor_depths.append(anc_depth)


            # #calculate the viewpoint angles for inplane and outplane relative to anchor
            oup_vp_sim = (anc_R[2] * oup_R[2]).sum() # oup_vp_angle = arccos(oup_vp_sim)
            inp_vp_sim = (anc_R[2] * inp_R[2]).sum() # inp_vp_angle = arccos(inp_vp_sim)
            # #inp_vp_sim > oup_vp_sim is favored, inp_R is supposed to be closer to anc_R compared with oup_R

            # #the out-of-plane depth pairs (anc, out) are supposed to be having different depth distribution
            hist_diff = data_augment.divergence_depth(anc_depth, oup_depth, 
                            min_dep_pixels=self.min_dep_pixel_threshold, bins_num=self.hist_bin_num) 
            if (inp_vp_sim <= oup_vp_sim) or (hist_diff < self.min_hist_filter_threshold):
                valid_rot_idexes.append(0)  # invalid negative depth pair due to equivalent depth distribution
            else:
                valid_rot_idexes.append(1)

        del context
        valid_rot_indexes = torch.tensor(valid_rot_idexes, dtype=torch.uint8)

        anchor_masks = torch.stack(anchor_masks, dim=0).unsqueeze(1)
        anchor_depths = torch.stack(anchor_depths, dim=0).unsqueeze(1)

        inplane_masks = torch.stack(inplane_masks, dim=0).unsqueeze(1)
        inplane_depths = torch.stack(inplane_depths, dim=0).unsqueeze(1)

        jitter_inplane_depths = torch.stack(jitter_inplane_depths, dim=0).unsqueeze(1)

        outplane_masks = torch.stack(outplane_masks, dim=0).unsqueeze(1)
        outplane_depths = torch.stack(outplane_depths, dim=0).unsqueeze(1)

        anchor_extrinsic = rigid.RT_to_matrix(anchor_R, anchor_T)
        inplane_extrinsic = rigid.RT_to_matrix(in_Rxyz, inplane_T)
        outplane_extrinsic = rigid.RT_to_matrix(out_Rxy, outplane_T)

        outplane_depths_aug = outplane_depths
        inplane_depths_aug = jitter_inplane_depths

        valid_anc_idxes = torch.ones_like(valid_rot_indexes)
        valid_inp_idxes = torch.ones_like(valid_rot_indexes)
        valid_out_idxes = torch.ones_like(valid_rot_indexes)
    
        if self.data_augment:
            if random.random() > 0.5:
                inplane_depths_aug = data_augment.custom_aug(inplane_depths_aug, 
                                                            noise_level=self.aug_guassian_std,
                                                            scale_jitter=self.aug_rescale_jitter, 
                                                            area_patch=self.aug_patch_area_ratio, 
                                                            nb_patch=self.aug_patch_max_num)
            if random.random() > 0.5:
                outplane_depths_aug = data_augment.custom_aug(outplane_depths_aug, 
                                                            noise_level=self.aug_guassian_std,
                                                            scale_jitter=self.aug_rescale_jitter, 
                                                            area_patch=self.aug_patch_area_ratio, 
                                                            nb_patch=self.aug_patch_max_num)
            if random.random() > 0.5:
                inplane_depths_aug, valid_inp_idxes = data_augment.batch_data_morph(inplane_depths_aug, 
                                                                                    min_dep_pixels=self.min_dep_pixel_threshold, 
                                                                                    hole_size=5, 
                                                                                    edge_size=5)
            if random.random() > 0.5:
                outplane_depths_aug, valid_out_idxes = data_augment.batch_data_morph(outplane_depths_aug, 
                                                                                    min_dep_pixels=self.min_dep_pixel_threshold, 
                                                                                    hole_size=5, 
                                                                                    edge_size=5)
        
        return {
            'anchor': {
                'mask': anchor_masks,
                'depth': anchor_depths,
                'extrinsic': anchor_extrinsic,
                'rotation_to_anchor': torch.eye(3).expand(self.num_inputs, -1, -1),
                'valid_idx': valid_rot_indexes * valid_anc_idxes,
                'obj_diameter': obj_diameters,
            },
            'inplane': {
                'mask': inplane_masks,
                'depth': inplane_depths,
                'aug_depth': train_utils.background_filter(inplane_depths_aug, obj_diameters),
                'extrinsic': inplane_extrinsic,
                'rotation_to_anchor': inplane_R,
                'valid_idx': valid_rot_indexes * valid_inp_idxes,
                'obj_diameter': obj_diameters,
            },
            'outplane': {
                'mask': outplane_masks,
                'depth': outplane_depths,
                'aug_depth': train_utils.background_filter(outplane_depths_aug, obj_diameters),
                'extrinsic': outplane_extrinsic,
                'rotation_to_anchor': outplane_R,
                'valid_idx': valid_rot_indexes * valid_out_idxes,
                'obj_diameter': obj_diameters,
            },
        }