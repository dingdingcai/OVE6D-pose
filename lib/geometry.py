"""
This code is borrowed from LatentFusion https://github.com/NVlabs/latentfusion/blob/master/latentfusion/modules/geometry.py
"""
import torch
from skimage import morphology
from torch.nn import functional as F
from lib import three


def inplane_2D_spatial_transform(R, img, mode='nearest', padding_mode='border', align_corners=False):    
    if R.dim() == 2:
        R = R[None, ...]
    Rz = R[:, :2, :2].transpose(-1, -2).clone()
    
    if img.dim() == 2:
        img = img[None, None, ...]
    if img.dim() == 3:
        img = img[None, ...]
    theta = F.pad(Rz, (0, 1))
    grid = F.affine_grid(theta.to(img.device), img.shape, align_corners=align_corners)
    new_img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return new_img

    
# @torch.jit.script
def masks_to_viewports(masks, pad: float = 10):
    viewports = []
    padding = torch.tensor([-pad, -pad, pad, pad], dtype=torch.float32, device=masks.device)

    for mask in masks:
        if mask.sum() == 0:
            height, width = mask.shape[-2:]
            viewport = torch.tensor([0, 0, width, height], dtype=torch.float32, device=masks.device)
        else:
            coords = torch.nonzero(mask.squeeze()).float()
            xmin = coords[:, 1].min()
            ymin = coords[:, 0].min()
            xmax = coords[:, 1].max()
            ymax = coords[:, 0].max()
            viewport = torch.stack([xmin, ymin, xmax, ymax])
        viewport = viewport + padding
        viewports.append(viewport)

    return torch.stack(viewports, dim=0)

# @torch.jit.script
def masks_to_centroids(masks):
    viewports = masks_to_viewports(masks, 0.0)
    cu = (viewports[:, 2] + viewports[:, 0]) / 2.0
    cv = (viewports[:, 3] + viewports[:, 1]) / 2.0

    return torch.stack((cu, cv), dim=-1)


def _erode_mask(mask, size=5):
    device = mask.device
    eroded = mask.cpu().squeeze(0).numpy()
    eroded = morphology.binary_erosion(eroded, selem=morphology.disk(size))
    eroded = torch.tensor(eroded, device=device, dtype=torch.bool).unsqueeze(0)
    if len(eroded) < 10:
        return mask
    return eroded


def _reject_outliers(data, m=1.5):
    mask = torch.abs(data - torch.median(data)) < m * torch.std(data)
    num_rejected = (~mask).sum().item()
    return data[mask], num_rejected


def _reject_outliers_med(data, m=2.0):
    median = data.median()
    med = torch.median(torch.abs(data - median))
    mask = torch.abs(data - median) / med < m
    num_rejected = (~mask).sum().item()
    return data[mask], num_rejected


def estimate_camera_dist(depth, mask):        
    num_batch = depth.shape[0]
    zs = torch.zeros(num_batch, device=depth.device)
    mask = mask.bool()
    for i in range(num_batch):
        _mask = _erode_mask(mask[i], size=3) # smooth mask, e.g. hole filling
        depth_vals = depth[i][_mask & (depth[i] > 0.0)]
        if len(depth_vals) > 0:
            depth_vals, num_rejected = _reject_outliers_med(depth_vals, m=3.0)
            if len(depth_vals) > 0:
                _min = depth_vals.min()
                _max = depth_vals.max()
            else:
                depth_vals = depth[i][_mask & (depth[i] > 0.0)]
                _min = depth_vals.min()
                _max = depth_vals.max()
        else:
            depth_vals = depth[i][depth[i] > 0.0]
            if len(depth_vals) > 0:
                _min = depth_vals.min()
                _max = depth_vals.max()
            else:
                _min = 1.0
                _max = 1.0
        zs[i] = (_min + _max) / 2.0
    return zs


def estimate_translation(depth, mask, intrinsic):
        
    depth, _ = three.ensure_batch_dim(depth, num_dims=3)
    mask, _ = three.ensure_batch_dim(mask, num_dims=3)
    z_cam = estimate_camera_dist(depth, mask)
    centroid_uv = masks_to_centroids(mask)

    u0 = intrinsic[..., 0, 2]
    v0 = intrinsic[..., 1, 2]
    fu = intrinsic[..., 0, 0]
    fv = intrinsic[..., 1, 1]
    x_cam = (centroid_uv[:, 0] - u0) / fu * z_cam
    y_cam = (centroid_uv[:, 1] - v0) / fv * z_cam

    return x_cam, y_cam, z_cam


def _grid_sample(tensor, grid, **kwargs):
    return F.grid_sample(tensor.float(), grid.float(),align_corners=False, **kwargs)


# @torch.jit.script
def bbox_to_grid(bbox, in_size, out_size):
    h = in_size[0]
    w = in_size[1]
    xmin = bbox[0].item()
    ymin = bbox[1].item()
    xmax = bbox[2].item()
    ymax = bbox[3].item()
    grid_y, grid_x = torch.meshgrid([
        torch.linspace(ymin / h, ymax / h, out_size[0], device=bbox.device) * 2 - 1,
        torch.linspace(xmin / w, xmax / w, out_size[1], device=bbox.device) * 2 - 1,
    ])
    return torch.stack((grid_x, grid_y), dim=-1)


# @torch.jit.script
def bboxes_to_grid(boxes, in_size, out_size):
    grids = torch.zeros(boxes.size(0), out_size[1], out_size[0], 2, device=boxes.device)
    for i in range(boxes.size(0)):
        box = boxes[i]
        grids[i, :, :, :] = bbox_to_grid(box, in_size, out_size)
    return grids


class Camera(torch.nn.Module):
    def __init__(self, intrinsic, extrinsic=None, viewport=None, width=640, height=480, rotation=None, translation=None):
        super().__init__()
        if intrinsic.dim() == 2:
            intrinsic = intrinsic.unsqueeze(0)
        if intrinsic.shape[1] == 3 and intrinsic.shape[2] == 3:
            intrinsic = three.rigid.intrinsic_to_3x4(intrinsic)
        
        if viewport is None:
            viewport = (torch.tensor((0, 0, width, height), dtype=torch.float32).view(1, 4).expand(intrinsic.shape[0], -1))
        if viewport.dim() == 1:
            viewport = viewport.unsqueeze(0)

        self.width = width
        self.height = height
        self.register_buffer('viewport', viewport.to(intrinsic.device)) # Nx4
        self.register_buffer('intrinsic', intrinsic) # Nx3x4 matrix
        
        if extrinsic is not None:
            if extrinsic.dim() == 2:
                extrinsic = extrinsic.unsqueeze(0) # Nx4x4
            homo_rotation_mat, homo_translation_mat = three.rigid.decompose(extrinsic)
            rotation = homo_rotation_mat[:, :3, :3].contiguous() # Nx3x3
            translation = homo_translation_mat[:, :3, -1].contiguous() # Nx3
        
        # if translation is None:
        #     raise ValueError("translation must be given through extrinsic or explicitly.")
        # elif translation.dim() == 1:
        #     translation = translation.unsqueeze(0)

        if translation is not None and translation.dim() == 1:
            translation = translation.unsqueeze(0)
        
        
        # if rotation is None:
        #     raise ValueError("rotation must be given through extrinsic or explicitly.")
        # elif rotation.dim() == 2:
        #     rotation = rotation.unsqueeze(0) # Nx3x3

        if rotation is not None and rotation.dim() == 2:
            rotation = rotation.unsqueeze(0) # Nx3x3
        if translation is not None:
            self.register_buffer('translation', translation.to(intrinsic.device))
        else:
            self.register_buffer('translation', None)
        if rotation is not None:
            self.register_buffer('rotation', rotation.to(intrinsic.device))
        else:
            self.register_buffer('rotation', None)
        
        
    
    def to_kwargs(self):
        return {
            'intrinsic': self.intrinsic,
            'extrinsic': self.extrinsic,
            'viewport': self.viewport,
            'height': self.height,
            'width': self.width,
        }
    
    @classmethod
    def from_kwargs(self, kwargs):
        _kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, list):
                _kwargs[k] = torch.tensor(v, dtype=torch.float32)
            else:
                _kwargs[k] = v
        return cls(**_kwargs)
        
    @property
    def device(self):
        return self.intrinsic.device
    
    @property
    def translation_matrix(self):
        eye = torch.eye(4, device=self.translation.device)
        homo_translation_mat = F.pad(self.translation.unsqueeze(2), (3, 0, 0, 1)) # Nx3 ==> Nx4x4
        homo_translation_mat += eye
        return homo_translation_mat
    
    @property
    def rotation_matrix(self):
        homo_rotation_mat = F.pad(self.rotation, (0, 1, 0, 1)) # Nx3x3==> Nx4x4
        homo_rotation_mat[:, -1, -1] = 1.0
        return homo_rotation_mat
    
    @property
    def extrinsic(self):
        homo_extrinsic_mat = self.translation_matrix @ self.rotation_matrix
        return homo_extrinsic_mat
    
    @extrinsic.setter
    def extrinsic(self, extrinsic):
        homo_rotation_mat, homo_translation_mat = three.rigid.decompose(extrinsic)
        rotation = homo_rotation_mat[:, :3, :3].contiguous() # Nx3x3
        translation = homo_translation_mat[:, :3, -1].contiguous() # Nx3
        self.rotation.copy_(rotation)
        self.translation.copy_(translation)

    @property
    def inv_translation_matrix(self):
        eye = torch.eye(4, device=self.translation.device)
        homo_inv_translation_mat = F.pad(-self.translation.unsqueeze(2), (3, 0, 0, 1))
        homo_inv_translation_mat += eye
        return homo_inv_translation_mat
    
    @property
    def inv_intrinsic(self):
        return torch.inverse(self.intrinsic[:, :3, :3])

    @property
    def viewport_height(self):
        return self.viewport[:, 3] - self.viewport[:, 1]
    
    @property
    def viewport_width(self):
        return self.viewport[:, 2] - self.viewport[:, 0]
    
    @property
    def viewport_centroid(self):
        cx = (self.viewport[:, 2] + self.viewport[:, 0]) / 2.0
        cy = (self.viewport[:, 3] + self.viewport[:, 1]) / 2.0
        return torch.stack((cx, cy), dim=-1) # N x 2
    
    @property
    def u0(self):
        return self.intrinsic[:, 0, 2]

    @property
    def v0(self):
        return self.intrinsic[:, 1, 2]

    @property
    def fu(self):
        return self.intrinsic[:, 0, 0]

    @property
    def fv(self):
        return self.intrinsic[:, 1, 1]

    @property
    def fov_u(self):
        return torch.atan2(self.fu, self.viewport_width / 2.0)

    @property
    def fov_v(self):
        return torch.atan2(self.fv, self.viewport_height / 2.0)
    
    @property
    def obj_to_cam(self):
        return self.translation_matrix @ self.rotation_matrix # Nx4x4, i.e. camera extrinsic or object pose

    @property
    def cam_to_obj(self):
        return self.rotation_matrix.transpose(2, 1) @ self.inv_translation_matrix # Nx4x4
    
    @property
    def obj_to_image(self):
        """
           projection onto image plane based on camera intrinsic
        """
        return self.intrinsic @ self.obj_to_cam # Nx3x4, projection

    @property
    def position(self):
        """
           obtain camera position based on camera extrinsic
        """
        # C = (-R^T)*t
        cam_position = -self.rotation_matrix[:, :3, :3].transpose(2, 1) @ self.translation_matrix[:, :3, 3, None]
        cam_position = cam_position.squeeze(-1) # Nx3x1 ==> Nx3
        return cam_position
    @property
    def direction(self):
        """
           the direction of the vector from object center to camera center, i.e. normalize camera postion
        """
        vector_direction = self.position / torch.norm(self.position, dim=1, p=2, keepdim=True) # Nx3
        return vector_direction
    
    @property
    def length(self):
        return self.intrinsic.shape[0]
    
    def rotate(self, rotation):
        rotation, unsqueezed = three.core.ensure_batch_dim(rotation, 2)
        if rotation.shape[0] == 1:
            rotation = rotation.expand_as(self.rotation)
        self.rotation = rotation @ self.rotation
        return self
        
    def translate(self, offset):
        """
            move postion of the camera based on given offset
        """
        assert offset.shape[-1] == 3 or offset.shape[-1] ==1, "offset must be an single number or tuple(x, y, z)"
        offset, unsqueezed = three.core.ensure_batch_dim(offset, 1) # 3==>1x3
        if offset.shape[0] == 1:
            offset = offset.expand_as(self.position) # N x 3
        homo_position = three.core.homogenize(self.position + offset).unsqueeze(-1) # cam new position, Nx4x1
        self.translation = -self.rotation_matrix @ homo_position.squeeze(2) # the relative translation of object
        return self
    
    def zoom(self, image, target_size, target_dist,
                           zs=None, centroid_uvs=None, target_fu=None, target_fv=None, scale_mode='bilinear'):
        """
        zoom the image and crop the image based on the given square size
        Args:
        image: the target image for zooming transformation
        target_size: the target crop image size
        target_dist: the target zoom distance from the origin
        target_fu: the target horizontal focal length
        target_fv: the target vertical focal length
        zs: the oringal distance from image to camera
        centroid_uvs: the target center for zooming 
        """
        K = self.intrinsic
        fu = K[:, 0, 0]
        fv = K[:, 1, 1]
        if zs is None:
            zs = self.translation_matrix[:, 2, 3] # if not given, set it from camera extrinsic
        
        if target_fu is None:
            target_fu = fu # if not given, set it from camera intrinsic, fx
        if target_fv is None:
            target_fv = fv # if not given, set it from camera intrinsic, fy
        
        if centroid_uvs is None:
            origin = (torch.tensor((0, 0, 0, 1.0), device=self.device).view(1, -1, 1).expand(self.length, -1, -1))
            uvs = K @ self.obj_to_cam @ origin # center of interest (centered with object)
            uvs = (uvs[:, :2] / uvs[:, 2, None]).transpose(2, 1).squeeze(1)
            centroid_uvs = uvs.clone().float()
        
        if isinstance(target_size, torch.Tensor):
            target_size = target_size.to(self.device)

        if isinstance(target_dist, torch.Tensor):
            target_dist = target_dist.to(self.device)
        
        bbox_u = 1.0 * target_dist / zs / fu * target_fu * target_size / self.width
        bbox_v = 1.0 * target_dist / zs / fv * target_fv * target_size / self.height
        
        center_u = centroid_uvs[:, 0] / self.width  # object center from pixel coordinate to scale ratio
        center_v = centroid_uvs[:, 1] / self.height
    
        boxes = torch.zeros(centroid_uvs.size(0), 4, device=self.device)
        boxes[:, 0] = (center_u - bbox_u / 2) * float(self.width)
        boxes[:, 1] = (center_v - bbox_v / 2) * float(self.height)
        boxes[:, 2] = (center_u + bbox_u / 2) * float(self.width)
        boxes[:, 3] = (center_v + bbox_v / 2) * float(self.height)
        camera_new = Camera(intrinsic=self.intrinsic, 
                            extrinsic=None,  
                            viewport=boxes,
                            width=self.width, 
                            height=self.height, 
                            rotation=self.rotation, 
                            translation=self.translation)
        if image is None:
            return camera_new
                    
        in_size = torch.tensor((self.height, self.width), device=self.device)
        out_size = torch.tensor((target_size, target_size), device=self.device)
        grids = bboxes_to_grid(boxes, in_size, out_size)
        zoomed_image = _grid_sample(image, grids, mode=scale_mode, padding_mode='zeros')

        return zoomed_image, camera_new
        
    def crop_to_viewport(self, image, target_size, scale_mode='nearest'):
        in_size = torch.tensor((self.height, self.width), device=self.device)
        out_size = torch.tensor((target_size, target_size), device=self.device)
        grid = bboxes_to_grid(self.viewport, in_size, out_size)
        return _grid_sample(image, grid, mode=scale_mode)
    
    def uncrop(self, image, scale_mode='nearest'):
        camera_new = Camera(intrinsic=self.intrinsic, 
                            extrinsic=None,
                            width=self.width, 
                            height=self.height, 
                            rotation=self.rotation, 
                            translation=self.translation)
        if image is None:
            return camera_new
        
        yy, xx = torch.meshgrid([torch.arange(0, self.height, device=self.device, dtype=torch.float32),
                                 torch.arange(0, self.width, device=self.device, dtype=torch.float32)])
        yy = yy.unsqueeze(0).expand(image.shape[0], -1, -1)
        xx = xx.unsqueeze(0).expand(image.shape[0], -1, -1)
        yy = (yy - self.viewport[:, 1, None, None]) / self.viewport_height[:, None, None] * 2 - 1
        xx = (xx - self.viewport[:, 0, None, None]) / self.viewport_width[:, None, None] * 2 - 1
        grid = torch.stack((xx, yy), dim=-1)
        uncroped_image = _grid_sample(image, grid, mode=scale_mode, padding_mode='zeros')
        
        return uncroped_image, camera_new
    
    def pixel_coords_uv(self, out_size):
        if isinstance(out_size, int):
            out_size = (out_size, out_size)

        v_pixel, u_pixel = torch.meshgrid([
            torch.linspace(0.0, 1.0, out_size[0], device=self.device),
            torch.linspace(0.0, 1.0, out_size[1], device=self.device),
        ])

        u_pixel = u_pixel.expand(self.length, -1, -1)
        u_pixel = (u_pixel * self.viewport_width.view(-1, 1, 1) + self.viewport[:, 0].view(-1, 1, 1))
        v_pixel = v_pixel.expand(self.length, -1, -1)
        v_pixel = (v_pixel * self.viewport_height.view(-1, 1, 1) + self.viewport[:, 1].view(-1, 1, 1))

        return u_pixel, v_pixel
    
    def depth_camera_coords(self, depth):
        u_pixel, v_pixel = self.pixel_coords_uv((depth.shape[-2], depth.shape[-1]))
        z_cam = depth.view_as(u_pixel)

        u0 = self.u0.view(-1, 1, 1)
        v0 = self.v0.view(-1, 1, 1)
        fu = self.fu.view(-1, 1, 1)
        fv = self.fv.view(-1, 1, 1)
        x_cam = (u_pixel - u0) / fu * z_cam
        y_cam = (v_pixel - v0) / fv * z_cam

        return x_cam, y_cam, z_cam

    def __getitem__(self, idx):
        return Camera(intrinsic=self.intrinsic[idx], 
                      extrinsic=None,  
                      viewport=self.viewport[idx],
                      width=self.width, 
                      height=self.height, 
                      rotation=self.rotation[idx], 
                      translation=self.translation[idx])
    
    def __setitem__(self, idx, camera):
        self.intrinsic[idx] = camera.intrinsic
        self.viewport[idx] = camera.viewport
        self.rotation[idx] = camera.rotation
        self.translation[idx] = camera.translation
        
    def __len__(self):
        return self.length
    
    def __iter__(self):
        cameras = [self[i] for i in range(len(self))]
        return iter(cameras)
                                                
    def clone(self):
        return Camera(self.intrinsic.clone(),
                      extrinsic=None,
                      viewport=self.viewport.clone(),
                      rotation=self.rotation.clone(),
                      translation=self.translation.clone(),
                      width=self.width,
                      height=self.height)      

    def detach(self):
        return Camera(self.intrinsic.detach(),
                      extrinsic=None,
                      viewport=self.viewport.detach(),
                      rotation=self.rotation.detach(),
                      translation=self.translation.detach(),
                      width=self.width,
                      height=self.height)
    def __repr__(self):
        return (
            f"Camera(count={self.intrinsic.size(0)})"
        )


