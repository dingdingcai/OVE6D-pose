"""
This code is partially borrowed from LatentFusion
"""

import os
import math
import torch
import trimesh
import pyrender
import numpy as np
import torch.nn.functional as F
from pyrender import RenderFlags
from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix

from utility import meshutils

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def uniform_z_rotation(n, eps_degree=0):
    """
    uniformly sample N examples range from 0 to 360
    """
    assert n > 0, "sample number must be nonzero"
    eps_rad = eps_degree / 180.0 * math.pi
    x_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps
    y_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps
    z_radians = (torch.arange(n) + 1)/(n + 1) * math.pi * 2
    target_euler_radians = torch.stack([x_radians, y_radians, z_radians], dim=-1)
    target_rotation_matrix = euler_angles_to_matrix(target_euler_radians, "XYZ")
    return target_rotation_matrix

def uniform_xy_rotation(n, eps_degree=0):
    """
    uniformly sample N examples range from 0 to 360
    """
    assert n > 0, "sample number must be nonzero"
    target_rotation_matrix = random_xyz_rotation(1) @ evenly_distributed_rotation(n)
    return target_rotation_matrix

def random_z_rotation(n, eps_degree=0):
    """
    randomly sample N examples range from 0 to 360
    """
    eps_rad = eps_degree / 180. * math.pi
    x_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps
    y_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps
    z_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * math.pi # -pi, pi
    target_euler_radians = torch.stack([x_radians, y_radians, z_radians], dim=-1)
    target_euler_matrix = euler_angles_to_matrix(target_euler_radians, "XYZ")
    return target_euler_matrix 

def random_xy_rotation(n, eps_degree=0, rang_degree=180):
    """
    randomly sample N examples range from 0 to 360
    """
    eps_rad = eps_degree / 180. * math.pi
    rang_rad = rang_degree / 180 * math.pi
    x_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * rang_rad # -pi, pi
    y_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * rang_rad # -pi, pi
    
    z_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps

    target_euler_radians = torch.stack([x_radians, y_radians, z_radians], dim=-1)
    target_euler_matrix = euler_angles_to_matrix(target_euler_radians, "XYZ")
    return target_euler_matrix 

def random_xyz_rotation(n, eps_degree=180):
    """
    randomly sample N examples range from 0 to 360 
    """
    eps_rad = eps_degree / 180. * math.pi
    x_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -pi, pi
    y_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -pi, pi
    z_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps

    target_euler_radians = torch.stack([x_radians, y_radians, z_radians], dim=-1)
    target_euler_matrix = euler_angles_to_matrix(target_euler_radians, "XYZ")
    return target_euler_matrix 

def evenly_distributed_rotation(n, random_seed=None):
    """
    uniformly sample N examples on a sphere
    """
    def normalize(vector, dim: int = -1):
        return vector / torch.norm(vector, p=2.0, dim=dim, keepdim=True)

    if random_seed is not None:
        torch.manual_seed(random_seed) # fix the sampling of viewpoints for reproducing evaluation

    indices = torch.arange(0, n, dtype=torch.float32) + 0.5

    phi = torch.acos(1 - 2 * indices / n)
    theta = math.pi * (1 + 5 ** 0.5) * indices
    points = torch.stack([
        torch.cos(theta) * torch.sin(phi), 
        torch.sin(theta) * torch.sin(phi), 
        torch.cos(phi),], dim=1)
    forward = -points
    
    down = normalize(torch.randn(n, 3), dim=1)
    right = normalize(torch.cross(down, forward))
    down = normalize(torch.cross(forward, right))
    R_mat = torch.stack([right, down, forward], dim=1)
    return R_mat

def load_object(path, scale=1.0, size=1.0, recenter=True, resize=True,
                bound_type='diameter', load_materials=False) -> meshutils.Object3D:
    """
    Loads an object model as an Object3D instance.

    Args:
        path: the path to the 3D model
        scale: a scaling factor to apply after all transformations
        size: the reference 'size' of the object if `resize` is True
        recenter: if True the object will be recentered at the centroid
        resize: if True the object will be resized to fit insize a cube of size `size`
        bound_type: how to compute size for resizing. Either 'diameter' or 'extents'

    Returns:
        (meshutils.Object3D): the loaded object model
    """
    obj = meshutils.Object3D(path, load_materials=load_materials)

    if recenter:
        obj.recenter('bounds')

    if resize:
        if bound_type == 'diameter':
            object_scale = size / obj.bounding_diameter
        elif bound_type == 'extents':
            object_scale = size / obj.bounding_size
        else:
            raise ValueError(f"Unkown size_type {bound_type!r}")

        obj.rescale(object_scale)
    else:
        object_scale = 1.0

    if scale != 1.0:
        obj.rescale(scale)

    return obj, obj.bounding_diameter

def _create_object_node(obj: meshutils.Object3D):
    smooth = True
    # Turn smooth shading off if vertex normals are unreliable.
    if obj.are_normals_corrupt():
        smooth = False

    mesh = pyrender.Mesh.from_trimesh(obj.meshes, smooth=smooth)
    node = pyrender.Node(mesh=mesh)

    return node


class SceneContext(object):
    """
    A wrapper class containing all contextual information needed for rendering.
    """

    def __init__(self, obj, intrinsic: torch.Tensor):
        self.obj = obj
        self.intrinsic = intrinsic.squeeze()
        self.extrinsic = None
        self.scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(0.1, 0.1, 0.1))
        
        fx = self.intrinsic[0, 0].item()
        fy = self.intrinsic[1, 1].item()
        cx = self.intrinsic[0, 2].item()
        cy = self.intrinsic[1, 2].item()

        self.camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        self.camera_node = self.scene.add(self.camera, name='camera')
        self.object_node = _create_object_node(self.obj)

        self.scene.add_node(self.object_node)

    def object_to_camera_pose(self, object_pose):
        """
        Take an object pose and converts it to a camera pose.

        Takes a matrix that transforms object-space points to camera-space points and converts it
        to a matrix that takes OpenGL camera-space points and converts it into object-space points.
        """
        CAM_REF_POSE = torch.tensor((
            (1, 0, 0, 0),
            (0, -1, 0, 0),
            (0, 0, -1, 0),
            (0, 0, 0, 1),
        ), dtype=torch.float32)
        
        camera_transform = self.inverse_transform(object_pose)

        # We must flip the z-axis before performing our transformation so that the z-direction is
        # pointing in the correct direction when we feed this as OpenGL coordinates.
        return CAM_REF_POSE.t()[None, ...] @ camera_transform @ CAM_REF_POSE[None, ...]

    def set_pose(self, translation, rotation):
        extrinsic = self.RT_to_matrix(R=rotation, T=translation)
        self.extrinsic = extrinsic
        camera_pose = self.object_to_camera_pose(extrinsic).squeeze().numpy()
        assert len(camera_pose.shape) == 2, 'camera pose for pyrender must be 4 x 4'
        self.scene.set_pose(self.camera_node, camera_pose)
          
    def inverse_transform(self, matrix):
        if matrix.dim() == 2:
            matrix = matrix[None, ...]
        R = matrix[:, :3, :3] # B x 3 x 3
        T = matrix[:, :3, 3:4] # B x 3 x 1
        R_inv = R.transpose(-2, -1) # B x 3 x 3
        t_inv = (R_inv @ T).squeeze(2)# B x 3

        out = torch.zeros_like(matrix)
        out[:, :3, :3] = R_inv[:, :3, :3]
        out[:, :3, 3] = -t_inv
        out[:, 3, 3] = 1
        return out

    def RT_to_matrix(self, R, T):
        if R.shape[-1] == 3:
            R = F.pad(R, (0, 1, 0, 1)) # 4 x 4
        if R.dim() == 2:
            R = R[None, ...]
        if T.dim() == 1:
            T = T[None, ...]
        R[:, :3, 3] = T
        R[:, -1, -1] = 1.0
        return R


class Renderer(object):
    """
    A thin wrapper around the PyRender renderer.
    """
    def __init__(self, width, height):
        self._renderer = pyrender.OffscreenRenderer(width, height)
        self._render_flags = RenderFlags.SKIP_CULL_FACES | RenderFlags.RGBA

    @property
    def width(self):
        return self._renderer.viewport_width

    @property
    def height(self):
        return self._renderer.viewport_height

    def __del__(self):
        self._renderer.delete()

    def render(self, context):
        color, depth = self._renderer.render(context.scene, flags=self._render_flags)
        color = color.copy().astype(np.float32) / 255.0
        color = torch.tensor(color)
        depth = torch.tensor(depth)
        # mask = color[..., 3]
        mask = (depth > 0).float()
        color = color[..., :3]
        return color, depth, mask
    

def rendering_views(obj_mesh, intrinsic, R, T, height=540, width=720):
    obj_scene = SceneContext(obj=obj_mesh, intrinsic=intrinsic)  # define a scene
    obj_renderer = Renderer(width=width, height=height)  # define a renderer
    obj_depths = list()
    obj_masks = list()
    if R.dim() == 2:
        R = R[None, ...]
    if T.dim() == 1:
        T = T[None, ...]
    for anc_R, anc_T in zip(R, T):    
        obj_scene.set_pose(rotation=anc_R, translation=anc_T)
        color, depth, mask = obj_renderer.render(obj_scene)
        obj_depths.append(depth)
        obj_masks.append(mask)
    del obj_scene
    obj_depths = torch.stack(obj_depths, dim=0).unsqueeze(1)
    obj_masks = torch.stack(obj_masks, dim=0).unsqueeze(1)
    return obj_depths, obj_masks

def render_uniform_sampling_views(model_path, intrinsic, scale=1.0, num_views=1000, dist=0.8, height=540, width=720):
    obj, obj_scale = load_object(model_path, resize=False, recenter=False)
    obj.rescale(scale=scale) # from millimeter normalize to meter
    obj_scene = SceneContext(obj=obj, intrinsic=intrinsic)  # define a scene
    obj_renderer = Renderer(width=width, height=height)  # define a renderer

    obj_R = evenly_distributed_rotation(n=num_views) # uniform rotational views sampling from a shpere, N x 3 x 3
    obj_T = torch.zeros_like(obj_R[:, :, 0]) # constant distance, N x 3
    obj_T[:, -1] = dist
    
    obj_diameter = (((obj.vertices.max(0) - obj.vertices.min(0))**2).sum())**0.5
    obj_T = obj_T * obj_diameter # scaling according to specific object size
    
    obj_depths = list()
    obj_masks = list()
    
    for anc_R, anc_T in zip(obj_R, obj_T):    
        obj_scene.set_pose(rotation=anc_R, translation=anc_T)
        color, depth, mask = obj_renderer.render(obj_scene)
        obj_depths.append(depth)
        obj_masks.append(mask)
    obj_depths = torch.stack(obj_depths, dim=0).unsqueeze(1)
    obj_masks = torch.stack(obj_masks, dim=0).unsqueeze(1)
    del obj_scene
    # del obj_renderer
    return obj_depths, obj_masks, obj_R, obj_T

def render_RT_views(model_path, intrinsic, R, T, scale=1.0, height=540, width=720):
    obj_mesh, obj_scale = load_object(model_path, resize=False, recenter=False)
    obj_mesh.rescale(scale=scale) # from millimeter normalize to meter
    obj_scene = SceneContext(obj=obj_mesh, intrinsic=intrinsic)  # define a scene
    obj_renderer = Renderer(width=width, height=height)  # define a renderer
    obj_depths = list()
    obj_masks = list()
    if R.dim() == 2:
        R = R[None, ...]
        T = T[None, ...]
    for anc_R, anc_T in zip(R, T):    
        obj_scene.set_pose(rotation=anc_R, translation=anc_T)
        color, depth, mask = obj_renderer.render(obj_scene)
        obj_depths.append(depth)
        obj_masks.append(mask)
    del obj_scene
    # del obj_renderer
    obj_depths = torch.stack(obj_depths, dim=0).unsqueeze(1)
    obj_masks = torch.stack(obj_masks, dim=0).unsqueeze(1)
    return obj_depths, obj_masks

def render_single_view(model_path, intrinsic, R, T, scale=1.0, height=540, width=720):
    assert R.dim() == 2 and T.dim() == 1, "pyrender R and T shape " + R.shape
    obj, obj_scale = load_object(model_path, resize=False, recenter=False)
    obj.rescale(scale=scale) # from millimeter normalize to meter
    obj_scene = SceneContext(obj=obj, intrinsic=intrinsic)  # define a scene
    obj_renderer = Renderer(width=width, height=height)  # define a renderer
    obj_scene.set_pose(rotation=R, translation=T)
    color, depth, mask = obj_renderer.render(obj_scene)
    del obj_scene
    # del obj_renderer
    return depth, mask

def render_sampling_pair_views(mesh_file, intrinsic, num_views=1000, dist=0.8, height=540, width=720, dist_jitter=0.2):
    obj_trimesh = trimesh.load(mesh_file)
    obj_trimesh.vertices = obj_trimesh.vertices / 1000.0
    # obj_trimesh.vertices = obj_trimesh.vertices - obj_trimesh.vertices.mean(0)
    
    obj_mesh = pyrender.Mesh.from_trimesh(obj_trimesh)
    
    obj_scene = SceneContext(mesh=obj_mesh, intrinsic=intrinsic)
    obj_renderer = Renderer(width=width, height=height)

    Rxy = random_xy_rotation(num_views, eps_degree=2)
    Rz = random_z_rotation(num_views, eps_degree=2)
    camera_T = torch.tensor([0.0, 0.0, dist], dtype=torch.float32).repreat(num_views, 1)
    camera_T = camera_T + (torch.rand_like(camera_T) - 0.5) * dist_jitter
    
    diameter = (((obj_trimesh.vertices.max(0)[0] - obj_trimesh.vertices.min(0)[0])**2).sum())**0.5
    camera_T = camera_T.clone() * diameter
    
    obj_Rxyz_depths = list()
    obj_Rxyz_masks = list()
    obj_Rxy_depths = list()
    obj_Rxy_masks = list()

    for anc_R, anc_T in zip(Rxy, camera_T):    
        obj_scene.set_pose(rotation=anc_R, translation=anc_T)
        color, depth, mask = obj_renderer.render(obj_scene)
        obj_Rxy_depths.append(depth)
        obj_Rxy_masks.append(mask)
     
    obj_Rxy_depths = torch.stack(obj_Rxy_depths, dim=0)
    obj_Rxy_masks = torch.stack(obj_Rxy_masks, dim=0)
    
    Rxyz = Rz @ Rxy
    for anc_R, anc_T in zip(Rxyz, camera_T):    
        obj_scene.set_pose(rotation=anc_R, translation=anc_T)
        color, depth, mask = obj_renderer.render(obj_scene)
        obj_Rxyz_depths.append(depth)
        obj_Rxyz_masks.append(mask)
     
    obj_Rxyz_depths = torch.stack(obj_Rxyz_depths, dim=0)
    obj_Rxyz_masks = torch.stack(obj_Rxyz_masks, dim=0)
    
    
    # del obj_renderer
    return obj_Rxy_depths, obj_Rxyz_depths, obj_Rxy_masks, obj_Rxyz_masks, Rxy, Rxyz, camera_T, Rz