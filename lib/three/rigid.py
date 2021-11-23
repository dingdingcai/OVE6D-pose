from typing import Tuple

import torch
from torch.nn import functional as F

from lib.three import core


def intrinsic_to_3x4(matrix):
    matrix, unsqueezed = core.ensure_batch_dim(matrix, num_dims=2)

    zeros = torch.zeros(1, 3, 1, dtype=matrix.dtype).expand(matrix.shape[0], -1, -1).to(matrix.device)
    mat = torch.cat((matrix, zeros), dim=-1)

    if unsqueezed:
        mat = mat.squeeze(0)

    return mat


@torch.jit.script
def RT_to_matrix(R, T):
    R, unsqueezed = core.ensure_batch_dim(R, num_dims=2)
    if R.shape[-1] == 3:
        R = F.pad(R, (0, 1, 0, 1)) # 4 x 4
    if R.dim() == 2:
        R = R[None, ...]
    if T.dim() == 1:
        T = T[None, ...]
    R[:, :3, 3] = T
    R[:, -1, -1] = 1.0
    if unsqueezed:
        R = R.squeeze(0)
    return R
    

@torch.jit.script
def matrix_3x3_to_4x4(matrix):
    matrix, unsqueezed = core.ensure_batch_dim(matrix, num_dims=2)

    mat = F.pad(matrix, [0, 1, 0, 1])
    mat[:, -1, -1] = 1.0

    if unsqueezed:
        mat = mat.squeeze(0)

    return mat


@torch.jit.script
def rotation_to_4x4(matrix):
    return matrix_3x3_to_4x4(matrix)


@torch.jit.script
def translation_to_4x4(translation):
    translation, unsqueezed = core.ensure_batch_dim(translation, num_dims=1)

    eye = torch.eye(4, device=translation.device)
    mat = F.pad(translation.unsqueeze(2), [3, 0, 0, 1]) + eye

    if unsqueezed:
        mat = mat.squeeze(0)

    return mat


def translate_matrix(matrix, offset):
    matrix, unsqueezed = core.ensure_batch_dim(matrix, num_dims=2)

    out = inverse_transform(matrix)
    out[:, :3, 3] += offset
    out = inverse_transform(out)

    if unsqueezed:
        out = out.squeeze(0)

    return out


def scale_matrix(matrix, scale):
    matrix, unsqueezed = core.ensure_batch_dim(matrix, num_dims=2)

    out = inverse_transform(matrix)
    out[:, :3, 3] *= scale
    out = inverse_transform(out)

    if unsqueezed:
        out = out.squeeze(0)

    return out


def decompose(matrix):
    matrix, unsqueezed = core.ensure_batch_dim(matrix, num_dims=2)

    # Extract rotation matrix.
    origin = (torch.tensor([0.0, 0.0, 0.0, 1.0], device=matrix.device, dtype=matrix.dtype)
              .unsqueeze(1)
              .unsqueeze(0))
    origin = origin.expand(matrix.size(0), -1, -1)
    R = torch.cat((matrix[:, :, :3], origin), dim=-1)

    # Extract translation matrix.
    eye = torch.eye(4, 3, device=matrix.device).unsqueeze(0).expand(matrix.size(0), -1, -1)
    T = torch.cat((eye, matrix[:, :, 3].unsqueeze(-1)), dim=-1)

    if unsqueezed:
        R = R.squeeze(0)
        T = T.squeeze(0)

    return R, T


def inverse_transform(matrix):
    matrix, unsqueezed = core.ensure_batch_dim(matrix, num_dims=2)

    R, T = decompose(matrix)
    R_inv = R.transpose(1, 2)
    t = T[:, :4, 3].unsqueeze(2)
    t_inv = (R_inv @ t)[:, :3].squeeze(2)

    out = torch.zeros_like(matrix)
    out[:, :3, :3] = R_inv[:, :3, :3]
    out[:, :3, 3] = -t_inv
    out[:, 3, 3] = 1

    if unsqueezed:
        out = out.squeeze(0)

    return out


def extrinsic_to_position(extrinsic):
    extrinsic, unsqueezed = core.ensure_batch_dim(extrinsic, num_dims=2)

    rot_mat, trans_mat = decompose(extrinsic)
    position = rot_mat.transpose(2, 1) @ trans_mat[:, :, 3, None]
    position = core.dehomogenize(position.squeeze(-1))

    if unsqueezed:
        position = position.squeeze(0)
    return position


@torch.jit.script
def random_translation(n: int,
                       x_bound: Tuple[float, float],
                       y_bound: Tuple[float, float],
                       z_bound: Tuple[float, float]):
    trans_x = core.uniform(n, *x_bound)
    trans_y = core.uniform(n, *y_bound)
    trans_z = core.uniform(n, *z_bound)
    translation = torch.stack((trans_x, trans_y, trans_z), dim=-1)
    return translation
