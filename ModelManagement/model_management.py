import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

import UtilityManagement.config as cf
from UtilityManagement.pytorch_util import *
import tensorflow as tf


gpu_check = is_gpu_avaliable()
devices = torch.device("cuda") if gpu_check else torch.device("cpu")

IMG_WIDTH = cf.camera_info["WIDTH"]
IMG_HEIGHT = cf.camera_info['HEIGHT']


# Get RT Matrix using Exponential Map
def get_RTMatrix_using_exponential_logarithm_mapping(se_vector):

    output = []
    for i in range(se_vector.shape[0]):
        vector = se_vector[i]
        v = vector[:3]
        w = vector[3:]

        theta = torch.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])

        w_cross = torch.Tensor([0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0]).to(devices)
        w_cross = torch.reshape(w_cross, [3, 3])

        thetaa = theta.detach()
        if thetaa == 0:
            print("Theta is 0", thetaa)
            A = 0
            B = 0
            C = 0
        else :
            A = torch.sin(theta) / theta
            B = (1.0 - torch.cos(theta)) / (torch.pow(theta, 2))
            C = (1.0 - A) / (torch.pow(theta, 2))

        w_cross_square = torch.matmul(w_cross, w_cross)

        R = torch.eye(3).to(devices) + A * w_cross + B * w_cross_square
        Q = torch.eye(3).to(devices) + B * w_cross + C * w_cross_square

        t = torch.matmul(Q, torch.unsqueeze(v, 1))

        T = torch.cat([R, t], 1)

        output.append(T.tolist())

    return torch.Tensor(output)


# Get RT Matrix using Euler Angles
def get_RTMatrix_using_EulerAngles(se_vector, order='ZYX'):

    print("Input Tensor Size : ", se_vector.shape)

    output = []
    for i in range(se_vector.shape[0]):
        vector = se_vector[i][0]
        translation = vector[:3]
        rotation = vector[3:]
        rx = rotation[0]; ry = rotation[1]; rz = rotation[2]

        if order == 'ZYX':
            matR = torch.Tensor([ torch.cos(rz)*torch.cos(ry),
                                  torch.cos(rz)*torch.sin(ry)*torch.sin(rx) - torch.sin(rz)*torch.cos(rx),
                                  torch.cos(rz)*torch.sin(ry)*torch.cos(rx) + torch.sin(rz)*torch.sin(rx),
                                  torch.sin(rz)*torch.cos(ry),
                                  torch.sin(rz)*torch.sin(ry)*torch.sin(rx) + torch.cos(rz)*torch.cos(rx),
                                  torch.sin(rz)*torch.sin(ry)*torch.cos(rx) - torch.cos(rz)*torch.sin(rx),
                                  -(torch.sin(ry)),
                                  torch.cos(ry)*torch.sin(rx),
                                  torch.cos(ry)*torch.cos(rx)])
            matR = torch.reshape(matR, [3, 3])

        t = -torch.matmul((torch.transpose(matR, 0, 1)), torch.unsqueeze(translation, 1))

        T = torch.cat([matR, t], 1)

        output.append(T.tolist())

    print("Output Tensor Size : ", torch.Tensor(output).shape)
    return torch.Tensor(output)


def get_transformed_matrix(depth_map, RTMatrix, KMatrix, small_transform):

    output_depth_map = []
    for i in range(depth_map.shape[0]):
        idx_depth_map = depth_map[i]
        idx_RTMatrix = RTMatrix[i]
        batch_grids, transformed_depth_map = get_3D_meshgrid_batchwise_diff_tensorflow(IMG_HEIGHT, IMG_WIDTH, idx_depth_map[0, :, :], idx_RTMatrix,
                                       KMatrix, small_transform)

        x_all = tf.reshape(batch_grids[:, 0], (IMG_HEIGHT, IMG_WIDTH))
        y_all = tf.reshape(batch_grids[:, 1], (IMG_HEIGHT, IMG_WIDTH))

        bilinear_sampling_depth_map = get_bilinear_sampling_depth_map(transformed_depth_map, x_all, y_all)

        output_depth_map.append(bilinear_sampling_depth_map.numpy())

    return torch.Tensor(output_depth_map)


def get_3D_meshgrid_batchwise_diff(height, width, depth_map, RTMatrix, KMatrix, small_transform):

    # 1D-Space Value
    x_index = torch.linspace(-1.0, 1.0, width)
    y_index = torch.linspace(-1.0, 1.0, height)
    z_index = torch.arange(0, width * height)

    x_t, y_t = torch.meshgrid(x_index, y_index)

    # flatten ( 2D-Space Value )
    x_t_flat = torch.reshape(x_t, [1, -1])
    y_t_flat = torch.reshape(y_t, [1, -1])
    ZZ = torch.reshape(depth_map, [-1])

    # 0이 아닌 값들이 Point Cloud가 속해 있는 값이므로 mask를 씌어 해당 값들만 살린다.
    zeros_target = torch.zeros_like(ZZ)
    mask = torch.not_equal(ZZ, zeros_target)
    ones = torch.ones_like(x_t_flat)
    mask_flat = torch.unsqueeze(mask, 1)

    x_t_flat_sparse = torch.unsqueeze(torch.masked_select(torch.transpose(x_t_flat, 0, 1), mask_flat), 0)
    y_t_flat_sparse = torch.unsqueeze(torch.masked_select(torch.transpose(y_t_flat, 0, 1), mask_flat), 0)
    ones_sparse = torch.unsqueeze(torch.masked_select(torch.transpose(ones, 0, 1), mask_flat), 0)

    # 0이 아닌 값들로 이루어진 Sparse한 좌표 값
    sampling_grid_2d_sparse = torch.cat([x_t_flat_sparse, y_t_flat_sparse, ones_sparse], 0)

    # 실제 Depth Map에 해당하는 값을 찾는다.
    ZZ_saved = torch.masked_select(ZZ, mask)
    ones_saved = torch.unsqueeze(torch.ones_like(ZZ_saved), 0)

    # Sparse한 좌표 값에 Depth 값을 곱하고 카메라 Intrinsic Parameter의 역행렬 값을 곱하여 실제 Point Cloud 좌표를 도출한다.
    projection_grid_3d = torch.matmul(torch.inverse(KMatrix), sampling_grid_2d_sparse * ZZ_saved)

    # 3D Point Cloud 를 Homogeneous Coordinate에 표현
    homog_point_3d = torch.cat([projection_grid_3d, ones_saved], 0)

    # Random Transformation 시 진행한 RTMatrix, cam_translation의 반대로 cam_translation_inv와 예측한 RTMatrix 곱
    final_transformation_matrix = torch.matmul(RTMatrix, small_transform)[:3, :]
    warped_sampling_grid = torch.matmul(final_transformation_matrix, homog_point_3d)

    # Homogeneous Coordinate 상의 점과 곱하여 3차원 결과 도출
    # Camera Intrinsic Parameter를 곱하여 이미지에 매핑 가능한 3차원 좌표 도출
    points_2d = torch.matmul(KMatrix, warped_sampling_grid[:3, :])

    Z = points_2d[2, :]

    x_dash_pred = points_2d[0, :]
    y_dash_pred = points_2d[1, :]

    x = (points_2d[0, :]/Z)
    y = (points_2d[1, :]/Z)

    mask_int = mask.type(torch.int32)

    update_indices = torch.unsqueeze(torch.masked_select(mask_int*z_index, mask), 1)

    # Use Tensorflow Function
    updated_Z = torch.from_numpy(tf.scatter_nd(update_indices, Z, tf.constant([width*height])).numpy())
    updated_x = torch.from_numpy(tf.scatter_nd(update_indices, x, tf.constant([width*height])).numpy())
    neg_ones = torch.ones_like(updated_x) * -1.0
    updated_x_fin = torch.where(torch.eq(updated_Z, zeros_target), neg_ones, updated_x)
    updated_y = torch.from_numpy(tf.scatter_nd(update_indices, y, tf.constant([width*height])).numpy())
    updated_y_fin = torch.where(torch.eq(updated_Z, zeros_target), neg_ones, updated_y)

    reprojected_grid = torch.stack([updated_x_fin, updated_y_fin], 1)

    transformed_depth = torch.reshape(updated_Z, (height, width))

    return reprojected_grid, transformed_depth



def get_3D_meshgrid_batchwise_diff_tensorflow(height, width, depth_map, RTMatrix, KMatrix, small_transform):

    # 1D-Space Value
    x_index = tf.linspace(-1.0, 1.0, width)
    y_index = tf.linspace(-1.0, 1.0, height)
    z_index = tf.range(0, width * height)

    x_t, y_t = tf.meshgrid(x_index, y_index)

    # flatten ( 2D-Space Value )
    x_t_flat = tf.reshape(x_t, [1, -1])
    y_t_flat = tf.reshape(y_t, [1, -1])
    ZZ = tf.reshape(depth_map.cpu(), [-1])

    # 0이 아닌 값들이 Point Cloud가 속해 있는 값이므로 mask를 씌어 해당 값들만 살린다.
    zeros_target = tf.zeros_like(ZZ)
    mask = tf.not_equal(ZZ, zeros_target)
    ones = tf.ones_like(x_t_flat)

    sampling_grid_2d = tf.concat([x_t_flat, y_t_flat, ones], 0)
    sampling_grid_2d_sparse = tf.transpose(tf.boolean_mask(tf.transpose(sampling_grid_2d), mask))
    ZZ_saved = tf.boolean_mask(ZZ, mask)
    ones_saved = tf.expand_dims(tf.ones_like(ZZ_saved), 0)

    # Sparse한 좌표 값에 Depth 값을 곱하고 카메라 Intrinsic Parameter의 역행렬 값을 곱하여 실제 Point Cloud 좌표를 도출한다.
    projection_grid_3d = tf.matmul(tf.compat.v1.matrix_inverse(KMatrix.cpu()), sampling_grid_2d_sparse * ZZ_saved)

    # 3D Point Cloud 를 Homogeneous Coordinate에 표현
    homog_point_3d = tf.concat([projection_grid_3d, ones_saved], 0)

    # Random Transformation 시 진행한 RTMatrix, cam_translation의 반대로 cam_translation_inv와 예측한 RTMatrix 곱
    final_transformation_matrix = tf.matmul(RTMatrix.cpu(), small_transform.cpu())[:3, :]
    warped_sampling_grid = tf.matmul(final_transformation_matrix, homog_point_3d)

    # Homogeneous Coordinate 상의 점과 곱하여 3차원 결과 도출
    # Camera Intrinsic Parameter를 곱하여 이미지에 매핑 가능한 3차원 좌표 도출
    points_2d = tf.matmul(KMatrix.cpu(), warped_sampling_grid[:3, :])

    Z = points_2d[2, :]

    x_dash_pred = points_2d[0, :]
    y_dash_pred = points_2d[1, :]

    x = tf.transpose(points_2d[0, :]/Z)
    y = tf.transpose(points_2d[1, :]/Z)

    mask_int = tf.cast(mask, 'int32')

    update_indices = tf.expand_dims(tf.boolean_mask(mask_int*z_index, mask), 1)

    # Use Tensorflow Function
    updated_Z = (tf.scatter_nd(update_indices, Z, tf.constant([width*height])))
    updated_x = (tf.scatter_nd(update_indices, x, tf.constant([width*height])))
    neg_ones = tf.ones_like(updated_x) * -1.0
    updated_x_fin = tf.where(tf.equal(updated_Z, zeros_target), neg_ones, updated_x)
    updated_y = (tf.scatter_nd(update_indices, y, tf.constant([width*height])))
    updated_y_fin = tf.where(tf.equal(updated_Z, zeros_target), neg_ones, updated_y)

    reprojected_grid = tf.stack([updated_x_fin, updated_y_fin], 1)

    transformed_depth = tf.reshape(updated_Z, (height, width))

    return torch.from_numpy(reprojected_grid.numpy()), torch.from_numpy(transformed_depth.numpy())


def reverse_all(z):
    z = tf.cast(z, 'float32')
    w = tf.floor((tf.sqrt(8.*z + 1.) - 1.) / 2.0)
    t = (w**2 + w) / 2.0
    y = tf.clip_by_value(tf.expand_dims(z - t, 1), 0.0, IMG_HEIGHT - 1)
    x = tf.clip_by_value(tf.expand_dims(w - y[:, 0], 1), 0.0, IMG_WIDTH - 1)
    return tf.concat([y, x], 1)


def get_pixel_value(img, x, y):
    indices = tf.stack([y, x], 2)
    indices = tf.reshape(indices, (IMG_HEIGHT*IMG_WIDTH, 2))
    values = tf.reshape(img, [-1])

    Y = tf.cast(indices[:, 0], 'float32')
    X = tf.cast(indices[:, 1], 'float32')
    Z = (X + Y) * (X + Y + 1) / 2 + Y

    filtered, idx = tf.unique(tf.squeeze(Z))
    updated_values = tf.compat.v1.unsorted_segment_max(values, idx, filtered.shape[0])

    updated_indices = reverse_all(filtered)
    updated_indices = tf.cast(updated_indices, 'int32')
    resolved_map = torch.from_numpy(tf.scatter_nd(updated_indices, updated_values, img.shape).numpy())

    return resolved_map


def get_bilinear_sampling_depth_map(depth_map, x_func, y_func):

    max_y = tf.constant(IMG_HEIGHT - 1, dtype=tf.int32)
    max_x = tf.constant(IMG_WIDTH - 1, dtype=tf.int32)

    x = 0.5 * ((x_func + 1.0) * tf.cast(IMG_WIDTH - 1, dtype=tf.float32))
    y = 0.5 * ((y_func + 1.0) * tf.cast(IMG_HEIGHT - 1, dtype=tf.float32))

    x = tf.clip_by_value(x, 0.0, tf.cast(max_x, 'float32'))
    y = tf.clip_by_value(y, 0.0, tf.cast(max_y, 'float32'))

    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)

    Ia = get_pixel_value(depth_map, x0, y0)
    Ib = get_pixel_value(depth_map, x0, y1)
    Ic = get_pixel_value(depth_map, x1, y0)
    Id = get_pixel_value(depth_map, x1, y1)

    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    loc = wa*Ia + wb*Ib + wc*Ic + wd*Id

    return loc




