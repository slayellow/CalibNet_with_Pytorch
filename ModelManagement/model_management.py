import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
import UtilityManagement.config as cf
from UtilityManagement.pytorch_util import *
import tensorflow as tf
import emd_cuda

gpu_check = is_gpu_avaliable()
devices = torch.device("cuda") if gpu_check else torch.device("cpu")

IMG_WIDTH = cf.camera_info["WIDTH"]
IMG_HEIGHT = cf.camera_info['HEIGHT']


class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


def earth_mover_distance(xyz1, xyz2, transpose=True):
    """Earth Mover Distance (Approx)

    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n1)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.

    Returns:
        cost (torch.Tensor): (b)

    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)

    loss = torch.sum(cost).to(devices)
    return loss


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
        else:
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
        rx = rotation[0];
        ry = rotation[1];
        rz = rotation[2]

        if order == 'ZYX':
            matR = torch.Tensor([torch.cos(rz) * torch.cos(ry),
                                 torch.cos(rz) * torch.sin(ry) * torch.sin(rx) - torch.sin(rz) * torch.cos(rx),
                                 torch.cos(rz) * torch.sin(ry) * torch.cos(rx) + torch.sin(rz) * torch.sin(rx),
                                 torch.sin(rz) * torch.cos(ry),
                                 torch.sin(rz) * torch.sin(ry) * torch.sin(rx) + torch.cos(rz) * torch.cos(rx),
                                 torch.sin(rz) * torch.sin(ry) * torch.cos(rx) - torch.cos(rz) * torch.sin(rx),
                                 -(torch.sin(ry)),
                                 torch.cos(ry) * torch.sin(rx),
                                 torch.cos(ry) * torch.cos(rx)])
            matR = torch.reshape(matR, [3, 3])

        t = -torch.matmul((torch.transpose(matR, 0, 1)), torch.unsqueeze(translation, 1))

        T = torch.cat([matR, t], 1)

        output.append(T.tolist())

    print("Output Tensor Size : ", torch.Tensor(output).shape)
    return torch.Tensor(output)


def sparsify_cloud_tensorflow(S):
    point_limit = 4096
    no_points = tf.shape(S)[0]
    no_partitions = no_points / tf.constant(point_limit, dtype=tf.int32)
    a = [tf.expand_dims(tf.range(0, tf.cast(no_partitions, dtype=tf.int32) * point_limit), 1)]
    saved_points = tf.gather_nd(S, a)
    saved_points = tf.reshape(saved_points, [point_limit, no_partitions, 3])
    saved_points_sparse = tf.reduce_mean(saved_points, 1)

    return saved_points_sparse


def sparsify_cloud(S):
    point_limit = 4096
    no_points = torch.tensor(S.shape[0])
    no_partitions = (no_points / torch.tensor(point_limit, dtype=torch.int32)).type(torch.int32)
    a = torch.unsqueeze(torch.unsqueeze(torch.arange(0, no_partitions * point_limit), 1), 0)

    ##### Use Tensorflow
    saved_points = torch.from_numpy(tf.gather_nd(tf.Variable(S.detach().cpu()), tf.Variable(a.detach().cpu())).numpy())

    saved_points = torch.reshape(saved_points, [point_limit, no_partitions.type(torch.int32), 3])
    saved_points_sparse = torch.mean(saved_points, 1)

    return saved_points_sparse


def get_transformed_matrix(depth_map, RTMatrix, KMatrix, small_transform):
    output_depth_map = []
    output_sparse_cloud = []
    for i in range(depth_map.shape[0]):
        idx_depth_map = depth_map[i]
        idx_RTMatrix = RTMatrix[i]

        batch_grids, transformed_depth_map, sparse_cloud = get_3D_meshgrid_batchwise_diff(IMG_HEIGHT, IMG_WIDTH,
                                                                                          idx_depth_map[0, :, :],
                                                                                          idx_RTMatrix,
                                                                                          KMatrix, small_transform)

        x_all = torch.reshape(batch_grids[:, 0], (IMG_HEIGHT, IMG_WIDTH))
        y_all = torch.reshape(batch_grids[:, 1], (IMG_HEIGHT, IMG_WIDTH))

        bilinear_sampling_depth_map = get_bilinear_sampling_depth_map(transformed_depth_map, x_all, y_all)

        output_depth_map.append(bilinear_sampling_depth_map.detach().cpu().numpy())
        output_sparse_cloud.append(sparse_cloud.detach().cpu().numpy())

    return torch.Tensor(output_depth_map), torch.Tensor(output_sparse_cloud)


def get_3D_meshgrid_batchwise_diff(height, width, depth_map, RTMatrix, KMatrix, small_transform):
    # 1D-Space Value
    x_index = torch.linspace(-1.0, 1.0, width)
    y_index = torch.linspace(-1.0, 1.0, height)
    z_index = torch.arange(0, width * height).to(devices)

    x_t, y_t = torch.meshgrid(x_index, y_index)
    x_t = torch.transpose(x_t, 0, 1)
    y_t = torch.transpose(y_t, 0, 1)

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
    sampling_grid_2d_sparse = torch.cat([x_t_flat_sparse, y_t_flat_sparse, ones_sparse], 0).to(devices)

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
    point_cloud = torch.stack([x_dash_pred, y_dash_pred, Z], 1)

    sparse_point_cloud = sparsify_cloud(point_cloud)

    x = (points_2d[0, :] / Z)
    y = (points_2d[1, :] / Z)

    mask_int = mask.type(torch.int32)

    update_indices = torch.unsqueeze(torch.masked_select(mask_int * z_index, mask), 1)

    updated_Z = torch.squeeze(
        torch.zeros(width * height, 1, dtype=torch.float32).to(devices).scatter_(0, update_indices, torch.unsqueeze(Z, 1)))
    updated_x = torch.squeeze(
        torch.zeros(width * height, 1, dtype=torch.float32).to(devices).scatter_(0, update_indices, torch.unsqueeze(x, 1)))
    neg_ones = torch.ones_like(updated_x) * -1.0
    updated_x_fin = torch.where(torch.eq(updated_Z, zeros_target), neg_ones, updated_x)
    updated_y = torch.squeeze(
        torch.zeros(width * height, 1, dtype=torch.float32).to(devices).scatter_(0, update_indices, torch.unsqueeze(y, 1)))
    updated_y_fin = torch.where(torch.eq(updated_Z, zeros_target), neg_ones, updated_y)

    reprojected_grid = torch.stack([updated_x_fin, updated_y_fin], 1)

    transformed_depth = torch.reshape(updated_Z, (height, width))

    return reprojected_grid, transformed_depth, sparse_point_cloud


def reverse_all(z):
    z = z.type(torch.float32)
    w = torch.floor((torch.sqrt(8. * z + 1.) - 1.) / 2.0)
    t = (w ** 2 + w) / 2.0
    y = torch.clamp(torch.unsqueeze(z - t, 1), 0.0, IMG_HEIGHT - 1)
    x = torch.clamp(torch.unsqueeze(w - y[:, 0], 1), 0.0, IMG_WIDTH - 1)
    return torch.cat([y, x], 1)


def get_pixel_value(img, x, y):
    indices = torch.stack([y, x], 2)
    indices = torch.reshape(indices, (IMG_HEIGHT * IMG_WIDTH, 2))
    values = torch.reshape(img, [-1])

    Y = indices[:, 0].type(torch.float32)
    X = indices[:, 1].type(torch.float32)
    Z = (X + Y) * (X + Y + 1) / 2 + Y

    ##### Use Tensorflow
    filtered, idx = tf.unique(tf.squeeze(Z.detach().cpu()))
    filtered, idx = torch.from_numpy(filtered.numpy()), torch.from_numpy(idx.numpy())
    updated_values = torch.from_numpy(tf.compat.v1.unsorted_segment_max(values.detach().cpu(), idx, filtered.shape[0]).numpy())

    updated_indices = reverse_all(filtered)
    updated_indices = updated_indices.type(torch.int32)

    ##### Use Tensorflow
    resolved_map = torch.from_numpy(tf.scatter_nd(updated_indices, updated_values, img.shape).numpy())

    return resolved_map.to(devices)


def get_bilinear_sampling_depth_map(depth_map, x_func, y_func):
    max_y = torch.tensor(IMG_HEIGHT - 1, dtype=torch.int32)
    max_x = torch.tensor(IMG_WIDTH - 1, dtype=torch.int32)

    x = 0.5 * ((x_func + 1.0) * torch.tensor(IMG_WIDTH - 1, dtype=torch.float32))
    y = 0.5 * ((y_func + 1.0) * torch.tensor(IMG_HEIGHT - 1, dtype=torch.float32))

    x = torch.clamp(x, 0.0, max_x.type(torch.float32))
    y = torch.clamp(y, 0.0, max_y.type(torch.float32))

    x0 = torch.floor(x).type(torch.int32)
    x1 = x0 + 1
    y0 = torch.floor(y).type(torch.int32)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)
    y0 = torch.clamp(y0, 0, max_y)
    y1 = torch.clamp(y1, 0, max_y)

    Ia = get_pixel_value(depth_map, x0, y0)
    Ib = get_pixel_value(depth_map, x0, y1)
    Ic = get_pixel_value(depth_map, x1, y0)
    Id = get_pixel_value(depth_map, x1, y1)

    x0 = x0.type(torch.float32)
    x1 = x1.type(torch.float32)
    y0 = y0.type(torch.float32)
    y1 = y1.type(torch.float32)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    loc = wa * Ia + wb * Ib + wc * Ic + wd * Id

    #
    # zeors_target_numpy =  loc.numpy()
    # zeors_target_tensorflow_numpy = loc_tensorflow.numpy()
    # torch_list = []
    # tensorflow_list = []
    # count = 0
    # for i in range(zeors_target_numpy.shape[0]):
    #     for j in range(zeors_target_numpy.shape[1]):
    #         print("Torch : ", round(zeors_target_numpy[i][j], 4 ), " || Tensorflow : ", round(zeors_target_tensorflow_numpy[i][j], 4))
    #         if round(zeors_target_numpy[i][j], 4 ) == round(zeors_target_tensorflow_numpy[i][j], 4):
    #             continue
    #         else:
    #             torch_list.append(round(zeors_target_numpy[i][j], 4 ))
    #             tensorflow_list.append(round(zeors_target_tensorflow_numpy[i][j], 4))
    #             count = count + 1
    #
    # for i, j in zip(torch_list, tensorflow_list):
    #     print("Diff : ", i, j, abs(round(i-j, 3)))
    # print("Another Point Position : ", count)
    # exit(0)

    return loc
