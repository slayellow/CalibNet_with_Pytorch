import numpy as np


# 데이터셋 관련 경로

paths = dict(
    dataset_path = "/home/HONG/CalibNet_with_Pytorch/DataManagement/parsed_set.txt",
    pretrained_path = "/home/HONG/PretrainedParameter",
    training_img_result_path = "/home/HONG/CalibNet_Result",
    validation_img_result_path = "/home/HONG/CalibNet_Result"
)
#
# paths = dict(
#     dataset_path = "/Users/jinseokhong/data/CalibNet_DataSet/parsed_set_example.txt",
#     pretrained_path = "/Users/jinseokhong/data/CalibNet_DataSet",
#     training_img_result_path = "/Users/jinseokhong/data/CalibNet_Result",
#     validation_img_result_path = "/Users/jinseokhong/data/CalibNet_Result"
# )


# 카메라 관련 파라메타

camera_info = dict(
    WIDTH = 1242,
    HEIGHT = 375,

    fx = 7.215377e+02,
    fy = 7.215377e+02,
    cx = 6.095593e+02,
    cy = 1.728540e+02,

    cam_transform_02 = np.array([1.0, 0.0, 0.0, (-4.485728e+01)/7.215377e+02,
                                 0.0, 1.0, 0.0, (-2.163791e-01)/7.215377e+02,
                                 0.0, 0.0, 1.0, (-2.745884e-03),
                                 0.0, 0.0, 0.0, 1.0]).reshape(4, 4),

    cam_transform_02_inv = np.array([1.0, 0.0, 0.0, (4.485728e+01)/7.215377e+02,
                                     0.0, 1.0, 0.0, (2.163791e-01)/7.215377e+02,
                                     0.0, 0.0, 1.0, (2.745884e-03),
                                     0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
)

fx_scaled = 2 * camera_info['fx'] / np.float32(camera_info['WIDTH'])
fy_scaled = 2 * camera_info['fy'] / np.float32(camera_info['HEIGHT'])
cx_scaled = -1 * 2 * (camera_info['cx'] - 1.0) / np.float32(camera_info['WIDTH'])
cy_scaled = -1 * 2 * (camera_info['cy'] - 1.0) / np.float32(camera_info['HEIGHT'])

camera_intrinsic_parameter = np.array([[fx_scaled, 0.0, cx_scaled],
                                       [0.0, fy_scaled, cy_scaled],
                                       [0.0, 0.0, 1.0]], dtype= np.float32)


# 네트워크 구성 관련 파라메타

network_info = dict(
    batch_size = 20,                        # batch_size take during training
    epochs = 100,                            # total number of epoch
    learning_rate = 5e-4,                   # learining rate
    beta1 = 0.9,                            # momentum term for Adam Optimizer
    freq_print = 10,
    num_worker = 8
)
