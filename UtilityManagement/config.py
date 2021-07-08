import numpy as np


# 데이터셋 관련 경로

paths = dict(
    resnet_pretrained_path = "",
    dataset_path = "/Users/jinseokhong/data/CalibNet_DataSet/parsed_set_example.txt",
    checkpoint_path = "",
    training_img_result_path = "",
    validation_img_result_path = ""
)


# 카메라 관련 파라메타

camera_info = dict(
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


# 네트워크 구성 관련 파라메타

network_info = dict(
    batch_size = 20,                        # batch_size take during training
    epochs = 40,                            # total number of epoch
    learning_rate = 5e-4,                   # learining rate
    beta1 = 0.9,                            # momentum term for Adam Optimizer
    load_epoch = 0                          # Load checkpoint no.0 at the start of training
)
