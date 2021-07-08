import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import cv2


class CalibNetDataset(Dataset):
    def __init__(self, path, training=True):
        """
        Args:
            csv_file (string): csv 파일의 경로
            training (bool) : training 인지 validation 인지 여부 확
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.dataset = np.loadtxt(path, dtype = str)
        self.training = training
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.data_num = len(self.dataset) * 0.8

        if self.training:
            self.dataset = self.dataset[:int(self.data_num)]
        else:
            self.dataset = self.dataset[int(self.data_num):]

        self.source_depth_map = self.dataset[:, 0]
        self.target_depth_map = self.dataset[:, 1]
        self.source_image = self.dataset[:, 2]
        self.target_image = self.dataset[:, 3]
        self.transforms = np.float32(self.dataset[:, 4:])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        source_map = np.float32(cv2.imread(self.source_depth_map[idx], flags=cv2.IMREAD_GRAYSCALE))
        source_map[0:5, :] = 0.0
        source_map[:, 0:5] = 0.0
        source_map[source_map.shape[0] - 5:, :] = 0
        source_map[:, source_map.shape[1] - 5:] = 0
        source_map = (source_map - 40.0) / 40.0

        target_map = np.float32(cv2.imread(self.target_depth_map[idx], flags=cv2.IMREAD_GRAYSCALE))
        target_map[0:5, :] = 0.0
        target_map[:, 0:5] = 0.0
        target_map[target_map.shape[0] - 5:, :] = 0
        target_map[:, target_map.shape[1] - 5:] = 0
        target_map = (target_map - 40.0) / 40.0

        source_img = np.float32(cv2.imread(self.source_image[idx], flags=cv2.IMREAD_COLOR))
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        source_img[0:5, :, :] = 0.0
        source_img[:, 0:5, :] = 0.0
        source_img[source_img.shape[0] - 5:, :, :] = 0
        source_img[:, source_img.shape[1] - 5:, :] = 0
        source_img = (source_img - 127.5) / 127.5

        target_img = np.float32(cv2.imread(self.target_image[idx], flags=cv2.IMREAD_COLOR))
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        target_img[0:5, :, :] = 0.0
        target_img[:, 0:5, :] = 0.0
        target_img[target_img.shape[0] - 5:, :, :] = 0
        target_img[:, target_img.shape[1] - 5:, :] = 0
        target_img = (target_img - 127.5) / 127.5

        transformed = np.linalg.inv(self.transforms[idx].reshape(4, 4))     # transform의 역행렬

        data = {'source_depth_map': self.transform(source_map), 'target_depth_map': self.transform(target_map),
                'source_image': self.transform(source_img), 'target_image':self.transform(target_img),
                'transform_matrix': transformed}

        return data


def get_loader(dataset, batch_size, shuffle=True, num_worker=0):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
        pin_memory=True,
        sampler=None
        )
    return dataloader



