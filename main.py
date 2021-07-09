from DataManagement.data_management import *
import UtilityManagement.config as cf
from ModelManagement.PytorchModel.CalibNet import *
from ModelManagement.model_management import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import math

# dataset test code
trainingset = CalibNetDataset(cf.paths['dataset_path'], training=True)
data_loader = get_loader(trainingset, batch_size=cf.network_info['batch_size'], shuffle=True, num_worker=0)

a = 0
b = 0
for i_batch, sample_batched in enumerate(data_loader):
    print(i_batch, sample_batched['source_depth_map'].size(),
          sample_batched['source_image'].size(),
          sample_batched['transform_matrix'].size())
    a = sample_batched['source_depth_map']
    b = sample_batched['source_image']
    # observe 4th batch and stop.
    # for i in range(len(sample_batched['source_depth_map'])):
    #     plt.figure()
    #     plt.imshow(sample_batched['source_depth_map'][i])
    #     plt.axis('off')
    #     plt.ioff()
    #     plt.show()

# model test code
model = CalibNet18(18, 6)
summary(model, [(1, 3, 375, 1242), (1, 1, 375, 1242)], torch.device("cpu"))

# exponential map test code
vector = torch.Tensor([[[0, 0, 0, 0, 0, math.pi/2]],[[0, 0, 1, 0, math.pi, 0]]])
T = get_RTMatrix_using_EulerAngles(vector)
print(T)
T = get_RTMatrix_using_exponential_logarithm_mapping(vector)
print(T)

# training test code
model.train()
output_vector, max_pool = model(b, a)
print(output_vector.shape, max_pool.shape)

