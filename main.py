from DataManagement.data_management import *
import UtilityManagement.config as cf
from ModelManagement.PytorchModel.CalibNet import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


trainingset = CalibNetDataset(cf.paths['dataset_path'], training=True)

data_loader = get_loader(trainingset, batch_size=cf.network_info['batch_size'], shuffle=True, num_worker=0)

for i_batch, sample_batched in enumerate(data_loader):
    print(i_batch, sample_batched['source_depth_map'].size(),
          sample_batched['source_image'].size(),
          sample_batched['transform_matrix'].size())

    # observe 4th batch and stop.
    # for i in range(len(sample_batched['source_depth_map'])):
    #     plt.figure()
    #     plt.imshow(sample_batched['source_depth_map'][i])
    #     plt.axis('off')
    #     plt.ioff()
    #     plt.show()

model = CalibNet18(18, 6)

summary(model, [(1, 3, 375, 1242), (1, 1, 375, 1242)], torch.device("cpu"))

