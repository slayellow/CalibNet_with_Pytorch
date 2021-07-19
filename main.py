import torch.autograd

from DataManagement.data_management import *
from ModelManagement.PytorchModel.CalibNet import *
from ModelManagement.model_management import *
from UtilityManagement.AverageMeter import *
import matplotlib.pyplot as plt
import time


ALPHA_LOSS = 1.0
BETHA_LOSS = 1.0
gpu_check = is_gpu_avaliable()
devices = torch.device("cuda") if gpu_check else torch.device("cpu")

# dataset test code
trainingset = CalibNetDataset(cf.paths['dataset_path'], training=True)
data_loader = get_loader(trainingset, batch_size=cf.network_info['batch_size'], shuffle=True, num_worker=cf.network_info['num_worker'])

validationset = CalibNetDataset(cf.paths['dataset_path'], training=False)
valid_loader = get_loader(validationset, batch_size=cf.network_info['batch_size'], shuffle=False, num_worker=cf.network_info['num_worker'])

# model test code
model = CalibNet18(18, 6).to(devices)
summary(model, [(1, 3, 375, 1242), (1, 1, 375, 1242)], devices)

K_final = torch.tensor(cf.camera_intrinsic_parameter, dtype=torch.float32).to(devices)
small_transform = torch.tensor(cf.camera_info['cam_transform_02_inv'], dtype=torch.float32).to(devices)

loss_fucntion = loss_MSE().to(devices)
optimizer = set_Adam(model, learning_rate=cf.network_info['learning_rate'])

pretrained_path = cf.paths['pretrained_path']
if os.path.isfile(os.path.join(pretrained_path, model.get_name() + '.pth')):
    print("Pretrained Model Open : ", model.get_name() + ".pth")
    checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
    start_epoch = checkpoint['epoch']
    load_weight_parameter(model, checkpoint['state_dict'])
    load_weight_parameter(optimizer, checkpoint['optimizer'])
else:
    print("No Pretrained Model")
    start_epoch = 0

for epoch in range(start_epoch, cf.network_info['epochs']):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for i_batch, sample_bathced in enumerate(data_loader):
        data_time.update(time.time() - end)

        source_depth_map = sample_bathced['source_depth_map']
        source_image = sample_bathced['source_image']
        target_depth_map = sample_bathced['target_depth_map']
        expected_transform = sample_bathced['transform_matrix']
        if gpu_check:
            source_depth_map = source_depth_map.to(devices)
            source_image = source_image.to(devices)
            target_depth_map = target_depth_map.to(devices)
            expected_transform = expected_transform.to(devices)

        output_vector, first_max_pool = model(source_image, source_depth_map)
        output_vector = torch.autograd.Variable(output_vector, requires_grad=True).to(devices)
        T = get_RTMatrix_using_exponential_logarithm_mapping(output_vector).to(devices)
        depth_map_predicted, sparse_cloud_predicted = get_transformed_matrix(first_max_pool * 40.0 + 40.0, T, K_final,
                                                     small_transform)
        depth_map_predicted = torch.autograd.Variable(depth_map_predicted, requires_grad=True).to(devices)
        sparse_cloud_predicted = torch.autograd.Variable(sparse_cloud_predicted, requires_grad=True).to(devices)

        depth_map_expected, sparse_cloud_expected = get_transformed_matrix(first_max_pool * 40.0 + 40.0, expected_transform, K_final,
                                                    small_transform)

        depth_map_expected = depth_map_expected.to(devices)
        sparse_cloud_expected = sparse_cloud_expected.to(devices)

        cloud_loss = earth_mover_distance(sparse_cloud_predicted, sparse_cloud_expected, transpose=False)

        photometric_loss = loss_fucntion((depth_map_predicted[:, 10:-10, 10:-10] - 40.0) / 40.0,
                             (depth_map_expected[:, 10:-10, 10:-10] - 40.0) / 40.0)

        loss = ALPHA_LOSS*photometric_loss + BETHA_LOSS*cloud_loss

        losses.update(loss.item(), source_depth_map.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i_batch % cf.network_info['freq_print'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch+1, i_batch, len(data_loader),
                                                        batch_time=batch_time, data_time=data_time, loss=losses))

        a = source_depth_map
        b = target_depth_map
        c = depth_map_predicted
        d = depth_map_expected

        # count = 0
        # for i in range(sparse_cloud_expected.shape[1]):
        #     for y in range(sparse_cloud_expected_pytorch.shape[2]):
        #         if sparse_cloud_expected[0, i, y].detach().numpy() == 0.0:
        #             if sparse_cloud_expected_pytorch[0, i, y].detach().numpy() == 0.0:
        #                 continue
        #             else:
        #                 count = count+1
        #         else:
        #             if sparse_cloud_expected_pytorch[0, i, y].detach().numpy() == 0.0:
        #                 count = count+1
        #             else:
        #                 continue
        # print("Another Point Position : ", count)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Sharing x per column, y per row')
    ax1.imshow(a[0, 0, :, :].cpu())
    ax1.axis('off')
    ax3.imshow(b[0, 0, :, :].cpu())
    ax3.axis('off')
    ax2.imshow(c[0, :, :].cpu().detach().numpy())
    ax2.axis('off')
    ax4.imshow(d[0, :, :].cpu().detach().numpy())
    ax4.axis('off')
    plt.savefig(os.path.join(cf.paths['training_img_result_path'], 'Training_'+str(epoch+1)+'.png'))
    plt.close()

    valid_batch_time = AverageMeter()
    valid_data_time = AverageMeter()
    valid_losses = AverageMeter()

    model.eval()

    end = time.time()
    for i_batch, sample_bathced in enumerate(valid_loader):
        data_time.update(time.time() - end)

        source_depth_map = sample_bathced['source_depth_map']
        source_image = sample_bathced['source_image']
        target_depth_map = sample_bathced['target_depth_map']
        expected_transform = sample_bathced['transform_matrix']
        if gpu_check:
            source_depth_map = source_depth_map.to(devices)
            source_image = source_image.to(devices)
            target_depth_map = target_depth_map.to(devices)
            expected_transform = expected_transform.to(devices)

        output_vector, first_max_pool = model(source_image, source_depth_map)
        output_vector = torch.autograd.Variable(output_vector, requires_grad=True).to(devices)
        T = get_RTMatrix_using_exponential_logarithm_mapping(output_vector).to(devices)
        depth_map_predicted, sparse_cloud_predicted = get_transformed_matrix(first_max_pool * 40.0 + 40.0, T, K_final,
                                                                             small_transform)
        depth_map_predicted = torch.autograd.Variable(depth_map_predicted, requires_grad=True).to(devices)
        sparse_cloud_predicted = torch.autograd.Variable(sparse_cloud_predicted, requires_grad=True).to(devices)

        depth_map_expected, sparse_cloud_expected = get_transformed_matrix(first_max_pool * 40.0 + 40.0,
                                                                           expected_transform, K_final,
                                                                           small_transform)

        depth_map_expected = depth_map_expected.to(devices)
        sparse_cloud_expected = sparse_cloud_expected.to(devices)

        cloud_loss = earth_mover_distance(sparse_cloud_predicted, sparse_cloud_expected, transpose=False)

        photometric_loss = loss_fucntion((depth_map_predicted[:, 10:-10, 10:-10] - 40.0) / 40.0,
                                         (depth_map_expected[:, 10:-10, 10:-10] - 40.0) / 40.0)

        loss = ALPHA_LOSS * photometric_loss + BETHA_LOSS * cloud_loss
        losses.update(loss.item(), source_depth_map.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i_batch % cf.network_info['freq_print'] == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i_batch, len(valid_loader),
                                                        batch_time=batch_time, data_time=data_time, loss=losses))

        a = source_depth_map
        b = target_depth_map
        c = depth_map_predicted
        d = depth_map_expected
        e = source_image

    save_checkpoint({
        'epoch': epoch + 1,
        'arch' : model.get_name(),
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}, False, os.path.join(pretrained_path, model.get_name()),'pth')

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Sharing x per column, y per row')
    ax1.imshow(a[0, 0, :, :].cpu())
    ax1.axis('off')
    ax3.imshow(b[0, 0, :, :].cpu())
    ax3.axis('off')
    ax2.imshow(c[0, :, :].cpu().detach().numpy())
    ax2.axis('off')
    ax4.imshow(d[0, :, :].cpu().detach().numpy())
    ax4.axis('off')
    plt.savefig(os.path.join(cf.paths['validation_img_result_path'], 'Validation_'+str(epoch+1)+'.png'))
    plt.close()

print("Train Finished!!")
