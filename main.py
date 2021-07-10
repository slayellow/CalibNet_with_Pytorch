from DataManagement.data_management import *
from ModelManagement.PytorchModel.CalibNet import *
from ModelManagement.model_management import *
from UtilityManagement.AverageMeter import *
from scipy.misc import imsave
import matplotlib.pyplot as plt
import time


gpu_check = is_gpu_avaliable()
devices = torch.device("cuda") if gpu_check else torch.device("cpu")

# dataset test code
trainingset = CalibNetDataset(cf.paths['dataset_path'], training=True)
data_loader = get_loader(trainingset, batch_size=cf.network_info['batch_size'], shuffle=True, num_worker=0)

validationset = CalibNetDataset(cf.paths['dataset_path'], training=False)
valid_loader = get_loader(validationset, batch_size=cf.network_info['batch_size'], shuffle=False, num_worker=0)

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
        T = get_RTMatrix_using_exponential_logarithm_mapping(output_vector)
        depth_map_predicted = get_transformed_matrix(first_max_pool * 40.0 + 40.0, T, K_final, small_transform)
        depth_map_predicted = torch.autograd.Variable(depth_map_predicted, requires_grad=True)
        print("Predicted RT Matrix " , T)
        print("Expected RT Matrix : " ,expected_transform)
        depth_map_expected = get_transformed_matrix(first_max_pool * 40.0 + 40.0, expected_transform, K_final, small_transform)

        loss = loss_fucntion((depth_map_predicted[:, 10:-10, 10:-10] - 40.0) / 40.0, (depth_map_expected[:, 10:-10, 10:-10] - 40.0) / 40.0)

        losses.update(loss.item(), source_depth_map.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i_batch % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch+1, i_batch+1, len(data_loader),
                                                        batch_time=batch_time, data_time=data_time, loss=losses))

        a = source_depth_map
        b = target_depth_map
        c = depth_map_predicted
        d = depth_map_expected

    plt.subplot(4, 1, 1)
    plt.imshow(a[0, 0, :, :])
    plt.axis('off')
    plt.ioff()
    plt.title('Input Depth Map', fontsize=7)
    plt.subplot(4, 1, 2)
    plt.imshow(b[0, 0, :, :])
    plt.axis('off')
    plt.ioff()
    plt.title('Target Depth Map', fontsize=7)
    plt.subplot(4, 1, 3)
    plt.imshow(c[0, :, :].detach().numpy())
    plt.axis('off')
    plt.ioff()
    plt.title('Predicted Depth Map', fontsize=7)
    plt.subplot(4, 1, 4)
    plt.imshow(d[0, :, :])
    plt.axis('off')
    plt.ioff()
    plt.title('Expected Depth Map', fontsize=7)
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
        T = get_RTMatrix_using_exponential_logarithm_mapping(output_vector).to(devices)
        depth_map_predicted = get_transformed_matrix(first_max_pool * 40.0 + 40.0, T, K_final, small_transform).to(devices)
        depth_map_predicted = torch.autograd.Variable(depth_map_predicted, requires_grad=True).to(devices)

        depth_map_expected = get_transformed_matrix(first_max_pool * 40.0 + 40.0, expected_transform, K_final, small_transform).to(devices)

        loss = loss_fucntion((depth_map_predicted[:, 10:-10, 10:-10] - 40.0) / 40.0, (depth_map_expected[:, 10:-10, 10:-10] - 40.0) / 40.0)

        losses.update(loss.item(), source_depth_map.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i_batch % 1 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i_batch+1, len(valid_loader),
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

    plt.subplot(4, 1, 1)
    plt.imshow(a[0, 0, :, :])
    plt.axis('off')
    plt.ioff()
    plt.title('Input Depth Map', fontsize=7)
    plt.subplot(4, 1, 2)
    plt.imshow(b[0, 0, :, :])
    plt.axis('off')
    plt.ioff()
    plt.title('Target Depth Map', fontsize=7)
    plt.subplot(4, 1, 3)
    plt.imshow(c[0, :, :].detach().numpy())
    plt.axis('off')
    plt.ioff()
    plt.title('Predicted Depth Map', fontsize=7)
    plt.subplot(4, 1, 4)
    plt.imshow(d[0, :, :])
    plt.axis('off')
    plt.ioff()
    plt.title('Expected Depth Map', fontsize=7)
    plt.savefig(os.path.join(cf.paths['validation_img_result_path'], 'Validation_'+str(epoch+1)+'.png'))
    plt.close()

print("Train Finished!!")
