from UtilityManagement.pytorch_util import *
from ModelManagement.PytorchModel.ResNet import *
from ModelManagement.PytorchModel.DepthNet import *
import math
import os
import warnings


class CalibNet(nn.Module):
    # ResNet18 + DepthNet18
    def __init__(self, layer_num, classes):
        super(CalibNet, self).__init__()

        self.channels = (384, 192, 96)

        self.model_name = 'CalibNet_{}'.format(layer_num)

        self.resnet = ResNet18(layer_num, 0)
        self.depthnet = DepthNet18(layer_num, 0)

        self.conv0 = set_conv(512 + 256, self.channels[0], kernel=3, strides=2, padding=1)
        self.bn0 = set_batch_normalization(self.channels[0])
        self.relu0 = set_relu(True)

        self.conv1 = set_conv(self.channels[0], self.channels[1], kernel=3, strides=2, padding=1)
        self.bn1 = set_batch_normalization(self.channels[1])
        self.relu1 = set_relu(True)

        self.conv2 = set_conv(self.channels[1], self.channels[2], kernel=1, strides=2)
        self.bn2 = set_batch_normalization(self.channels[2])
        self.relu2 = set_relu(True)

        self.conv_rot = set_conv(self.channels[2], self.channels[2], kernel=1, strides=1, padding=0)
        self.bn_rot = set_batch_normalization(self.channels[2])
        self.relu_rot = set_relu(True)
        self.dropout_rot = set_dropout(0.7)
        self.fcl_rot = set_dense(3*6*96, 3)

        self.conv_tr = set_conv(self.channels[2], self.channels[2], kernel=1, strides=1, padding=0)
        self.bn_tr = set_batch_normalization(self.channels[2])
        self.relu_tr = set_relu(True)
        self.dropout_tr = set_dropout(0.7)
        self.fcl_tr = set_dense(3*6*96, 3)

    def forward(self, x1, x2):
        x1 = self.resnet(x1)
        x2 = self.depthnet(x2)

        x = set_concat([x1, x2], axis=1)

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        rot = self.conv_rot(x)
        rot = self.bn_rot(rot)
        rot = self.relu_rot(rot)
        rot = rot.view(rot.size(0), -1)
        rot = self.dropout_rot(rot)
        rot = self.fcl_rot(rot)

        tr = self.conv_tr(x)
        tr = self.bn_tr(tr)
        tr = self.relu_tr(tr)
        tr = tr.view(tr.size(0), -1)
        tr = self.dropout_tr(tr)
        tr = self.fcl_tr(tr)

        x = set_concat([tr, rot], axis=1)

        return x


    def get_name(self):
        return self.model_name

    def initialize_weights(self, init_weights):
        if init_weights is True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()

                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()


def CalibNet18(layer_num, classes):
    pretrained_path ="./Log/"
    model = CalibNet(layer_num, classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        print('Pretrained Model!')
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        model.initialize_weights(init_weights=True)

    return model