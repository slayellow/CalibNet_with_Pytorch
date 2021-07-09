from UtilityManagement.pytorch_util import *
import math
import os
import warnings


class BasicBlock(nn.Module):
    expansions = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = set_conv(in_channels, out_channels, kernel=3, strides=stride, padding=1)
        self.bn1 = set_batch_normalization(out_channels)
        self.relu = set_relu(True)

        self.conv2 = set_conv(out_channels, out_channels, kernel=3, padding=1)
        self.bn2 = set_batch_normalization(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansions = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = set_conv(in_channels, out_channels, kernel=1, padding=0)
        self.bn1 = set_batch_normalization(out_channels)
        self.conv2 = set_conv(out_channels, out_channels, kernel=3, strides=stride, padding=1)
        self.bn2 = set_batch_normalization(out_channels)
        self.conv3 = set_conv(out_channels, out_channels * 4, kernel=1, padding=0)
        self.bn3 = set_batch_normalization(out_channels * 4)
        self.relu = set_relu(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DepthNet(nn.Module):
    # DepthNet-18     : BasicBlock        :   [2, 2, 2, 2]
    # DepthNet-34     : BasicBlock        :   [3, 4, 6, 3]
    # DepthNet-50     : BottleNeckBlock   :   [3, 4, 6, 3]
    # DepthNet-101    : BottleNeckBlock   :   [3, 4, 23, 3]
    # DepthNet-152    : BottleNeckBlock   :   [3, 4, 36, 3]
    # Channel       : [32, 64, 128, 256]
    def __init__(self, layer_num, classes):
        super(DepthNet, self).__init__()

        self.model_name = 'DepthNet_{}'.format(layer_num)

        # ResNet의 기본 구성
        blocks = {18: (2, 2, 2, 2),
                  34: (3, 4, 6, 3),
                  50: (3, 4, 6, 3),
                  101: (3, 4, 23, 3),
                  152: (3, 4, 36, 3)}
        channels = (32, 64, 128, 256)

        in_channel = 1
        self.inplanes = 32

        if layer_num is 18 or layer_num is 34:
            self.block = BasicBlock
        elif layer_num is 50 or layer_num is 101 or layer_num is 152:
            self.block = Bottleneck
        else:
            warnings.warn("클래스가 구성하는 Layer 갯수와 맞지 않습니다.")

        self.firstmaxpool = set_max_pool(kernel=5, strides=1)
        self.conv0 = set_conv(in_channel, channels[0], kernel=7, strides=2, padding=3)
        self.bn0 = set_batch_normalization(channels[0])
        self.relu0 = set_relu(True)
        self.maxpool0 = set_max_pool(kernel=3, strides=2, padding=1)
        # Block
        self.layer1 = self._make_layer(self.block, channels[0], blocks[layer_num][0])
        self.layer2 = self._make_layer(self.block, channels[1], blocks[layer_num][1], stride=2)
        self.layer3 = self._make_layer(self.block, channels[2], blocks[layer_num][2], stride=2)
        self.layer4 = self._make_layer(self.block, channels[3], blocks[layer_num][3], stride=2)

        # ***** Not Used ***** #
        # self.gap = set_avg_pool(kernel=7)
        # self.fcl = set_dense(channels[3] * self.block.expansions, classes)

    def forward(self, x):
        max_pool = self.firstmaxpool(x)
        x = self.conv0(max_pool)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)

        # Block
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # ***** Not Used ***** #
        # x = self.gap(x)
        # x = x.view(x.size(0), -1)
        # x = self.fcl(x)
        return x, max_pool

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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.block.expansions:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * self.block.expansions, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.block.expansions),
            )

        layers = []
        layers.append(self.block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * self.block.expansions
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

def DepthNet18(layer_num, classes):
    pretrained_path ="./Log/"
    model = DepthNet(layer_num, classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        print('Pretrained Model!')
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        model.initialize_weights(init_weights=True)

    return model


def DepthNet34(layer_num, classes):
    pretrained_path ="./Log/"
    model = DepthNet(layer_num, classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        model.initialize_weights(init_weights=True)

    return model



def DepthNet50(layer_num, classes):
    pretrained_path ="./Log/"
    model = DepthNet(layer_num, classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        model.initialize_weights(init_weights=True)

    return model



def DepthNet101(layer_num, classes):
    pretrained_path ="./Log/"
    model = DepthNet(layer_num, classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        model.initialize_weights(init_weights=True)

    return model


def DepthNet152(layer_num, classes):
    pretrained_path ="./Log/"
    model = DepthNet(layer_num, classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        model.initialize_weights(init_weights=True)

    return model
