# -*- codeing = utf-8 -*-
# @Author : 刘永奇
# @File : models.py
# @Software : PyCharm


import torch.nn as nn
import torch.nn.functional as F
import torchvision

from helper.resnet import resnet18

class ResidualBlock(nn.Module):

    def __init__(self, in_channels):

        # 调用父类nn.Module的构造函数
        super(ResidualBlock, self).__init__()
        # 定义第一个卷积层，输入通道数和输出通道数都为in_channels，卷积核大小为3，填充为1以保持特征图尺寸不变
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # 定义第二个卷积层，输入通道数和输出通道数都为in_channels，卷积核大小为3，填充为1以保持特征图尺寸不变
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):

        # 保存输入作为残差，用于后续的残差连接
        residual = x
        # 第一个卷积层，然后使用ReLU激活函数
        out = F.relu(self.conv1(x))
        # 第二个卷积层
        out = self.conv2(out)
        # 残差连接，将卷积后的结果与原始输入相加
        out += residual
        # 最后再使用ReLU激活函数
        out = F.relu(out)
        return out


class Net(nn.Module):

    def __init__(self, num_classes=4):

        # 调用父类nn.Module的构造函数
        super(Net, self).__init__()
        # 第一个卷积层，输入通道数为3（通常对应RGB图像），输出通道数为16，卷积核大小为5
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        # 第二个卷积层，输入通道数为16，输出通道数为32，卷积核大小为5
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        # 第一个残差块，输入通道数为16
        self.rblock1 = ResidualBlock(16)
        # 第二个残差块，输入通道数为32
        self.rblock2 = ResidualBlock(32)
        # 最大池化层，池化核大小为2，用于下采样特征图
        self.mp = nn.MaxPool2d(2)
        # 全连接层，输入特征数需要根据前面层的输出计算，这里暂时设为89888，输出特征数为分类任务的类别数量
        self.fc = nn.Linear(89888, num_classes)

    def forward(self, x):
        # 获取输入的批量大小
        in_size = x.size(0)
        # 第一个卷积层，然后使用ReLU激活函数，再进行最大池化
        x = self.mp(F.relu(self.conv1(x)))
        # 通过第一个残差块
        x = self.rblock1(x)
        # 第二个卷积层，然后使用ReLU激活函数，再进行最大池化
        x = self.mp(F.relu(self.conv2(x)))
        # 通过第二个残差块
        x = self.rblock2(x)
        # 将特征图展平为一维向量，以便输入到全连接层
        x = x.view(in_size, -1)
        # 通过全连接层得到分类得分
        x = self.fc(x)
        return x


def get_model(model_name=None, num_classes=4, pretrained=True):

    # 打印当前选择的模型名称
    print('==> model name {}'.format(model_name))
    if model_name == "custom":
        # 如果选择自定义模型，创建Net类的实例
        model = Net(num_classes=num_classes)
    elif model_name == "resnet18":
        model = resnet18(pretrained=pretrained)
        # 获取原模型最后一层（全连接层）的输入特征数
        num_ftrs = model.fc.in_features
        # 将原模型的最后一层替换为新的全连接层，输出维度为num_classes
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet34":  # 使用torch库里面写好的resnet34
        # 调用torchvision.models中的resnet34函数创建模型，并根据pretrained参数决定是否加载预训练权重
        model = torchvision.models.resnet34(pretrained=pretrained)
        # 获取原模型最后一层（全连接层）的输入特征数
        num_ftrs = model.fc.in_features
        # 将原模型的最后一层替换为新的全连接层，输出维度为num_classes
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        # 如果传入的模型名称不在支持范围内，抛出值错误异常
        raise ValueError("Unknown model value of : {}".format(model_name))

    return model