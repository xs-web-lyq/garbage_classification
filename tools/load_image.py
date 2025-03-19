# -*- codeing = utf-8 -*-
# @Author : 刘永奇
# @File : models.py
# @Software : PyCharm
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image


# 定义图像预处理步骤，这里将图像转换为张量，可根据需求添加图像大小调整等操作
transform = transforms.Compose([
    # transforms.Resize((256, 256)),  # 如果需要可以调整图像大小
    transforms.ToTensor()  # 转换为张量
])


def process_image(image_path, process_type):
    """
    对输入图像进行指定类型的处理（卷积或最大池化）并保存处理后的图像。

    参数:
    image_path (str): 输入图像的路径。
    process_type (str): 处理类型，取值为"conv"表示卷积处理，"maxpooling"表示最大池化处理。

    返回:
    None
    """
    # 加载图像并转换为RGB模式
    image = Image.open(image_path).convert("RGB")

    # 将图像转换为张量并添加批次维度，以便后续输入到神经网络层
    input_tensor = transform(image).unsqueeze(0)

    # 根据指定的处理类型进行相应的操作
    if process_type == "conv":
        # 定义卷积层参数
        in_channels, out_channels = 3, 3
        kernel_size = 3
        # 创建卷积层
        conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        # 使用卷积层处理图像
        output = conv_layer(input_tensor)
    elif process_type == "maxpooling":
        # 定义最大池化层参数
        kernel_size = 3
        # 创建最大池化层
        maxpooling_layer = torch.nn.MaxPool2d(kernel_size=kernel_size)
        # 使用最大池化层处理图像
        output = maxpooling_layer(input_tensor)
    else:
        raise ValueError("process_type must be either 'conv' or'maxpooling'")

    # 将处理后的张量转换回PIL图像格式
    output_image = transforms.ToPILImage()(output.squeeze(0))

    # 保存处理后的图像
    output_image.save("./tools/output_image.jpg")


if __name__ == '__main__':
    # 指定处理类型，可以是conv或maxpooling
    process_type = "maxpooling"
    # 指定输入图像路径，这里需要替换为实际的图像路径
    image_path = "./tools/input_image.jpg"
    process_image(image_path, process_type)
