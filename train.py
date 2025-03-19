# -*- codeing = utf-8 -*-
# @Author : 刘永奇
# @File : models.py
# @Software : PyCharm


# 导入必要的库
import torch  # PyTorch深度学习框架
import torchvision.transforms as transforms  # 图像预处理工具
import torch.optim as optim  # 优化器模块
import torch.nn as nn  # 神经网络模块
from torch.utils.data import Dataset, DataLoader  # 数据集加载工具

import argparse  # 命令行参数解析
import matplotlib.pyplot as plt  # 绘图库
import os  # 操作系统接口
import json  # JSON文件处理
from PIL import Image  # 图像处理库
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 获取模型
from helper.models import get_model  # 自定义模型加载函数

# 计算类别权重
class_counts =  torch.tensor([69.0,84.0,119.0, 31.0], dtype=torch.float32)
class_weights = 1.0 / class_counts  # 样本数倒数
class_weights = class_weights / class_weights.sum()  # 归一化

# 自定义数据集类（核心组件：数据加载）
class ImageFolderDataset(Dataset):
    def __init__(self, root, label_file, transform=None):
        """
        初始化数据集类（关键步骤：数据路径处理和数据转换设置）
        - root: 图像存储根目录
        - label_file: 包含图像标签的JSON文件路径
        - transform: 图像预处理流水线
        """
        self.root = root
        self.transform = transform
        self.data_paths = []  # 存储图像路径
        self.labels = []      # 存储对应标签

        # 读取标签文件（关键步骤：数据标注加载）
        with open(label_file, 'r') as f:
            label_data = json.load(f)
            for image_name, label_list in label_data.items():
                image_path = os.path.join(self.root, image_name)
                self.data_paths.append(image_path)
                self.labels.append(label_list[0])  # 假设每个图像只有一个标签

    def __len__(self):
        """返回数据集样本数量（必要方法）"""
        return len(self.data_paths)

    def __getitem__(self, idx):
        """获取单个样本（核心方法：数据加载和预处理）"""
        image_path = self.data_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')  # 加载图像并转为RGB
        
        # 应用数据增强/预处理（关键步骤：图像变换）
        if self.transform:
            image = self.transform(image)
        return image, label


# 数据准备组件（核心组件：数据预处理和加载器创建）
def get_dataset_urbanpipe(args):
    # 训练集数据增强策略（关键步骤：提升模型泛化能力）
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪缩放
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize(  # 标准化（ImageNet统计量）
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 验证集预处理（关键区别：无数据增强，仅做中心裁剪）
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    # 创建数据集实例（核心对象：训练/测试数据容器）
    train_dataset = ImageFolderDataset(
        root=args.base_path,
        label_file=args.label_file_train,
        transform=train_transform
    )
    value_dataset = ImageFolderDataset(
        root=args.base_path,
        label_file=args.label_file_val,
        transform=val_transform
    )
    test_dataset = ImageFolderDataset(
        root=args.base_path,
        label_file=args.label_file_test,
        transform=test_transform
    )

    # 创建数据加载器（关键组件：批量数据生成器）
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,         # 训练时打乱数据
        batch_size=args.batch_size,
        num_workers=4         # 多进程加速（实际使用时根据硬件调整）
    )
    value_loader = DataLoader(
        value_dataset,
        shuffle=False,        # 验证时不需打乱
        batch_size=args.batch_size
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,        # 测试时不需打乱
        batch_size=args.batch_size
    )
    return train_loader, value_loader, test_loader 


# 模型训练组件（核心组件：前向传播+反向传播）
def train(model, train_loader, criterion, optimizer, epoch, device,model_name):
    """
    训练单epoch流程（关键步骤：模型参数更新）
    - model: 待训练模型
    - train_loader: 训练数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - epoch: 当前训练轮次
    - device: 计算设备（CPU/GPU）
    """
    model.train()  # 切换训练模式（启用Dropout等）
    running_loss = 0.0
    correct = 0
    total = 0

    # 迭代训练批次数据（核心循环）
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # 数据迁移到设备

        optimizer.zero_grad()  # 梯度清零（关键步骤：防止梯度累积）
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播（核心计算：梯度计算）
        loss.backward()
        optimizer.step()  # 参数更新

        # 统计指标
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 计算epoch指标
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    print(f'{model_name}-Epoch {epoch+1}: Train Loss: {train_loss:.3f}, Accuracy: {train_acc*100:.2f}%')
    return train_loss, train_acc


# 模型验证组件（核心功能：模型性能评估）
def value(model, value_loader, criterion, device,model_name):
    """模型测试流程（关键区别：无梯度计算）"""
    model.eval()  # 切换评估模式（关闭Dropout等）
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():  # 禁用梯度计算（节省内存）
        for inputs, labels in value_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    value_loss = running_loss / len(value_loader)
    value_acc = correct / total
    print(f'                value Loss: {value_loss:.3f}, Accuracy: {value_acc*100:.2f}%')
    return value_loss, value_acc



# 模型测试组件（核心功能：模型性能评估）
def test(model1, model2,test_loader, criterion, device,weight_custom,weight_resnet):
    """模型测试流程（关键区别：无梯度计算）"""
    model1.eval()  # 切换评估模式（关闭Dropout等）
    model2.eval()  # 切换评估模式（关闭Dropout等）
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算（节省内存）
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            resnet_output = model1(inputs)
            custom_output = model2(inputs)
            # 加权平均
            final_output = weight_custom * custom_output + weight_resnet * resnet_output
            
            

            # 统计指标
            _, predicted = torch.max(final_output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total
    print(f'模型融合后的 Accuracy: {test_acc*100:.2f}%')
    return  test_acc


# 主流程组件（核心组件：训练流程编排）
def main(args):
    # 设备选择（关键配置：GPU加速）
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f'Using device: {device}')

    # 获取数据（关键步骤：数据准备）
    train_loader, value_loader, test_loader= get_dataset_urbanpipe(args)

    # 模型初始化（核心组件：模型架构选择）
    model1 = get_model(
        model_name=args.model_name1,
        num_classes=args.num_classes,
        pretrained=args.load_pretrained  # 使用预训练权重（迁移学习关键）
    ).to(device)

    model2 = get_model(
        model_name=args.model_name2,
        num_classes=args.num_classes,
        pretrained=args.load_pretrained  # 使用预训练权重（迁移学习关键）
    ).to(device)

    # 损失函数和优化器（核心组件：学习目标定义和优化策略）
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # 分类任务常用损失函数
    optimizer1 = optim.Adam(
        model1.parameters(),
        lr=args.learning_rate,
    )
    optimizer2 = optim.Adam(
        model2.parameters(),
        lr=args.learning_rate,
    )

    # 训练指标记录（可视化准备）
    train_losses1, train_accuracies1= [], []
    value_losses1, value_accuracies1 = [], []
    train_losses2, train_accuracies2 = [], []
    value_losses2, value_accuracies2 = [], []
    test_losses, test_accuracies = [], []

    # 训练循环（核心流程：迭代优化）
    for epoch in range(args.epochs):
        # 单个epoch训练
        train_loss, train_acc = train(
            model1, train_loader, criterion,
            optimizer1, epoch, device,args.model_name1
        )
        
        # 模型验证
        value_loss, value_acc = value(
            model1, value_loader, criterion, device,args.model_name1
        )
        
        # 记录指标
        train_losses1.append(train_loss)
        train_accuracies1.append(train_acc)
        value_losses1.append(value_loss)
        value_accuracies1.append(value_acc)

    for epoch in range(args.epochs):
        # 单个epoch训练
        train_loss, train_acc = train(
            model2, train_loader, criterion,
            optimizer2, epoch, device,args.model_name2
        )
        
        # 模型验证
        value_loss, value_acc = value(
            model2, value_loader, criterion, device,args.model_name2
        )
        
        # 记录指标
        train_losses2.append(train_loss)
        train_accuracies2.append(train_acc)
        value_losses2.append(value_loss)
        value_accuracies2.append(value_acc)

    # 模型测试
    test_acc = test(
        model1, model2, test_loader, criterion, device,args.weight_custom,args.weight_resnet
    )
    # 记录指标
    test_accuracies.append(test_acc)

    # 可视化训练过程（关键分析工具）
    plt.figure(figsize=(10, 5))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses1, label='Train')
    plt.plot(value_losses1, label='value')
    plt.title('custom_Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies1, label='Train')
    plt.plot(value_accuracies1, label='value')
    plt.title('custom_Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # 保存结果
    plt.savefig(f'training_plot_{args.save_fig_name1}.png')
    plt.close()  # 防止内存泄漏


    # 可视化训练过程（关键分析工具）
    plt.figure(figsize=(10, 5))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses2, label='Train')
    plt.plot(value_losses2, label='value')
    plt.title('resnet_Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies2, label='Train')
    plt.plot(value_accuracies2, label='value')
    plt.title('resnet_Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # 保存结果
    plt.savefig(f'training_plot_{args.save_fig_name2}.png')
    plt.close()  # 防止内存泄漏


# 参数配置组件（核心功能：实验配置管理）
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图像分类训练脚本")
    
    # 数据参数
    parser.add_argument("--base_path", type=str, default="./dataset/images",
                        help="图像数据根目录")
    parser.add_argument("--label_file_train", type=str,
                        default="./dataset/label_file/label_file_train.json",
                        help="训练集标签文件路径")
    parser.add_argument("--label_file_val", type=str,
                        default="./dataset/label_file/label_file_val.json",
                        help="验证集标签文件路径")
    parser.add_argument("--label_file_test", type=str,
                        default="./dataset/label_file/label_file_test.json",
                        help="测试集标签文件路径")
    
    # 模型参数
    parser.add_argument("--model_name1", type=str, default="custom",
                        help="模型架构名称（支持torchvision模型）")
    parser.add_argument("--model_name2", type=str, default="resnet18",
                        help="模型架构名称（支持torchvision模型）")
    parser.add_argument("--load_pretrained", action="store_true",
                    help="是否加载预训练权重，迁移学习的关键）")

    parser.add_argument("--num_classes", type=int, default=4,
                        help="分类类别数")


    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16,
                        help="训练批次大小（根据GPU显存调整）")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="初始学习率（重要超参数）")
    parser.add_argument("--epochs", type=int, default=20,
                        help="训练总轮次")


    # 定义加权平均的权重
    parser.add_argument("--weight_custom", type=float, default=0.4,
                        help="自定义模型的权重")
    parser.add_argument("--weight_resnet", type=float, default=0.6,
                        help="ResNet 模型的权重")
    
    # 实验管理
    parser.add_argument("--save_fig_name1", type=str, default="exp1",
                        help="结果图表保存名称（方便实验对比）")
    parser.add_argument("--save_fig_name2", type=str, default="exp2",
                        help="结果图表保存名称（方便实验对比）")
    parser.add_argument("--gpu", action="store_true",
                        help="是否使用GPU加速（需要CUDA环境）")
    
    args = parser.parse_args()
    
    # 启动训练流程
    main(args)
