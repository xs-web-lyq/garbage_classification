# -*- codeing = utf-8 -*-
# @Author : 刘永奇
# @File : models.py
# @Software : PyCharm

import os
import json
from sklearn.model_selection import train_test_split


# 定义类别标签映射字典（关键配置）
# 格式说明：{"文件夹名称": 对应数字标签}
label_dict = {
    "apple": 0,     # 苹果类别
    "banana": 1,    # 香蕉类别
    "battery": 2,   # 电池类别
    "brick": 3     # 砖头类别
}

# 数据集路径配置（根据实际情况修改）
images_folder = "../dataset/images"                  # 图片存储根目录
label_file_train = "../dataset/label_file/label_file_train.json"  # 训练集标签输出路径
label_file_val = "../dataset/label_file/label_file_val.json"      # 验证集标签输出路径
label_file_test = "../dataset/label_file/label_file_test.json"     #测试集

# 数据集划分参数
test_size = 0.1      # 测试集比例（0.1表示10%作为验证集）
value_size = 0.2     # 验证集比例

random_state = 42    # 随机种子（保证划分结果可复现）

# ---------------------------
# 数据处理流程（核心逻辑）
# ---------------------------

def generate_label_files():
    """
    生成训练集和验证集标签文件的主函数
    执行流程：
    1. 遍历图片目录获取所有带标签的图片路径
    2. 划分训练集/验证集
    3. 生成JSON格式的标签文件
    """
    # 创建空字典存储图片路径与标签的映射关系
    # 格式：{"相对路径": [标签]}
    image_labels = {}

    # 遍历图片根目录下的每个子文件夹
    for folder_name in os.listdir(images_folder):
        folder_path = os.path.join(images_folder, folder_name)
        
        # 跳过非目录文件（确保只处理类别文件夹）
        if not os.path.isdir(folder_path):
            continue
        
        # 获取当前文件夹对应的数字标签
        label = label_dict.get(folder_name, -1)
        
        # 处理未定义类别的异常情况
        if label == -1:
            print(f"警告：检测到未定义类别文件夹 '{folder_name}'，已跳过")
            continue
        
        # 遍历当前类别文件夹下的所有图片文件
        for image_name in os.listdir(folder_path):
            # 检查文件扩展名（支持.jpg和.png格式）
            if image_name.lower().endswith(('.jpg', '.png')):
                # 构建相对路径（保留子目录结构）
                # 格式：类别文件夹/图片文件名（如 apple/img001.jpg）
                relative_path = os.path.join(folder_name, image_name)
                # 存储标签（用列表包裹以保持格式统一）
                image_labels[relative_path] = [label]

    # ---------------------------
    # 数据集划分（核心步骤）
    # ---------------------------
    # 将字典项转换为可划分的列表格式
    items = list(image_labels.items())
    # 使用sklearn的train_test_split进行划分
    # stratify参数可用于分层抽样（需要额外处理）
    train_data, test_data  = train_test_split(
        items,
        test_size=test_size,
        random_state=random_state
    )
    train_data, value_data  = train_test_split(
        train_data,
        test_size=value_size,
        random_state=random_state
    )
    # 转换回字典格式
    train_labels = dict(train_data)
    val_labels = dict(value_data)
    test_labels = dict(test_data)

    # ---------------------------
    # 文件保存（输出结果）
    # ---------------------------
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(label_file_train), exist_ok=True)
    os.makedirs(os.path.dirname(label_file_val), exist_ok=True)
    os.makedirs(os.path.dirname(label_file_test), exist_ok=True)

    # 写入训练集标签文件
    with open(label_file_train, "w") as f:
        json.dump(train_labels, f, indent=2)  # indent参数美化格式
    
    # 写入验证集标签文件
    with open(label_file_val, "w") as f:
        json.dump(val_labels, f, indent=2)

     # 写入验证集标签文件
    with open(label_file_test, "w") as f:
        json.dump(test_labels, f, indent=2)

    print("数据集划分完成！")
    print(f"训练集样本数: {len(train_labels)}")
    print(f"验证集样本数: {len(val_labels)}")
    print(f"测试集样本数: {len(test_labels)}")
    print(f"标签文件已保存至: {os.path.dirname(label_file_train)}")

if __name__ == "__main__":
    generate_label_files()

'''
python3 train.py --base_path="./dataset/images" --label_file_train="./dataset/label_file/label_file_train.json" --label_file_val="./dataset/label_file/label_file_val.json" --model_name="resnet18" --num_classes=4 --batch_size=32 --learning_rate=0.001 --epochs=50 --save_fig_name="exp1" --gpu
'''