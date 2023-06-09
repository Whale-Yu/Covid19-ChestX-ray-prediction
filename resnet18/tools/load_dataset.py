"""
加载数据集、数据预处理
    数据集目录格式:
        |-dataset
            |-images
                |-类别1
                |-类别2
                |-类别3
                |-类别4
                |-...
    注意:同一类别在同一文件夹下，不同类别文件夹是同级目录
"""

from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomResizedCrop, ColorJitter, RandomGrayscale
import torch.utils.data


# 加载指定目录下的图像，返回根据切分比例形成的数据加载器
def load_data(img_dir, shape=(224, 224), rate=0.8, batch_size=128):
    # 数据预处理
    transform = Compose(
        [
            Resize(shape),
            # RandomResizedCrop((224, 224)),    # 随机长宽比裁剪
            RandomHorizontalFlip(),  # 0.5的进行水平翻转
            RandomVerticalFlip(),  # 0.5的进行垂直翻转
            ToTensor(),  # PIL转tensor
            # 归一化   # 输入必须是Tensor
            Normalize(mean=[0.5063, 0.5063, 0.5063], std=[0.2390, 0.2390, 0.2390])  # normnalize.py运行得到
        ]
    )

    # 加载数据集
    dataset = ImageFolder(img_dir, transform=transform)

    all_num = len(dataset)
    train_num = int(all_num * rate)
    # print(all_num)
    # print(train_num)

    # 划分数据集
    train, test = torch.utils.data.random_split(dataset, [train_num, all_num - train_num])

    # 构建数据加载器（封装批处理的迭代器（加载器））
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset.class_to_idx


if __name__ == '__main__':
    # # # 测试
    train, test, class_to_idx = load_data("../dataset/image")
    # train数据集
    train_img_num, train_lab_num = 0, 0
    for image, label in train:
        # print(image, label)
        train_img_num += len(image)
        train_lab_num += len(label)
    print(f'训练集图片数量:{train_img_num};训练集标签数量:{train_lab_num}')
    # test数据集
    test_img_num, test_lab_num = 0, 0
    for image, label in test:
        # print(image, label)
        test_img_num += len(image)
        test_lab_num += len(label)
    print(f'训练集图片数量:{test_img_num};训练集标签数量:{test_lab_num}')

    print(f'数据集类别:{class_to_idx}')
