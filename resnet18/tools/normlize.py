"""
求自定义数据集的mean和std,用在load_dataset.py
"""

from torchvision.datasets import ImageFolder
import torch
from torchvision.transforms import Resize, ToTensor, Compose
from torch.utils.data import DataLoader

batch_size = 128
transform = Compose([ToTensor(), Resize((224, 224))])
train_dataset = ImageFolder(root='../dataset/image', transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)


def get_mean_std_value(loader):
    '''
    求数据集的均值和标准差
    :param loader:
    :return:
    '''
    data_sum, data_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        # data: [batch_size,channels,height,width]
        # 计算dim=0,2,3维度的均值和，dim=1为通道数量，不用参与计算
        data_sum += torch.mean(data, dim=[0, 2, 3])  # [batch_size,channels,height,width]
        # 计算dim=0,2,3维度的平方均值和，dim=1为通道数量，不用参与计算
        data_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])  # [batch_size,channels,height,width]
        # 统计batch的数量
        num_batches += 1
    # 计算均值
    mean = data_sum / num_batches
    # 计算标准差
    std = (data_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


mean, std = get_mean_std_value(train_loader)
print(f'mean = {mean},std = {std}')

# mean = tensor([0.5063, 0.5063, 0.5063]),std = tensor([0.2390, 0.2390, 0.2390])
