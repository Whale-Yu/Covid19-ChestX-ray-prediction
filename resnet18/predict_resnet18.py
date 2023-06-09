"""
    预测代码，对一张图片进行预测返回结果
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, Lambda
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
import random


class GarbageRecognizer:
    def __init__(self, module_file=""):
        super(GarbageRecognizer, self).__init__()
        self.module_file = module_file
        self.CUDA = torch.cuda.is_available()
        self.net = resnet18(pretrained=False, num_classes=3)
        if self.CUDA:
            self.net.cuda()
            device = 'cuda'
        else:
            device = 'cpu'
        state = torch.load(self.module_file, map_location=device)
        self.net.load_state_dict(state)
        print("加载模型完毕!")
        self.net.eval()

    @torch.no_grad()
    def recognzie(self, img):
        with torch.no_grad():
            # 开始识别
            if self.CUDA:
                img = img.cuda()
            # print(pre_img)
            img = img.view(-1, 3, 224, 224)
            y = self.net(img)
            p_y = torch.nn.functional.softmax(y, dim=1)
            p, cls_idx = torch.max(p_y, dim=1)
            return cls_idx.cpu(), p.cpu()


if __name__ == "__main__":
    # 模型
    model_file = 'model/r18_1.pth'
    recognizer = GarbageRecognizer(model_file)

    # # 下面转换用于独立的图像，并对其做预处理
    transform = Compose(
        [
            Resize((224, 224)),
            # RandomHorizontalFlip(),  # 0.5的进行水平翻转
            # RandomVerticalFlip(),  # 0.5的进行垂直翻转
            ToTensor(),  # PIL转tensor
            Lambda(lambda x: x.repeat(3, 1, 1)),
            Normalize(mean=[0.5063, 0.5063, 0.5063], std=[0.2390, 0.2390, 0.2390])  # normnalize.py运行得到
        ]
    )
    dataset = ImageFolder('./dataset/image', transform=transform)
    # print(dataset.class_to_idx)

    '''
    单张图片预测
    '''
    img_filename = 'inputs/待预测图片/COVID (925).png'
    print("预测单张图像：", img_filename)
    img = Image.open(img_filename)
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    cls, p = recognizer.recognzie(img)
    cls = dataset.classes[cls]
    print(cls, '{:}%'.format(p.numpy()[0] * 100))
