"""
同resnet文件下的predict_resnet18.py，做了微修改
"""

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Lambda, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.models import resnet18


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
            if self.CUDA:
                img = img.cuda()
            # print(pre_img)
            img = img.view(-1, 3, 224, 224)
            y = self.net(img)
            p_y = torch.nn.functional.softmax(y, dim=1)
            p, cls_idx = torch.max(p_y, dim=1)
            return cls_idx.cpu(), p.cpu()


if __name__ == "__main__":
    # 待预测图片路径
    img_path = '../assets/img/COVID(936)-default.png'  # 默认图片展示效果
    model_file_path = 'model/r18_1.pth'

    recognizer = GarbageRecognizer(model_file_path)

    # 下面转换用于待预测（除数据集）的图像，并对其做预处理
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
    # dataset = ImageFolder(dataset_path, transform=transform)
    # print(dataset.class_to_idx)
    # {'COVID': 0, 'NORMAL': 1, 'Viral_Pneumonia': 2}
    label_list = ['COVID', 'NORMAL', 'Viral_Pneumonia']

    '''
      单张图片预测
    '''
    img_filename = img_path
    print("预测单张图像:", img_filename)
    img = Image.open(img_filename)
    img1 = Image.fromarray(np.uint8(img))
    img2 = transform(img1)
    cls, p = recognizer.recognzie(img2)
    cls = label_list[cls]

    # ps = plt.subplot()
    # plt.imshow(img)
    # title = f"predicted:{cls} , p:{p.numpy()[0] * 100}"
    # ps.set_title(title, fontsize=10)
    # plt.show()
    print("识别结果: 类别:{}, 置信度:{:.2f}%".format(cls, (p.numpy()[0] * 100)))
    # print(cls, '{:}%'.format(p.numpy()[0] * 100))
