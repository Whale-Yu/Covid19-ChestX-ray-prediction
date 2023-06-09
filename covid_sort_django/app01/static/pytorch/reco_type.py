import numpy as np
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Lambda

from app01.static.pytorch.recognizer import Recognizer


def reco_type(img_path, model_file_path):
    # 待预测图片路径
    img_path = img_path
    model_file_path = model_file_path

    recognizer = Recognizer(model_file_path)

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
    print(label_list)

    '''
      单张图片预测
    '''
    img_filename = rf'{img_path}'
    print("预测单张图像:", img_filename)
    img = Image.open(img_filename)
    img1 = Image.fromarray(np.uint8(img))
    img2 = transform(img1)
    cls, p = recognizer.recognzie(img2)
    cls = label_list[cls]
    # res=f"类别:{cls}, 置信度:{format(p.numpy()[0] * 100)}%"
    # res = '类别:{}, 置信度:{:.2f}%'.format(cls, p.numpy()[0] * 100)

    classes = f'{cls}'
    score = '{:.2f}%'.format(p.numpy()[0] * 100)
    return classes, score


if __name__ == '__main__':
    # 测试
    img_path = r'C:\Users\yujunyu\Desktop\待预测图片\NORMAL (7).png'
    model = r'D:\PycharmProject(D)\外包\23-3-15-基于数据挖掘的新冠肺炎胸透识别与预测\django\app01\static\pytorch\model\r18_1.pth'
    classes, score = reco_type(img_path=img_path, model_file_path=model)
    print(classes, score)
