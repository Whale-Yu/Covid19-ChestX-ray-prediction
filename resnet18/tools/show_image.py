"""
可视化-原始数据集
"""

from torchvision.datasets import ImageFolder
import random
import matplotlib.pyplot as plt

# 加载数据
dataset = ImageFolder("../dataset/image")
sample_num = len(dataset)
print(f'数据集标签类别:{dataset.class_to_idx}')
print(f'数据集总数量:{sample_num}')


pre_num = 16
samples_idx = random.sample(range(0, sample_num), pre_num)
print(f'数据集随机采样{pre_num}张,并可视化')
# 循环采样，保存标签、图片
label_list, img_list = [], []
for idx in samples_idx:
    # 标签
    real_cls1 = dataset.classes[dataset[idx][1]]
    label_list.append(real_cls1)
    # 图片
    img = dataset[idx][0]
    img_list.append(img)
    # plt.imshow(img)
    # plt.show()
print(label_list)
print(img_list)

# 可视化
fig = plt.gcf()
fig.set_size_inches(10, 12)
# plt.rcParams['font.sans-serif'] = ['SimHei']      # 遇见中文或者其他无法显示，出现乱码
for i in range(0, pre_num):
    ax = plt.subplot(4, 4, i + 1)
    ax.imshow(img_list[i])
    title = f'label:{str(label_list[i])}'
    ax.set_title(title, fontsize=10)
    # ax.set_xticks([])  # 不显示坐标轴
    # ax.set_yticks([])
# plt.savefig("show_img.png") # 保存图片
plt.show()
