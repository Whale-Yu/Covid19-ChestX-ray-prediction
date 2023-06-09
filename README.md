# 基于ResNet的新冠肺炎胸透识别与预测系统


### 🔈介绍

基于ResNet的新冠肺炎胸透识别与预测系统，该系统可以帮助医生更快速准确地诊断新冠肺炎。包括两个部分： 一个图像分类模型和一个Web界面。

在图像分类模型中，我们使用了ResNet18模型对新冠肺炎胸透图像进行分类。我们通过多次训练和验证，创建了一个准确率较高的模型。

在Web界面中，用户可以上传胸透图像，并获得预测结果。我们将图像分类模型集成到Django框架中，为应用添加了一些有用的功能，如预测图片、数据集可视化分析等。

我们希望这个应用可以提高新冠肺炎的诊断速度和准确性，为医疗行业做出贡献。同时，我们也希望这个项目能够探索深度学习技术在医疗领域的应用，为未来的医疗科技发展提供参考。

[点我：demo演示视频](https://www.bilibili.com/video/BV1Gx4y1P74B)


### 📂文件夹说明：
resnet18文件夹为：从数据采集、数据处理、模型搭建、模型训练、模型验证与预测进行开发；

django文件夹为：基于django框架进行Web开发，实现上传图片并进行新冠肺炎胸透的识别与预测；

### 🐖环境依赖、项目目录结构、代码使用说明
详见各子文件夹


### 🐕快速启动预测页面

``conda create -n torch python=3.10``

``conda activate torch``

``pip install -r covid_sort_django/requirements.txt``

``cd covid_sort_django``

``python manage.py runserver``

---

如有问题欢迎提交pr或issue，希望参与到后期维护中
