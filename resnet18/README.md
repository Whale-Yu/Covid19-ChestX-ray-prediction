# project

项目目录结构 代码使用说明

## 项目目录结构

````
-/resnet18/     根目录
        
    -/dataset/image/        用于存放数据集，具体数据集目录结构见tools/load_dataset.py注释
    
    -/inputs/       用于存放一些输入的东西，比如待预测的图片
    
    -/model/        存放模型训练的结果，比如pth模型，训练过程可视化（acc、loss图），训练日志(log.txt)
    
    -/tools/        一些数据可视化、网络结构可视化、加载数据等一些py文件
        -/load_dataset.py       数据预处理、加载数据集
        -/normlize.py       求自定义数据集的mean和std,用在load_dataset.py
        -/resnet18.py       输出resnet网络结构
        -/show_hand_image.py        可视化预处理后的数据集
        -/show_hand.py      可视化原始数据集
        
    -predict_resnet18.py        预测代码-输入一张图片，返回预测结果
    -requirement.txt    环境依赖
    -train_resnet18.py      训练代码
    -训练记录.txt   记录每一次训练所使用的参数
````

## 代码使用说明


[点我：可视化数据集](tools/show_image.py)

[点我：数据预处理与加载数据集](tools/load_dataset.py)

[点我：可视化预处理后数据集](tools/show_hand_image.py)

[点我：搭建模型](tools/resnet18.py)

[点我：训练代码](train_resnet18.py)

[点我：预测代码](predict_resnet18.py)
