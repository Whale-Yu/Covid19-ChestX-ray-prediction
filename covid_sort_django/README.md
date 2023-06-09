# project

介绍 环境依赖 项目目录结构 代码使用说明

## 介绍
简单来说，就是使用django框架来实现可视化，实现上传图片并一键预测；

前端页面直接使用模板、预测模型是resnet18训练出来的；

总结说，这块开发就是将预测模型部署到前端页面，并实现上传、一键识别等功能；

难点主要在实现上传、一键识别的逻辑设计，配合前端页面并进行微调

## 环境依赖
详见requirement.txt

 
## 项目结构目录

### 一、covid_sort_django文件夹
当前django项目名为covid_sort_django，所以文件夹与其同名;

目录下包括：

1）settings.py 是django项目的一些配置，需要进行一些设置。

2）urls是django的总的路由，用来指向以后的app应用。

3）wsgi是django的wsgi接口，开发时不用管


### 二、app01文件夹

新建一个app01的命令
``python manager.py startapp app01``

app01是app名称，随便起

一个项目基本上是每开发一个模块就要新建一个app应用，是为了避免代码混乱

app文件目录下包括：

**1）migrations**:数据库迁移文件

**2）static**：

assets:存放一些前端文件(css、img、js、vender)，来自前端模板，

media:临时存放上传的图片；

pytorch：预测模型文件，包括pth模型文件，py预测文件
、临时存储、预测模型文件夹，其中


**3）templates**：存放前端页面html文件，所以其他几个html均套用templates.html模板

**4）admin.py：** 一般不管

**5）apps.py：**  一般不管

**6）models.py：** 一般不管

**7）views.py：** 函数处理文件，功能主要在此开发

### 三、manage.py是包含django的所有命令，以后有用

用来运行



### 四、参考资料
大白话讲django之django的目录结构
:https://zhuanlan.zhihu.com/p/52153427

## 代码使用说明
在功能开发完成好的前提下，在manage.py所在目录下，终端中输入命令：

``python .\manage.py runserver``

之后点击终端中的：http://127.0.0.1:8000/


即可显示Web页面 
