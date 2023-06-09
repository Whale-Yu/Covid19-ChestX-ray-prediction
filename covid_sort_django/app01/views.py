from django.shortcuts import render, HttpResponse


from .models import Img

from covid_sort_django.settings import MEDIA_ROOT
from django.contrib import messages

from matplotlib import pyplot as plt
from PIL import Image

from app01.static.pytorch.reco_type import reco_type


# Create your views here.


def home(request):
    # 处理一打开页面就显示首页
    return render(request, 'index.html')


def pre_defualt():
    # 预测一张默认图片作为效果展示
    classes, score = reco_type(
        img_path=r'D:\PycharmProject(D)\外包项目合集\23-3-15-基于数据挖掘的新冠肺炎胸透识别与预测\covid_sort_django\app01\static\assets\img\COVID(936)-default.png',
        model_file_path=r'D:\PycharmProject(D)\外包项目合集\23-3-15-基于数据挖掘的新冠肺炎胸透识别与预测\covid_sort_django\app01\static\pytorch\model\r18_1.pth')
    return classes, score


# 首页
def index(request):
    return render(request, 'index.html')


# 功能介绍
def intro(request):
    return render(request, 'intro.html')


# 本地上传
def upload(request):
    # 如果是GET请求，预测一张默认图并显示预测结果
    if request.method == 'GET':
        classes, score = pre_defualt()  # 预测一张默认图片
        print(classes, score)
        return render(request, 'upload_defualt.html', {'classes': classes, 'score': score})
    # 如果是POST请求，获取用户提交的数据
    else:
        img = request.POST.get('img')
        print(img)
        # 未选择图片后，点击上传并识别——警告出错（以下三种警告均可，前两种警告没有交互性）
        if img == '':
            # 警告提示方式一
            # return HttpResponse('失败（未选择图片）')
            # 警告提示方式二
            # return render(request, 'upload.html', {'error_msg': '上传失败（或未选择文件）'})
            # 警告提示方式三（资料：https://blog.csdn.net/qq_15158911/article/details/95976790）
            messages.error(request, '上传失败(未选择文件)')
            classes, score = pre_defualt()  # 预测一张默认图片
            return render(request, 'upload_defualt.html', {'classes': classes, 'score': score})
        else:
            img1 = Img(img_url=request.FILES.get('img'))
            img_path = str(img1.img_url)
            print(f"图片名称:{img_path}")
            img_all_path = ((MEDIA_ROOT).replace('\\', '/') + '/img/' + f'{img_path}').replace('\\', '/')
            print(f"图片路径:{img_all_path}")

            # 图片保存至本地 /static/media/img
            img1.save()

            # 解决bug：如果路径中存在其他非法字符会报错，如()或者空格都会出现错误
            try:
                # 返回预测类别和置信度
                classes, score = reco_type(img_path=img_all_path,
                                           model_file_path=r'D:\PycharmProject(D)\外包项目合集\23-3-15-基于数据挖掘的新冠肺炎胸透识别与预测\covid_sort_django\app01\static\pytorch\model\r18_1.pth')
                print(classes, score)

                return render(request, 'upload.html',
                              {'img_path': img_path, 'img': img1, 'classes': classes, 'score': score})
            except:
                messages.error(request, '图片名称存在非法字符（仅可使用汉字、字母、数字)')
                classes, score = pre_defualt()  # 预测一张默认图片
                return render(request, 'upload_defualt.html', {'classes': classes, 'score': score})


# 数据集
def data(request):
    return render(request, 'data.html')


# 团队成员
def team(request):
    return render(request, 'team.html')
