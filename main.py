"""
@author: Huang Wen
@file: main.py
@time: 2024/2/23 19:32
@desc: 
"""
import os
import cv2
import sys
import time
import torch
import numpy as np
from numpy import random
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from main_windows import Ui_MainWindow
from yolo_utils.utils import non_max_suppression, letterbox, scale_coords, plot_one_box
from models.experimental import attempt_load

_translate = QtCore.QCoreApplication.translate
weights_path = None  # 全局变量，存放权重文件的路径
detect_model = None
image_path = ''  # 全局变量，存放待检测文件的路径
camera_flag = False  # 全局变量，存储摄像头的状态，初始为关闭（False）
detect_names = ''
detect_colors = ''
cap = cv2.VideoCapture(0)


def load_weights():
    file_path, file_type = QFileDialog.getOpenFileNames(MainWindow, '选择权重文件', os.getcwd(),
                                                        "PT文件(*.pt);;所有文件(*)")
    if file_path:
        global weights_path, detect_model, detect_names, detect_colors
        weights_path = str(file_path[0])
        device = torch.device('cuda:0')
        try:
            detect_model = attempt_load(weights_path, map_location=device)  # load FP32 model
            detect_names = detect_model.module.names if hasattr(detect_model, 'module') else detect_model.names
            detect_colors = [[random.randint(0, 255) for _ in range(3)] for _ in detect_names]
        except Exception as e:
            print(str(e))
        out_text('已加载权重：'+weights_path)


def open_file():
    global image_path, cap
    file_path, _ = QFileDialog.getOpenFileNames(MainWindow, '选择需要检测的图像或视频', os.getcwd(),
                                                        "所有文件(*);;图像文件(*.jpg;*.jpeg;*.png);;视频文件(*.mp4;*.avi)")
    if file_path:
        if not weights_path:
            out_text('文件加载失败，请先加载权重文件！')
            return
        file_type = file_path[0].split('.')[1]
        image_path = str(file_path[0])
        if file_type in ['mp4', 'avi']:
            try:
                cap = cv2.VideoCapture(image_path)
                if cap.isOpened():
                    out_text('已加载视频：' + image_path)
                    timer_pic.start(5)
                else:
                    out_text('视频打开失败！')
            except Exception as e:
                ui.text_print.append('Exception:'+str(e))

        else:
            try:
                cap.release()  # 如果有正在播放的视频，释放掉
                img = cv2.imread(image_path)
                out_text('已加载图像：' + image_path)
                cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 视频流的长和宽
                height, width = cur_frame.shape[:2]
                pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(pixmap)
                # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
                ratio = max(width / ui.show_raw.width(), height / ui.show_raw.height())
                pixmap.setDevicePixelRatio(ratio)
                # 视频流置于label中间部分播放
                ui.show_raw.setAlignment(Qt.AlignCenter)
                ui.show_raw.setPixmap(pixmap)
                if weights_path:  # 如果加载了权重文件，则执行检测功能
                    my_detect(img)  # 调用检测模型
                    cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # 视频流的长和宽
                    height, width = cur_frame.shape[:2]
                    pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(pixmap)
                    # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
                    ratio = max(width / ui.show_result.width(), height / ui.show_result.height())
                    pixmap.setDevicePixelRatio(ratio)
                    # 视频流置于label中间部分播放
                    ui.show_result.setAlignment(Qt.AlignCenter)
                    ui.show_result.setPixmap(pixmap)
            except Exception as e:
                timestamp = '[' + time.strftime("%Y/%m/%d-%H:%M") + ']'
                ui.text_print.append(timestamp+'Exception:'+str(e))


def open_camera():
    global camera_flag, weights_path, cap
    if not camera_flag:
        if not weights_path:
            out_text('摄像头打开失败，请先加载权重文件！')
            return
        # 打开摄像头
        camera_flag = True
        cap = cv2.VideoCapture(0)
        ui.button_camera.setText(_translate("MainWindow", "关闭摄像头"))
        out_text('摄像头已打开')
        timer_pic.start(5)
    else:
        # 关闭摄像头
        camera_flag = False
        timer_pic.stop()
        cap.release()
        ui.button_camera.setText(_translate("MainWindow", "打开摄像头"))
        ui.show_raw.setText(_translate("MainWindow", "原始数据"))
        ui.show_result.setText(_translate("MainWindow", "检测结果"))
        out_text('摄像头已关闭')

def clean():
    global cap, camera_flag
    camera_flag = False
    cap.release()
    timer_pic.stop()
    ui.show_raw.setText(_translate("MainWindow", "原始数据"))
    ui.show_result.setText(_translate("MainWindow", "检测结果"))
    ui.text_print.setHtml(_translate("MainWindow", " "))
    ui.text_detect.setHtml(_translate("MainWindow", " "))
    ui.button_camera.setText(_translate("MainWindow", "打开摄像头"))


def out_text(text):  # 输出内容至“打印输出”区域
    timestamp = '[' + time.strftime("%Y/%m/%d-%H:%M") + ']'
    ui.text_print.append(timestamp + text)


def my_detect(img):
    with torch.no_grad():
        image, ratio, dwdh = letterbox(img, auto=False)
        image = image.transpose((2, 0, 1))[::-1]
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = torch.from_numpy(image).float()
        im /= 255
        try:
            im_gpu = im.cuda()  # 将数据拷贝至GPU
            result = detect_model(im_gpu)[0]
        except Exception as e:
            timestamp = '[' + time.strftime("%Y/%m/%d-%H:%M") + ']'
            ui.text_print.append(timestamp+'推理出错：' + str(e))
        result = non_max_suppression(result, 0.5, 0.65)[0]
        result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img.shape)
        count = {}
        text = ''
        for *xyxy, conf, cls in result:
            if detect_names[int(cls)] in count:
                count[detect_names[int(cls)]] += 1
            else:
                count[detect_names[int(cls)]] = 1  # 初始化计数
            label = f'{detect_names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img, label=label, color=detect_colors[int(cls)], line_thickness=3)
        for i in count:
            text += str(count[i])+' '+i+', '
        ui.text_detect.append('检测结果：'+text)


def show_pic():
    # 执行检测功能
    global detect_model, cap, detect_names, detect_colors
    try:
        ret, img = cap.read()
        if ret:
            cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 视频流的长和宽
            height, width = cur_frame.shape[:2]
            pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
            ratio = max(width / ui.show_result.width(), height / ui.show_result.height())
            pixmap.setDevicePixelRatio(ratio)
            # 视频流置于label中间部分播放
            ui.show_raw.setAlignment(Qt.AlignCenter)
            ui.show_raw.setPixmap(pixmap)
            # 对每一帧进行检测
            with torch.no_grad():
                my_detect(img)   # 调用检测模型

            cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 视频流的长和宽
            height, width = cur_frame.shape[:2]
            pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
            ratio = max(width / ui.show_result.width(), height / ui.show_result.height())
            pixmap.setDevicePixelRatio(ratio)
            # 视频流置于label中间部分播放
            ui.show_result.setAlignment(Qt.AlignCenter)
            ui.show_result.setPixmap(pixmap)
    except Exception as e:
        ui.text_print.append(e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()  # 显示主界面
    ui.button_weights.clicked.connect(load_weights)  # 将按钮button_weights绑定至函数load_weights
    ui.button_file.clicked.connect(open_file)  # 将按钮button_file绑定至函数open_file
    ui.button_camera.clicked.connect(open_camera)  # 将按钮button_camera绑定至函数open_camera
    ui.button_clean.clicked.connect(clean)  # 将按钮button_clean绑定至函数clean
    timer_pic = QTimer()
    timer_pic.timeout.connect(show_pic)
    sys.exit(app.exec_())
