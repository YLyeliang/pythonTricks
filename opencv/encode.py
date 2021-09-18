# -*- coding: utf-8 -*-
# @Time : 2021/9/8 上午9:53
# @Author: yl
# @File: encode.py

# 图像编解码
import cv2
import os
import numpy as np


def from_bytes():
    bytes = 0
    img = np.array(img_bytes).tostring()
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, 1)
    debug = 1


import base64

if __name__ == '__main__':
    root = '../../ylocr/data/recovery_color'
    imgs = os.listdir(root)
    base64_list = []
    for file in imgs:
        img = os.path.join(root, file)
        img = cv2.imread(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        ratio = 120 / min(h, w)
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        img_gray = cv2.resize(img_gray, (new_w, new_h))
        img = cv2.resize(img, (150, 150))
        img_list = img.reshape(-1)
        img_list = list(img_list)
        length = len(str(img_list))
        img_bytes = cv2.imencode('.jpg', img_gray)[1]
        img_string = np.array(img_bytes).tobytes()
        base64_string = base64.b64encode(img_string)
        dd = base64.b64decode(base64_string)
        print(img_string)
        print(len(img_string))
        print(base64_string)
        print(len(base64_string))
        # from_bytes()
        base64_list.append(base64_string)
        debug = 1
    print(base64_list)
