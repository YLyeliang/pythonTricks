# -*- coding: utf-8 -*-
# @Time : 2021/9/6 上午11:12
# @Author: yl
# @File: nms.py
import numpy as np


def nms(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0]  # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    rets = dets[keep]
    return rets, keep


if __name__ == '__main__':
    boxes = np.array([[49.1, 32.4, 51.0, 35.9],
                      [49.3, 32.9, 51.0, 35.3],
                      [49.2, 31.8, 51.0, 35.4],
                      [35.1, 11.5, 39.1, 15.7],
                      [35.6, 11.8, 39.3, 14.2],
                      [35.3, 11.5, 39.9, 14.5],
                      [35.2, 11.7, 39.7, 15.7]], dtype=np.float32)
    scores = np.array([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3], dtype=np.float32)
    iou_threshold = 0.6
    dets, inds = nms(boxes, scores, iou_threshold)
    debug = 1
