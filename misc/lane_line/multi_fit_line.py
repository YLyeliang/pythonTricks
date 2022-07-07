import os
import os.path as osp
import cv2
import numpy as np
from collections import defaultdict
from ransac_fit import cubic_ransac_curve_fit, quadratic_ransac_curve_fit, linear_regression_regularization, get_score
from tqdm import tqdm
from copy import deepcopy
from utils import *

"""
感知后处理曲线拟合优化：
Input: 点集txt,每一行为 帧数_线id x,y x,y ...
Output: coeff (c0, c1, c2, c3)
"""
color_map = ((0, 255, 0), (255, 0, 0), (128, 128, 0), (255, 255, 0), (128, 258, 0), (258, 128, 0))


# 点过滤
def ptFilter(points):
    # version 0.2
    new_points = []
    for i, pt in enumerate(points):
        if i == 0:
            pre = pt[0]
        else:
            if pt[0] <= 120 and pt[0] - pre <= 20:
                new_points.append(pt)
                pre = pt[0]
    # points = [pt for pt in points if pt[0] <= 120]   # version 0.1
    new_points = np.array(new_points)
    return new_points


# 点采样
def ptSample(points, dis=3):
    """
    version 0.1:
    sample point data uniformly
    Args:
        points:
    """
    new_points = []
    start_x = points[0][0]
    end_x = points[-1][0]

    # if len(points) <= 30:
    #     dis = 3

    # if end_x - start_x <= 30:
    #     return points

    for i, pt in enumerate(points):
        if i == 0:
            last_p = pt[0]
            new_points.append(pt)
        else:
            if pt[0] - last_p >= dis:
                new_points.append(pt)
                last_p = pt[0]
    while len(new_points) < 6:
        dis /= 2
        new_points = []
        for i, pt in enumerate(points):
            if i == 0:
                last_p = pt[0]
                new_points.append(pt)
            else:
                if pt[0] - last_p >= dis:
                    new_points.append(pt)
                    last_p = pt[0]

    new_points = np.array(new_points)
    return new_points


def ptSamplev2(points, dis=[3, 2, 1], average=False):
    """
    version 0.2:
    采样时近处距离大，远处距离小。保证近处密集点变稀疏，中远处的点尽量不变。
    0-20, 20-40, 40-...
    average: 是否对距离内的所有点取均值
    """
    new_points = []
    for i, pt in enumerate(points):
        if i == 0:
            last_p = pt[0]
            last_i = i
            # new_points.append()
        else:
            if pt[0] < 20:  # 近处的点
                if pt[0] - last_p >= dis[0]:  # 间隔大一些采样
                    if average:
                        avg_x = np.average(points[last_i:i + 1, 0])
                        avg_y = np.average(points[last_i:i + 1, 1])
                        avg_p = [avg_x, avg_y]
                        new_points.append(avg_p)
                    else:
                        new_points.append(points[i])

                    last_i = i
                    last_p = pt[0]
            elif pt[0] < 40:
                if pt[0] - last_p >= dis[1]:
                    if average:
                        avg_x = np.average(points[last_i:i + 1, 0])
                        avg_y = np.average(points[last_i:i + 1, 1])
                        avg_p = [avg_x, avg_y]
                        new_points.append(avg_p)
                    else:
                        new_points.append(points[i])
                    last_i = i
                    last_p = pt[0]
            else:
                if pt[0] - last_p >= dis[2]:
                    # if average:
                    #     avg_x = np.average(points[last_i:i + 1, 0])
                    #     avg_y = np.average(points[last_i:i + 1, 1])
                    #     avg_p = [avg_x, avg_y]
                    #     new_points.append(avg_p)
                    # else:
                    new_points.append(points[i])
                    last_i = i
                    last_p = pt[0]
    if len(new_points) <= 8:
        return ptSample(points, dis[0] / 2)
    return np.array(new_points)


# 曲线拟合
def polyfit(x_arr, y_arr, deg=3, w=None, regression=False):
    # x_arr = np.array([pt[1] for pt in points])
    # y_arr = np.array([pt[0] for pt in points])
    coeff = np.polyfit(x_arr, y_arr, deg, w=w)[::-1]
    return coeff


def vis(points, coeff, showline, color=(0, 255, 0), inliers=None, show_coeff=True, index=0):
    """
    Visualize the points and fitted curve
    Args:
        points(np.array):  set of (x,y) x is longitudinal, y is lateral
        coeff(list|np.array): coeff of curve
    """
    showline_w = 600
    showline_h = 640
    show_w_scl = 30
    show_h_scl = 5

    # plot fitted line
    last_x = points[-1][0]
    last_x = max(last_x, min(last_x + 20, 80))
    start_x = points[0][0]
    x_space = np.linspace(start_x, last_x, 50)  # x here represents longitudinal
    for i, p in enumerate(x_space):
        x = coeff[0] + p * coeff[1] + p ** 2 * coeff[2] + p ** 3 * coeff[3]  # x here represents x coordinates of image
        x = int(-(x * show_w_scl) + showline_w / 2)
        y = int(showline_h - p * show_h_scl)
        if i == 0:
            p_s = (x, y)
        else:
            p_e = (x, y)
            cv2.line(showline, p_s, p_e, color=color, thickness=1)
            p_s = p_e

    # plot raw points
    for i, p in enumerate(points):  # x,y
        y = int(showline_h - p[0] * show_h_scl)
        center = (int(-(show_w_scl * p[1]) + showline_w / 2), y)
        if inliers is None:
            cv2.circle(showline, center, 2, (255, 255, 255), -1)
        elif not inliers[i]:
            cv2.circle(showline, center, 2, (0, 0, 255), -1)
        else:
            cv2.circle(showline, center, 2, (255, 255, 255), -1)

    if show_coeff:
        mid = points[int(len(points) / 2)]
        center = (int(-(show_w_scl * mid[1]) + showline_w / 2), int(showline_h - mid[0] * show_h_scl))
        x = center[0]
        y = center[1]
        for i, coe in enumerate(coeff):
            cv2.putText(showline, f"c{i}:{coe:.2e}", (x, y + i * 15 + index * 30), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        color=(0, 255, 0))


def vis_ls(points, coeff, showline, color=(0, 0, 255), coeff_color=(255, 128, 128), show_coeff=False, index=0):
    """
    show LS results
    """
    showline_w = 600
    showline_h = 640
    show_w_scl = 30
    show_h_scl = 5

    # plot fitted line
    last_x = points[-1][0]
    last_x = max(last_x, min(last_x + 20, 80))
    start_x = points[0][0]
    x_space = np.linspace(start_x, last_x, 50)  # x here represents longitudinal
    for i, p in enumerate(x_space):
        x = coeff[0] + p * coeff[1] + p ** 2 * coeff[2] + p ** 3 * coeff[3]  # x here represents x coordinates of image
        x = int(-(x * show_w_scl) + showline_w / 2)
        y = int(showline_h - p * show_h_scl)
        if i == 0:
            p_s = (x, y)
        else:
            p_e = (x, y)
            cv2.line(showline, p_s, p_e, color=color, thickness=1)
            p_s = p_e

    if show_coeff:
        mid = points[int(len(points) / 2)]
        center = (int(-(show_w_scl * mid[1]) + showline_w / 2), int(showline_h - mid[0] * show_h_scl))
        x = center[0]
        y = center[1]
        for i, coe in enumerate(coeff):
            cv2.putText(showline, f"c{i}:{coe:.2e}", (x, y + i * 15 - 300 + index * 30), cv2.FONT_HERSHEY_COMPLEX,
                        0.4,
                        color=coeff_color)


def read_points(file):
    with open(file, "r") as f:
        lines = f.readlines()
    frames_2d = defaultdict(defaultdict)
    for line in lines:
        line = line.rstrip('\n').strip()
        arr = line.split(" ")
        frame, line_id = arr[0].split("_")
        points = arr[1:]
        points = [[int(p) for p in pt.split(',')] for pt in points]
        points = np.array(points)  # x y
        frames_2d[frame][line_id] = points
    return frames_2d


def pt2dFilter(frames, a=3):
    """
    version 0.1:
    Filter 2d points that apparently far from cluster
    维护长度为3的数组，保存前三个两两相邻点的距离.某一时刻如果点的距离与之前点相差较大，可能为异常点。

    """
    keep_idx = defaultdict(defaultdict)
    for frame, lines in tqdm(frames.items()):
        for line_id, line in lines.items():
            keep = [True] * 4
            dis_vec = list(line[1:4] - line[:3])
            dis = [np.sqrt(a ** 2 + b ** 2) for a, b in dis_vec]

            pre = line[3]
            tmp = deepcopy(dis)  # for checking whole distance
            for i, pt in enumerate(line[4:]):
                cur_dis = pt - pre
                dis_vec.append(cur_dis)
                cur_dis = np.sqrt(cur_dis[0] ** 2 + cur_dis[1] ** 2)
                tmp.append(cur_dis)
                if cur_dis > a * max(dis):
                    keep.append(False)
                    pre = pre + dis_vec[-2]
                else:
                    keep.append(True)
                    pre = pt
                dis = dis[1:] + [cur_dis]
            keep_idx[frame][line_id] = keep
    return keep_idx


def pt2dFilterv2(frames):
    """
    对2D进行拟合，使用拟合后的线对远点做离群点判断
    """
    keep_idx = defaultdict(defaultdict)
    for frame, lines in tqdm(frames.items()):
        for line_id, line in lines.items():
            keep = [True] * len(line)
            sparse_line = np.concatenate([line[:-1:4], line[-5:]])
            coeff = polyfit(sparse_line[:, 0], sparse_line[:, 1], 2)
            # 对后5个点做离群点判断
            last_points = line[-5:]
            for i, pt in enumerate(last_points):
                if abs(quadratic_pred(pt[0], coeff) - pt[1]) > 10:
                    keep[i - 5] = False
            keep_idx[frame][line_id] = keep

    return keep_idx


def pd2dFilterv4(frames, a=3, b=1):
    keep_idx = defaultdict(defaultdict)
    for frame, lines in tqdm(frames.items()):
        for line_id, line in lines.items():
            direc_vec = line[1:] - line[:-1]  # 方向向量
            lat_vec = list(direc_vec[:, 0])
            lat_vec_abs = [abs(_) for _ in lat_vec]
            slop_vec = [pt[1] / pt[0] if pt[0] != 0 else 100 for pt in direc_vec]
            keep = [True] * 4
            dis_vec = list(line[1:4] - line[:3])
            dis = [np.sqrt(a ** 2 + b ** 2) for a, b in dis_vec]

            pre = line[3]
            for i, pt in enumerate(line[4:]):

                cur_dis = pt - pre
                dis_vec.append(cur_dis)
                cur_dis = np.sqrt(cur_dis[0] ** 2 + cur_dis[1] ** 2)
                if cur_dis > a * max(dis):
                    keep.append(False)
                    pre = pre + dis_vec[-2]
                else:
                    keep.append(True)
                    pre = pt
                dis = dis[1:] + [cur_dis]
            keep_idx[frame][line_id] = keep
    return keep_idx


def pt2dFilterv3(frames, a=3, b=1):
    """
    滑窗版本 version 0.2:
    维护滑动窗口，滑动窗口保存前k个点对的距离
    1. 每次bottom-up向上滑动一个点，计算当前点对的距离
    2. 如果当前点对的距离大于窗口中任何一个的a倍，则说明该点不符合点的距离增长规律，列为疑似离群点
    3. 构建点的向量数组，每个元素表示上个点与当前点的x轴方向距离，整个数组应该满足条件
        1) 直线：元素始终为正或者始终为负
        2) 右转弯： 当前车道左侧始终为正； 当前车道右侧始终为负 or 远端由负转正
        - 如果某一点对不符合上述规律，并且具有较大的偏差，则可能为噪点，考虑将其移除
    Process:
    1. 获取点对的横向距离数组
    2. 获取点对的斜率
    3. 自底向上遍历数组
        - 横向条件：如果当前|点对的横向距离| > max(|前三个点对横向距离|) * a
        - 纵向条件：如果当前|点对的纵向距离| > max(|前三个点对纵向距离|) * b
        - 斜率条件：如果当前|点对的斜率|
        1) 如果当前点对的横向距离<0
            - 查看前三个点对的横向距离
                - if all < 0, 判断同向距离 cur_dis < a * min(xxx)为噪点
                - if all > 0 or if -++, 可能为转折点. 横向距离不应该过长，abs(cur_dis) > b * min(abs(xxx))
        2) 如果当前点对的横向距离 > 0
            与条件1）性质一致
        - 对于剔除的噪点，保留剔除的索引，同时更新
            - 横向距离数组、斜率：沿用上一个点对的数值
    """
    keep_idx = defaultdict(defaultdict)

    for frame, lines in tqdm(frames.items()):
        for line_id, line in lines.items():
            direc_vec = line[1:] - line[:-1]  # 方向向量
            lat_vec = list(direc_vec[:, 0])
            lat_vec_abs = [abs(_) for _ in lat_vec]
            slop_vec = [pt[1] / pt[0] if pt[0] != 0 else 100 for pt in direc_vec]
            keep = [True] * 4
            dis_vec = list(line[1:4] - line[:3])
            dis = [np.sqrt(a ** 2 + b ** 2) for a, b in dis_vec]

            pre = line[3]
            # tmp = deepcopy(dis)  # for checking whole distance
            for i, pt in enumerate(line[4:]):  # start from 5th point
                # line[i] relates to lat_vec[i+3]
                cur_lat = lat_vec[i + 3]  # 当前点的横向距离
                pre_lat = lat_vec[i:i + 3]  # 上三个点的横向距离
                # if cur_lat > max(pre_lat):  # 比之前的三个都大
                noise_tag = False
                # if cur_lat < 0:
                #     if pre_lat[0] < 0 and pre_lat[1] < 0 and pre_lat[2] < 0 and cur_lat < a * min(
                #             pre_lat):  # 噪点:  # all < 0
                #         noise_tag = True
                #     elif pre_lat[0] >= 0 and pre_lat[1] >= 0 and pre_lat[2] >= 0 and abs(cur_lat) > b * min(pre_lat):
                #         noise_tag = True
                #     elif pre_lat[0] < 0 and pre_lat[2] >= 0 and abs(cur_lat) > b * min(
                #             [abs(_) for _ in pre_lat]):  # 转折
                #         noise_tag = True
                # else:
                #     if pre_lat[0] <= 0 and pre_lat[1] <= 0 and pre_lat[2] <= 0 and cur_lat > b * min(
                #             [abs(_) for _ in pre_lat]):  # 转折点:  # all <=0 cur >0
                #         noise_tag = True
                #     elif pre_lat[0] > 0 and pre_lat[1] > 0 and pre_lat[2] > 0 and cur_lat > a * min(
                #             pre_lat):
                #         noise_tag = True
                #     elif pre_lat[0] >= 0 and pre_lat[2] < 0 and cur_lat > b * min(
                #             [abs(_) for _ in pre_lat]):  # 转折
                #         noise_tag = True
                if noise_tag:
                    keep.append(False)
                    lat_vec[i + 3] = pre_lat[-1]
                else:
                    keep.append(True)

            keep_idx[frame][line_id] = keep
    return keep_idx


def vis_points2d(points, keep_idx, img_show):
    color = color_map[np.random.randint(0, len(color_map))]
    for i, pt in enumerate(points):
        if keep_idx[i]:
            cv2.circle(img_show, pt, 2, color, -1)
        else:
            cv2.circle(img_show, pt, 2, (0, 0, 255), -1)


def get_line_weights_by_dis(points):
    sample_ls_w = []
    weights = [0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6]  # from 0 to 100

    for w_i, pt in enumerate(points):
        index = int(pt[0] / 10)
        if index > len(weights) - 1:
            index = 1
        sample_ls_w.append(weights[index])
    return sample_ls_w


if __name__ == '__main__':
    # vis(0, 0)

    # out_root = "debug/iter_100_ths_02_filter_samplev2avg_quadra_cubic_min_samples6_2dfilterv2"
    out_root = "debug/iter_30_roi_samples431_6pt_multi_line_fit"
    if not osp.exists(out_root):
        os.makedirs(out_root)
    # frames_2d = read_points("points2d_debug.txt")
    frames_2d = read_points("/home/yel/Projects/zdrive/debug/orin_ori/points2d_debug.txt")

    frames_2d_keep = pt2dFilter(frames_2d)
    # frames_2d_keep = pt2dFilterv2(frames_2d)
    # frames_2d_keep = pt2dFilterv3(frames_2d)

    # vis points 2d
    # for frame, lines in tqdm(frames_2d.items()):
    #     img_show = np.zeros((640, 1280, 3), dtype=np.uint8)
    #     keep_lines = frames_2d_keep[frame]
    #     for line_id, line in lines.items():
    #         keep_line = keep_lines[line_id]
    #         vis_points2d(line, keep_line, img_show)
    #     if not osp.exists(out_root + "_2d"):
    #         os.makedirs(out_root + "_2d")
    #     out_path = osp.join(out_root + "_2d", f"frame_{frame}_2d.png")
    #     cv2.imwrite(out_path, img_show)

    # exit(-1)
    frame_1116 = frames_2d["1116"]

    # exit(1)
    frames = defaultdict(defaultdict)
    file = "points3d_raw_debug.txt"
    file = "/home/yel/Projects/zdrive/debug/orin_ori/points3d_raw_debug.txt"
    with open(file, 'r') as f:
        lines = f.readlines()

    # 横轴，纵轴
    for line in lines:
        line = line.rstrip('\n').strip()
        arr = line.split(" ")
        frame, line_id = arr[0].split("_")
        points = arr[1:]
        points = [[float(p) for p in pt.split(',')] for pt in points]
        points = np.array(points)
        points = points[:, ::-1]  # x: longitudinal y: lateral
        frames[frame][line_id] = points

    for frame, lines in tqdm(frames.items()):
        showline_w = 600
        showline_h = 640
        show_w_scl = 30
        show_h_scl = 5
        showline = np.zeros((showline_h, showline_w, 3), dtype=np.uint8)

        cv2.putText(showline, "det", (15, 24), cv2.FONT_HERSHEY_COMPLEX, 0.4, color=(0, 255, 0))
        cv2.putText(showline, "orin_ls", (15, 48), cv2.FONT_HERSHEY_COMPLEX, 0.4, color=(0, 0, 255))
        cv2.putText(showline, f"frame_{frame}", (15, 72), cv2.FONT_HERSHEY_COMPLEX, 0.6, color=(192, 25, 192))

        # initialize distance grid
        dis_arr = np.arange(10, 110, 10)
        for dis in dis_arr:
            y = showline_h - dis * show_h_scl
            # x_arr = np.arange(50, showline_w, 40)
            cv2.putText(showline, f"{dis}", (15, y), cv2.FONT_HERSHEY_COMPLEX, 0.4, color=(192, 192, 192))
            cv2.line(showline, (50, y), (showline_w, y), color=(192, 192, 192), thickness=1)
            # for x in x_arr:
            #     cv2.line(showline, (x, y), (x + 30, y), color=(192, 192, 192), thickness=1)
        lat_arr = np.arange(-9, 9, 1)
        for dis in lat_arr:
            x = int(-dis * show_w_scl + showline_w / 2)
            # y_arr = np.arange(120, showline_h, 40)
            cv2.putText(showline, f"{dis}", (x, 100), cv2.FONT_HERSHEY_COMPLEX, 0.4, color=(192, 192, 192))
            cv2.line(showline, (x, 120), (x, showline_h), color=(192, 192, 192), thickness=1)
            # for y in y_arr:
            #     cv2.line(showline, (x, y), (x, y + 30), color=(192, 192, 192), thickness=1)

        # debug
        if frame == '239':
            debug = 1
            # print(lines)
        # plot curve
        index = 0
        for line_id, line in lines.items():
            f_line = ptFilter(line)
            # f_line = ptSample(f_line, dis=6)
            f_line = ptSamplev2(f_line, dis=[4, 3, 2], average=False)
            cubic_model, cubic_coeff = linear_regression_regularization(f_line[:, 0], f_line[:, 1])
            quadra_model, quadra_coeff = linear_regression_regularization(f_line[:, 0], f_line[:, 1], degree=2)
            linear_model, linear_coeff = linear_regression_regularization(f_line[:, 0], f_line[:, 1], degree=1)
            cubic_score = get_score(cubic_model, f_line[:, 0], f_line[:, 1], degree=3)
            quadra_score = get_score(quadra_model, f_line[:, 0], f_line[:, 1], degree=2)
            linear_score = get_score(linear_model, f_line[:, 0], f_line[:, 1], degree=1)
            scores = [linear_score, quadra_score, cubic_score]
            coeffs = [linear_coeff, quadra_coeff, cubic_coeff]
            model_index = np.argmax(scores)
            coeff = coeffs[model_index]
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            for i, co in enumerate(coeffs):
                vis(f_line, co, showline, colors[i], None, False, index=index)
                cv2.putText(showline, f"curve: {model_index + 1} s: {scores[i]}",
                            (x, y + index * 30 + i * 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, colors[i])

            mid = f_line[int(len(f_line) / 2)]
            center = (int(-(show_w_scl * mid[1]) + showline_w / 2), int(showline_h - mid[0] * show_h_scl))
            x = center[0]
            y = center[1]

            #
            # if f_line[-1][0] - f_line[0][0] >= 30:
            #     coeff, inliers, outliers = cubic_ransac_curve_fit(f_line[:, 0], f_line[:, 1], ridge=True, lasso=False)
            # else:
            #     coeff, inliers, outliers = quadratic_ransac_curve_fit(f_line[:, 0], f_line[:, 1])

            # sample_ls_w = get_line_weights_by_dis(f_line)
            # sample_ls_coeff = polyfit(f_line[:, 0], f_line[:, 1], w=None)  # c0 c1 c2 c3
            # vis_ls(line, sample_ls_coeff, showline, color=(255, 0, 0), coeff_color=(255, 128, 128), show_coeff=True,
            #        index=index)

            # 采样后ls+regularization
            # sample_ls_coeff_regular = linear_regression_regularization(f_line[:, 0], f_line[:, 1])
            # vis_ls(line, sample_ls_coeff_regular, showline, color=(0, 0, 255), coeff_color=(0, 0, 255), show_coeff=True,
            #        index=index + 5)

            # inliers = [True] * len(f_line)
            # if len(coeff) == 3:
            #     coeff = np.append(coeff, 0)
            p_start = (int(-line[0, 1] * show_w_scl + showline_w / 2), int(showline_h - line[0, 0] * show_h_scl))

            cv2.putText(showline, line_id, p_start, cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (0, 0, 255))
            # vis(f_line, coeff, showline, inliers, show_coeff=False, index=index)

            index += 1
        file_name = f"frame_{frame}_.png"
        out_path = osp.join(out_root, file_name)
        cv2.imwrite(out_path, showline)
