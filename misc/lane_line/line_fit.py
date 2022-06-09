import os
import os.path as osp
import cv2
import numpy as np
from collections import defaultdict
from ransac_fit import cubic_ransac_curve_fit, quadratic_ransac_curve_fit
from tqdm import tqdm

"""
感知后处理曲线拟合优化：
Input: 点集txt,每一行为 帧数_线id x,y x,y ...
Output: coeff (c0, c1, c2, c3)
"""


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


def ptSamplev2(points, dis=[6, 4, 3], average=False):
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
    if len(new_points) <= 6:
        return ptSample(points, dis[0] / 2)
    return np.array(new_points)


# 曲线拟合
def polyfit(x_arr, y_arr, deg=3):
    # x_arr = np.array([pt[1] for pt in points])
    # y_arr = np.array([pt[0] for pt in points])
    coeff = np.polyfit(x_arr, y_arr, deg)
    return coeff


def vis(points, coeff, showline, inliers=None):
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
            cv2.line(showline, p_s, p_e, color=(0, 255, 0), thickness=1)
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


def vis_ls(points, coeff, showline):
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
            cv2.line(showline, p_s, p_e, color=(0, 0, 255), thickness=1)
            p_s = p_e


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


def pt2dFilter(frames):
    keep_idx = defaultdict(defaultdict)
    for frame, lines in tqdm(frames.items()):
        for line_id, line in lines.items():
            pass


if __name__ == '__main__':
    # vis(0, 0)

    out_root = "debug/iter_100_ths_02_filter_samplev2avg_quadra_cubic_min_samples6"
    # out_ransac = out_root + "_ransac"
    if not osp.exists(out_root):
        os.makedirs(out_root)
    # if not osp.exists(out_ransac):
    #     os.makedirs(out_ransac)
    frames_2d = read_points("points2d_debug.txt")


    frames = defaultdict(defaultdict)
    file = "points3d_raw_debug.txt"
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
            x_arr = np.arange(50, showline_w, 40)
            cv2.putText(showline, f"{dis}", (15, y), cv2.FONT_HERSHEY_COMPLEX, 0.4, color=(192, 192, 192))
            for x in x_arr:
                cv2.line(showline, (x, y), (x + 30, y), color=(192, 192, 192), thickness=1)
        lat_arr = np.arange(-9, 9, 1)
        for dis in lat_arr:
            x = int(-dis * show_w_scl + showline_w / 2)
            y_arr = np.arange(120, showline_h, 40)
            cv2.putText(showline, f"{dis}", (x, 100), cv2.FONT_HERSHEY_COMPLEX, 0.4, color=(192, 192, 192))
            for y in y_arr:
                cv2.line(showline, (x, y), (x, y + 30), color=(192, 192, 192), thickness=1)

        # debug
        if frame == '239':
            debug = 1
            # print(lines)
        # plot curve

        for line_id, line in lines.items():
            least_coeff = polyfit(line[:, 0], line[:, 1])[::-1]
            vis_ls(line, least_coeff, showline)

            f_line = ptFilter(line)
            # f_line = ptSample(f_line, dis=6)
            # print(len(f_line))
            f_line = ptSamplev2(f_line, average=True)
            # print(len(f_line))
            if f_line[-1][0] - f_line[0][0] >= 30:
                coeff, inliers, outliers = cubic_ransac_curve_fit(f_line[:, 0], f_line[:, 1])
            else:
                coeff, inliers, outliers = quadratic_ransac_curve_fit(f_line[:, 0], f_line[:, 1])
            # coeff = polyfit(line[:, 0], line[:, 1])[::-1]  # c0 c1 c2 c3
            if len(coeff) == 3:
                coeff = np.append(coeff, 0)
            p_start = (int(-line[0, 1] * show_w_scl + showline_w / 2), int(showline_h - line[0, 0] * show_h_scl))

            cv2.putText(showline, line_id, p_start, cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (0, 0, 255))
            vis(f_line, coeff, showline, inliers)
        file_name = f"frame_{frame}_.png"
        out_path = osp.join(out_root, file_name)
        cv2.imwrite(out_path, showline)
