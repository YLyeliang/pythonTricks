import os
import os.path as osp
import cv2
import numpy as np
from collections import defaultdict
from ransac_fit import cubic_ransac_curve_fit
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
            if pt[0] <= 120 or pt[0] - pre > 20:
                new_points.append(pt)
                pre = pt[0]
    # points = [pt for pt in points if pt[0] <= 120]   # version 0.1
    points = np.array(points)
    return points


# 点采样
def ptSample(points):
    pass


# 曲线拟合
def polyfit(x_arr, y_arr, deg=3):
    # x_arr = np.array([pt[1] for pt in points])
    # y_arr = np.array([pt[0] for pt in points])
    coeff = np.polyfit(x_arr, y_arr, 3)
    return coeff


def vis(points, coeff, showline):
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
    # showline = np.zeros((showline_w, showline_h), dtype=np.uint8)
    # cv2.putText(showline, "-det", (15, 24), cv2.FONT_HERSHEY_COMPLEX, 0.4, color=(0, 255, 0))

    # cv2.imshow("debug", showline)
    # cv2.waitKey()
    # plot fitted line and raw points
    for i, p in enumerate(points):  # x,y
        x = coeff[0] + p[0] * coeff[1] + p[0] ** 2 * coeff[2] + p[0] ** 3 * coeff[3]  # y belong to lateral
        x = -int(x * show_w_scl) + showline_w / 2
        y = int(showline_h - p[0] * show_h_scl)
        if i == 0:
            p_s = (x, y)
        else:
            p_e = (x, y)
            cv2.line(showline, p_s, p_e, color=(0, 255, 0), thickness=1)
            p_s = p_e

        center = (-int(show_w_scl * p[1]) + showline_w / 2, y)
        cv2.circle(showline, center, 2, (255, 255, 255), -1)


if __name__ == '__main__':
    # vis(0, 0)

    out_root = ""
    if not osp.exists(out_root):
        os.makedirs(out_root)

    frames = defaultdict(defaultdict)
    file = "misc/points3d_raw_debug.txt"
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
        cv2.putText(showline, f"frame_{frame}", (15, 48), cv2.FONT_HERSHEY_COMPLEX, 0.6, color=(192, 25, 192))

        # initialize distance grid
        dis_arr = np.arange(10, 110, 10)
        for dis in dis_arr:
            y = showline_h - dis * show_h_scl
            x_arr = np.arange(50, showline_w, 50)
            cv2.putText(showline, f"{dis}", (15, y), cv2.FONT_HERSHEY_COMPLEX, 0.4, color=(192, 192, 192))
            for x in x_arr:
                cv2.line(showline, (x, y), (x + 30, y), color=(192, 192, 192), thickness=1)

        # plot curve
        for line_id, line in lines.items():
            # coeff = cubic_ransac_curve_fit(line[:, 0], line[:, 1])
            coeff = polyfit(line[:, 0], line[:, 1])[::-1]  # c0 c1 c2 c3
            vis(line, coeff, showline)
        file_name = f"frame_{frame}_.png"
        out_path = osp.join(out_root, file_name)
        cv2.imwrite(out_path, showline)
