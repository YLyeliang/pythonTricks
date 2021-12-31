import math

import cv2
import os
import multiprocessing as mp
import copy

video_name = "/home/yel/Downloads/16407826935298653.mp4"


def extract1(video_name, target_img_path):
    cap = cv2.VideoCapture(video_name)
    frame_idx = 0
    frame_step = 2
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # cv2.imwrite(
            #     os.path.join(target_img_path,
            #                  'frame{:d}.jpg'.format(frame_idx)), frame)
            frame_idx += frame_step  # i.e. at 30 fps, this advances one second
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            cap.retrieve()
            # cv2.imshow("hah",frame)
            # cv2.waitKey(0)

        else:
            cap.release()
            break


def extract2(video_name, target_img_path):
    cap = cv2.VideoCapture(video_name)
    frame_idx = 0
    frame_step = 2
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # if frame_idx % frame_step == 0:
            #     cv2.imwrite(
            #         os.path.join(target_img_path,
            #                      'frame{:d}.jpg'.format(frame_idx)), frame)
            frame_idx += 1
            # cv2.imshow("hah",frame)
            # cv2.waitKey(1)
        else:
            cap.release()
            break


def extract3(video_name, target_img_path, mp_num=4):
    cap = cv2.VideoCapture(video_name)
    frame_step = 1
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    step = max(1, math.floor(total_frame / 4))
    print(total_frame)
    print(step)

    def extract_video(cap, frame_idx, frame_step, frame_end):
        print("start")
        # if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        print(frame_idx)
        while cap.isOpened():
            print("read")
            ret, frame = cap.read()
            if frame_idx == frame_end:
                break
            if ret:
                cv2.imwrite(
                    os.path.join(target_img_path,
                                 'frame{:d}.jpg'.format(frame_idx)), frame)
                frame_idx += frame_step  # i.e. at 30 fps, this advances one second
                if frame_step != 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    p_list = []
    for i in range(mp_num):
        cap = cv2.VideoCapture(video_name)
        frame_start = i * step
        frame_end = (i + 1) * step
        print(frame_start, frame_end)

        p = mp.Process(target=extract_video, daemon=True, args=(cap, frame_start, frame_step, frame_end))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    cap.release()


def extract4(video_name, target_img_path):
    cap = cv2.VideoCapture(video_name)
    frame_idx = 0
    frame_step = 2
    # import numpy as np
    # frame = np.zeros((1024,768),dtype=np.uint8)
    while cap.isOpened():
        ret = cap.grab()
        if ret:
            if frame_idx % frame_step == 0:
                ret, frame = cap.retrieve()
                cv2.imwrite(
                    os.path.join(target_img_path,
                                 'frame{:d}.jpg'.format(frame_idx)), frame)
            frame_idx += 1
            # print(i, frame_idx)

            # cap.set(cv2.CAP_PROP_POS_FRAMES, 10000000)
            # cv2.imshow("hah",frame)
            # cv2.waitKey(1)
        else:
            cap.release()
            break


if __name__ == '__main__':
    path = 'img_path'
    path2 = 'img_path2'
    path3 = 'img_path3'
    import time

    os.makedirs(path, exist_ok=True)
    os.makedirs(path2, exist_ok=True)
    os.makedirs(path3, exist_ok=True)
    # s = time.time()
    # extract2(video_name, path2)
    # e = time.time()
    # print(e - s)

    s = time.time()
    extract1(video_name, path)
    e = time.time()
    print(e - s)

    s = time.time()
    extract4(video_name, path3)
    e = time.time()
    print(e - s)
