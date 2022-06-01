import math

import cv2
import os
import os.path as osp
import argparse

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="The dir of input images")
    parser.add_argument("-o", "--output_dir", help="The output dir of video")
    parser.add_argument("--fps", type=int, default=24, help="The fps of video")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    img_dir = args.i
    out_path = args.o
    fps = args.fps
    img_list = os.listdir(img_dir)
    shape = cv2.imread(osp.join(img_dir, img_list[0])).shape[:2]  # H W

    # initialize video settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if not osp.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    out = cv2.VideoWriter(out_path, fourcc, fps, shape[::-1])

    for img_p in tqdm(img_list):
        img = cv2.imread(os.path.join(img_dir, img_p))
        out.write(img)

    out.release()
