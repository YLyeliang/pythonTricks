import math

import cv2
import os
import os.path as osp
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="The dir of input images")
    parser.add_argument("-o", "--output_dir", help="The output dir of video")
    parser.add_argument("--start",type=int,default=0,help="index start from")
    parser.add_argument("--end", type=int, default=-1, help="index end in")
    parser.add_argument("--fps", type=int, default=24, help="The fps of video")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    img_dir = args.input_dir
    out_path = args.output_dir
    start = args.start
    end = args.end
    fps = args.fps
    img_list = os.listdir(img_dir)
    # img_list = [name for name in img_list if "track_point_un" in name]
    shape = cv2.imread(osp.join(img_dir, img_list[0])).shape[:2]  # H W

    img_list = sorted(img_list, key=lambda x: int(x.split("_")[1]))
    end = len(img_list) if end == -1 else end
    img_list = img_list[start:end]

    # initialize video settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = Path(out_path)
    pardir = out_path.parent.absolute()
    if not osp.exists(pardir):
        os.makedirs(pardir, exist_ok=True)
    out = cv2.VideoWriter(str(out_path), fourcc, fps, shape[::-1])

    for img_p in tqdm(img_list):
        img = cv2.imread(os.path.join(img_dir, img_p))
        out.write(img)

    out.release()
