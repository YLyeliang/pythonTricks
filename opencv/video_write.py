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
    parser.add_argument("--vconcat",type=bool,default=False,help = "whether stack vertically")
    parser.add_argument("--hconcat", type=bool, default=False, help="whether concat horizontally")
    return parser.parse_args()



def tmp_video_write():
    args = parse_args()
    img_dirs = args.input_dir
    img_dirs = img_dirs.split(" ")
    out_path = args.output_dir
    start = args.start
    end = args.end
    fps = args.fps

    img_list_list = []

    other_root = "/home/yel/Projects/zdrive/py_project/misctools/results_3d_ros_time"
    other_list = os.listdir(other_root)
    for i, img_dir in enumerate(img_dirs):
        img_list = os.listdir(img_dir)
        shape = cv2.imread(osp.join(img_dir, img_list[0])).shape[:2]  # H W
        img_list.sort()
        end = len(img_list) if end == -1 else end
        img_list = img_list[start:end]
        for img_p in img_list:
            img_name =  str(int(img_p.split(".")[0]) + 200)+".jpg"
            if img_name in other_list:
                img_list_list.append([osp.join(img_dir,img_p),osp.join(other_root,img_name)])

    # initialize video settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = Path(out_path)
    pardir = out_path.parent.absolute()
    if not osp.exists(pardir):
        os.makedirs(pardir, exist_ok=True)

    video_shape = [shape[1],shape[0]*2]
    out = cv2.VideoWriter(str(out_path), fourcc, fps, video_shape)

    for img1_p,img2_p in tqdm(img_list_list):
        video_img = cv2.imread(img1_p)
        img = cv2.imread(img2_p)
        img_name = img1_p.split("/")[-1]
        img_name2 = img2_p.split("/")[-1]
        cv2.putText(video_img, img_name, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0))
        cv2.putText(img, img_name2, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0))
        video_img = cv2.vconcat([video_img, img])
        out.write(video_img)

    out.release()


if __name__ == '__main__':
    # tmp_video_write()
    # exit(0)
    args = parse_args()
    img_dirs = args.input_dir
    img_dirs = img_dirs.split(" ")
    out_path = args.output_dir
    start = args.start
    end = args.end
    fps = args.fps

    img_list_list = []

    shape_list = []
    for i,img_dir in enumerate(img_dirs):
        img_list = os.listdir(img_dir)
        shape = cv2.imread(osp.join(img_dir, img_list[0])).shape[:2]  # H W
        shape_list.append(shape)

        # img_list = sorted(img_list, key=lambda x: int(x.split("_")[1]))
        # img_list.sort(key=lambda x:int(x.split("_")[1]))
        img_list.sort()
        end = len(img_list) if end == -1 else end
        img_list = img_list[start:end]
        img_list = [osp.join(img_dir,img_p) for img_p in img_list]
        img_list_list.append(img_list)

    # initialize video settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = Path(out_path)
    pardir = out_path.parent.absolute()
    if not osp.exists(pardir):
        os.makedirs(pardir, exist_ok=True)

    video_shape = list(shape_list[0][::-1])
    if args.vconcat:
        for i in range(1,len(shape_list)):
            video_shape[1]+=shape_list[i][0]

    out = cv2.VideoWriter(str(out_path), fourcc, fps, video_shape)

    for img_id in tqdm(range(len(img_list_list[0]))):
        video_img = cv2.imread(img_list_list[0][img_id])
        img_name = img_list_list[0][img_id].split("/")[-1]
        # cv2.putText(video_img,img_name,(20,60),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,0))
        for list_id in range(1,len(img_list_list)):
            img = cv2.imread(img_list_list[list_id][img_id])
            img_name = img_list_list[list_id][img_id].split("/")[-1]
            cv2.putText(img, img_name, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0))
            video_img = cv2.vconcat([video_img,img])
        out.write(video_img)

    out.release()
