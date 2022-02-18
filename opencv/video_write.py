import math

import cv2
import os

from tqdm import tqdm

out_path = "tmp.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, 3, (1024, 1068))

img_list = os.listdir("img_path")
for img_p in tqdm(img_list):
    img = cv2.imread(os.path.join("img_path", img_p))
    shape = img.shape[:2]
    out.write(img)

out.release()
