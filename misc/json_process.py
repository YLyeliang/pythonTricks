import json
import cv2
import numpy as np


def write_json(file="tmp.json"):
    a=[dict(a=1,b=2),dict(c=3,d=4)]
    # a = json.dumps(a,sort_keys=True,indent=4,separators=(",",":"))
    # print(a)
    with open(file,'w')  as f:
        json.dump(a,f,sort_keys=True,indent=4)


def read_json():
    name = "test_label.json"
    json_list = []
    with open(f"../{name}", 'r') as f:
        for line in f.readlines():
            json_dict = json.loads(line)
            json_list.append(json_dict)

    for json_dict in json_list:
        mask = np.zeros((1400, 1400), dtype=np.uint8)
        lanes = json_dict['lanes']
        h_samples = json_dict['h_samples']
        raw_file = json_dict['raw_file']

        points = []
        for lane in lanes:
            for i, point in enumerate(lane):
                if i> 0 and lane[i-1] >0 and lane[i] >0:
                # if point == -2 and lane[i + 1] == -2:
                #     continue
                    # line
                    cv2.line(mask,(lane[i-1],h_samples[i-1]),(lane[i],h_samples[i]),255,3)


                    # dot
                    # cv2.circle(mask, (point, h_samples[i]), 3, color=255, thickness=1)
        cv2.imshow("lane mask", mask)
        cv2.waitKey()

        debug = 1



if __name__ == '__main__':
    write_json()
