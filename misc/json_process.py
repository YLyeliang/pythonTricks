import json
import cv2
import numpy as np


def write_json(file="tmp.json"):
    a = [dict(a=1, b=2), dict(c=3, d=4)]
    # a = json.dumps(a,sort_keys=True,indent=4,separators=(",",":"))
    # print(a)
    with open(file, 'w') as f:
        json.dump(a, f, sort_keys=True, indent=4)


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
                if i > 0 and lane[i - 1] > 0 and lane[i] > 0:
                    # if point == -2 and lane[i + 1] == -2:
                    #     continue
                    # line
                    cv2.line(mask, (lane[i - 1], h_samples[i - 1]), (lane[i], h_samples[i]), 255, 3)

                    # dot
                    # cv2.circle(mask, (point, h_samples[i]), 3, color=255, thickness=1)
        cv2.imshow("lane mask", mask)
        cv2.waitKey()

        debug = 1


# sample_ddlane & metas not match
def sample_ddl_metas(data_version='v1001'):
    sample = "sample_ddlane.json"
    meta = "meta.tar"
    with open(sample, "r") as f:
        sample_ddlane_result = json.loads(f.read())
    with open(meta, "r") as f:
        metas_info = json.loads(f.read())
    sample_meta_list = [s['md5'] for s in sample_ddlane_result]
    meta_list = [m['md5'] for m in metas_info]
    count = {}
    for meta in meta_list:
        val = count.get(meta, 0)
        val += 1
        count[meta] = val
    for key, val in count.items():
        if val > 1:
            dup = key
    md5_to_meta_map = {}
    for meta in metas_info:
        if meta['md5'] == dup:
            debug = True
        md5_to_meta_map[meta['md5']] = meta

    for dt_info in sample_ddlane_result:
        md5 = dt_info['md5']
        meta = md5_to_meta_map[md5]
        single_frame_result = dt_info['single_frame_outputs']
        multi_frame_result = dt_info['multi_frame_outputs']
        meta[data_version]['groundTruth'] = {}
        meta[data_version]['groundTruth'][
            'single_frame_lane_result'] = single_frame_result
        meta[data_version]['groundTruth'][
            'multi_frame_lane_result'] = multi_frame_result

        lanes_dt = single_frame_result['maf_interface']['lane_perception'][
            'lanes']
        lanes = []
        keypoints = []
        points_3d = []

        intrinsic = meta[data_version]['frame']['camera'][0][
            'calibration']['intrinsic']['data']
        intrinsic = np.reshape(intrinsic, [3, 3])

        for i, lane_dt in enumerate(lanes_dt):
            if lane_dt['score'] < 0.5:
                continue
            points_x = lane_dt['points_2d_x']
            points_y = lane_dt['points_2d_y']
            points_v = lane_dt['points_2d_v']
            is_centerline = lane_dt['is_centerline']
            cls = i
            for x, y, depth in zip(points_x, points_y, points_v):
                y = y / 1
                points = [{'x': x, 'y': y}]
                keypoints.append({
                    'points': points,
                    'cls': cls,
                    'is_centerline': is_centerline,
                    'score': lane_dt['score']
                })

                if depth < 5 or y > 700 or x < 3 or y < 3:
                    continue

                theta_x = np.tan((x - intrinsic[0, 2]) / intrinsic[0, 0])
                theta_y = (y - intrinsic[1, 2]
                           ) / intrinsic[1, 1] * np.sqrt(theta_x ** 2 + 1)
                cam_3d_point = np.array(
                    [theta_x * depth, theta_y * depth, depth, 1.0])
                """radical depth"""
                # 3d points use meters.
                pts_3d = {
                    "x": cam_3d_point[0],
                    "y": cam_3d_point[1],
                    "z": cam_3d_point[2]
                }
                if pts_3d['x'] > 10 or pts_3d['z'] > 60 or pts_3d[
                    'x'] < -10 or pts_3d['z'] < 0:
                    continue
                points_3d.append({"points": [pts_3d], 'cls': cls})

        lane = {
            'miscData': {
                'keypoints': keypoints,
                'dataSlice': 1,
                'points_3d': points_3d
            }
        }
        lanes.append(lane)
        meta[data_version]['groundTruth']['lanes'] = lanes
    return metas_info


if __name__ == '__main__':
    # write_json()
    # sample_ddl_metas()
    import tempfile

    with tempfile.NamedTemporaryFile(prefix="sample_ddlane_meta_", suffix=".json") as temp_f:
        a = [dict(a=1, b=2), dict(c=3, d=4)]
        with open(temp_f.name, 'w') as f:
            json.dump(a, f, sort_keys=True, indent=4)
        with open(temp_f.name) as f:
            dd =json.load(f)
        debug = 1
