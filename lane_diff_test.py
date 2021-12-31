import logging
import tempfile
from typing import List
import os
import cv2
import numpy as np
import copy
from collections import defaultdict
from ppl.task import Task


class LaneDiffTask(Task):
    '''
    Pull the infer results of autolabeling from two big models from the data_pool;
    Diff the the results, send them to training data lake if match,
    write the relative path of the images needed to be manually annotated if not match,
    abort if the quality of the result is too low.
    '''

    def __init__(self, *kargs, **kwargs):
        super(LaneDiffTask, self).__init__(*kargs, **kwargs)
        self.ymax = 768
        self.match_thr = 20
        self.target_width = 1024
        self.target_height = 768
        self.show_diff_result = False

    def process(self, data, params):
        dt, gt = data['meta_1'], data['meta_2']

        meta_toPpl, path_toLabeled = self.diff(dt, gt, params)
        ToLabeled_txt = 'img_relative_path_toLabeled.txt'
        os.system('rm {}'.format(ToLabeled_txt))
        with open(ToLabeled_txt, 'w') as f:
            f.write('\n'.join(path_toLabeled))

        return meta_toPpl

    def diff(self, dts_meta, gts_meta, params):
        dts = self.meta_to_diff_format(dts_meta, params)
        gts = self.meta_to_diff_format(gts_meta, params)
        to_ppl = []
        to_labeled = []
        if self.show_diff_result:
            vis_path = '/syncdata/hejianye/vis_tmp'
            os.system('rm -r {}'.format(vis_path))
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
        gts_md5 = {gt['md5']: j for j, gt in enumerate(gts)}
        for i, dt in enumerate(dts):
            if dt['md5'] in gts_md5:
                gt = gts[gts_md5[dt['md5']]]
                try:
                    abort, flag, lane_hl = self.get_ego_edge_gt_hl(dt, gt)
                except Exception as e:
                    abort = True
                    print(e)
                if abort:
                    print(dt['md5'], 'abort', i)
                    continue
                if not flag:
                    print(dt['md5'], 'match', i)
                    to_ppl.append(dts_meta[i])
                if flag:
                    print(dt['md5'], 'not match', i)
                    to_labeled.append(dt['path'])
                if self.show_diff_result:
                    self.show_result(
                        dt, gt, vis_path=vis_path, lanes_hl=lane_hl, flag=flag)

        return to_ppl, to_labeled

    def get_ego_edge_gt_hl(self, _dt, _gt):
        left_dt_hl_line = []
        left_gt_hl_line = []
        right_dt_hl_line = []
        right_gt_hl_line = []
        l_left_dt_hl_line = []
        l_left_gt_hl_line = []
        r_right_dt_hl_line = []
        r_right_gt_hl_line = []

        dt = self.gen_format_lines(_dt, 'lane')
        gt = self.gen_format_lines(_gt, 'lane')
        abort = False

        if (len(dt) < 2) or (len(gt) < 2) or (self.get_y_margin(dt) < 100) or (
                self.get_y_margin(gt) < 100):
            abort = True
            return abort, True, {
                'gt_hl': [[], [], [], []],
                'dt_hl': [[], [], [], []]
            }

        dt = sorted(dt, key=lambda x: x[0])
        gt = sorted(gt, key=lambda x: x[0])
        dt_l_left_ego_index, dt_left_ego_index, dt_right_ego_index, dt_r_right_ego_index = self.get_ego_index_with_neighbor(
            dt)
        gt_l_left_ego_index, gt_left_ego_index, gt_right_ego_index, gt_r_right_ego_index = self.get_ego_index_with_neighbor(
            gt)

        if dt_left_ego_index == -1 or gt_left_ego_index == -1:
            abort = True
            return abort, True, {
                'gt_hl': [[], [], [], []],
                'dt_hl': [[], [], [], []]
            }

        left_flag = self.line_matcher(dt[dt_left_ego_index][-1],
                                      gt[gt_left_ego_index][-1])
        if (dt_l_left_ego_index == 999) and (gt_l_left_ego_index == 999):
            l_left_flag = True
        elif (dt_l_left_ego_index == 999) or (gt_l_left_ego_index == 999):
            l_left_flag = False
        else:
            l_left_flag = self.line_matcher(dt[dt_l_left_ego_index][-1],
                                            gt[gt_l_left_ego_index][-1])

        right_flag = self.line_matcher(dt[dt_right_ego_index][-1],
                                       gt[gt_right_ego_index][-1])
        if (dt_r_right_ego_index == 999) and (gt_r_right_ego_index == 999):
            r_right_flag = True
        elif (dt_r_right_ego_index == 999) or (gt_r_right_ego_index == 999):
            r_right_flag = False
        else:
            r_right_flag = self.line_matcher(dt[dt_r_right_ego_index][-1],
                                             gt[gt_r_right_ego_index][-1])

        if l_left_flag and left_flag and right_flag and r_right_flag:
            return abort, False, {
                'gt_hl': [[], [], [], []],
                'dt_hl': [[], [], [], []]
            }

        elif self.judge_outmost_line(l_left_flag, left_flag, right_flag,
                                     r_right_flag, dt_l_left_ego_index,
                                     gt_l_left_ego_index, dt_r_right_ego_index,
                                     gt_r_right_ego_index, dt, gt):
            return abort, False, {
                'gt_hl': [[], [], [], []],
                'dt_hl': [[], [], [], []]
            }
        else:
            if not left_flag:
                left_dt_hl_line = dt[dt_left_ego_index][2]
                left_gt_hl_line = gt[gt_left_ego_index][2]
            if not l_left_flag:
                if dt_l_left_ego_index == 999:
                    l_left_dt_hl_line = []
                else:
                    l_left_dt_hl_line = dt[dt_l_left_ego_index][2]
                if gt_l_left_ego_index == 999:
                    l_left_gt_hl_line = []
                else:
                    l_left_gt_hl_line = gt[gt_l_left_ego_index][2]
            if not right_flag:
                right_dt_hl_line = dt[dt_right_ego_index][2]
                right_gt_hl_line = gt[gt_right_ego_index][2]
            if not r_right_flag:
                if dt_r_right_ego_index == 999:
                    r_right_dt_hl_line = []
                else:
                    r_right_dt_hl_line = dt[dt_r_right_ego_index][2]
                if gt_r_right_ego_index == 999:
                    r_right_gt_hl_line = []
                else:
                    r_right_gt_hl_line = gt[gt_r_right_ego_index][2]
            return abort, True, {
                'gt_hl': [
                    l_left_gt_hl_line, left_gt_hl_line, right_gt_hl_line,
                    r_right_gt_hl_line
                ],
                'dt_hl': [
                    l_left_dt_hl_line, left_dt_hl_line, right_dt_hl_line,
                    r_right_dt_hl_line
                ]
            }

    def meta_to_diff_format(self, metas_info, params):
        data_version = params['data_version']
        imgs_info = []
        for meta_info in metas_info:
            img_info = {}
            img_path = meta_info['_id']
            img_md5 = meta_info['md5']
            img_tag = meta_info['tag']
            lanes_info = meta_info[data_version]

            width = lanes_info['frame']['camera'][0]['image']['width']
            height = lanes_info['frame']['camera'][0]['image']['height']

            points_info = lanes_info['groundTruth']['lanes'][0]['miscData'][
                'keypoints']

            lines_info = defaultdict(list)
            for point_info in points_info:
                line_id = point_info['cls']
                lines_info[line_id].append(point_info['points'][0])
            lanes = []
            for line_id, line_info in lines_info.items():
                lanes.append({'cpoints': line_info})
            img_info['height'] = height
            img_info['width'] = width
            img_info['Lanes'] = lanes
            img_info['path'] = img_path
            img_info['md5'] = img_md5
            img_info['tag'] = img_tag
            imgs_info.append(img_info)

        return imgs_info

    def gen_format_lines(self, lines, line_type):
        lst = []
        all_lines = lines['Lanes']
        height = lines['height']
        width = lines['width']
        scale_x = 1.0 * self.target_width / width
        scale_y = 1.0 * self.target_height / height

        for line in all_lines:
            if len(line['cpoints']) < 2:
                continue
            line_org = copy.deepcopy(line)
            for pts in line['cpoints']:
                pts['x'] = pts['x'] * scale_x
                pts['y'] = pts['y'] * scale_y
            pt = [
                self.gen_x_with_ymax(line['cpoints']),
                self.gen_slope(line['cpoints']), line_org, line['cpoints'],
                line_type,
                self.slice_cpoints(line['cpoints'], 576, 144)
            ]
            lst.append(pt)
        return lst

    def get_y_margin_line(self, line_points):
        point_y_list = []
        for pts in line_points:
            point_y_list.append(pts['y'])
        return max(point_y_list) - min(point_y_list)

    def get_y_margin(self, lines_info):
        y_margin_list = []
        for line_info in lines_info:

            y_margin_line = self.get_y_margin_line(line_info[3])
            y_margin_list.append(y_margin_line)
        return max(y_margin_list)

    def spline_interp_step(self, pts, step=1):
        res = []
        if len(pts) <= 2:
            res = pts
            return res
        tmp_param = self.cal_params(pts)

        if len(tmp_param) == 0:
            print('error during cal spline param')
            return res
        for f in tmp_param:
            st = []
            tmp = 0
            while tmp < f['h']:
                st.append(tmp)
                tmp += step

            for t in st[:]:
                x = self.getX(f, t)
                y = self.getY(f, t)
                res.append({'x': x, 'y': y})
        res.append(pts[len(pts) - 1])
        return res

    def cal_params(self, pts):
        params = []
        if len(pts) <= 2:
            return params
        h = []
        for i in range(0, len(pts) - 1):
            dx = pts[i]['x'] - pts[i + 1]['x']
            dy = pts[i]['y'] - pts[i + 1]['y']
            dis = np.sqrt(dx * dx + dy * dy)
            h.append(dis)
        A = []
        B = []
        C = []
        Dx = []
        Dy = []
        for i in range(0, len(pts) - 2):
            A.append(h[i])
            B.append(2 * (h[i] + h[i + 1]))
            C.append(h[i + 1])
            dx1 = (pts[i + 1]['x'] - pts[i]['x']) / h[i]
            dx2 = (pts[i + 2]['x'] - pts[i + 1]['x']) / h[i + 1]
            Dx.append(6 * (dx2 - dx1))
            dy1 = (pts[i + 1]['y'] - pts[i]['y']) / h[i]
            dy2 = (pts[i + 2]['y'] - pts[i + 1]['y']) / h[i + 1]
            Dy.append(6 * (dy2 - dy1))

        C[0] /= B[0]
        Dx[0] /= B[0]
        Dy[0] /= B[0]
        for i in range(1, len(pts) - 2):
            tmp = B[i] - A[i] * C[i - 1]
            C[i] /= tmp
            Dx[i] = (Dx[i] - A[i] * Dx[i - 1]) / tmp
            Dy[i] = (Dy[i] - A[i] * Dy[i - 1]) / tmp

        Mx = np.zeros(len(pts))
        My = np.zeros(len(pts))
        Mx[len(pts) - 2] = Dx[len(pts) - 3]
        My[len(pts) - 2] = Dy[len(pts) - 3]
        for i in range(len(pts) - 4, -1, -1):
            Mx[i + 1] = Dx[i] - C[i] * Mx[i + 2]
            My[i + 1] = Dy[i] - C[i] * My[i + 2]

        Mx[0] = 0
        Mx[-1] = 0
        My[0] = 0
        My[-1] = 0

        for i in range(0, len(pts) - 1):
            param = {}
            param['a_x'] = pts[i]['x']
            param['b_x'] = (pts[i + 1]['x'] - pts[i]['x']) / h[i] - (
                2 * h[i] * Mx[i] + h[i] * Mx[i + 1]) / 6
            param['c_x'] = Mx[i] / 2
            param['d_x'] = (Mx[i + 1] - Mx[i]) / (6 * h[i])
            param['a_y'] = pts[i]['y']
            param['b_y'] = (pts[i + 1]['y'] - pts[i]['y']) / h[i] - (
                2 * h[i] * My[i] + h[i] * My[i + 1]) / 6
            param['c_y'] = My[i] / 2
            param['d_y'] = (My[i + 1] - My[i]) / (6 * h[i])
            param['h'] = h[i]
            params.append(param)
        return params

    def getX(self, f, t):
        return f['a_x'] + f['b_x'] * t + f['c_x'] * t * t + f['d_x'] * t * t * t

    def getY(self, f, t):
        return f['a_y'] + f['b_y'] * t + f['c_y'] * t * t + f['d_y'] * t * t * t

    def calc_y_cross(self, p0, p1, y):
        eps = 1e-6
        if abs(p0['y'] - p1['y']) < eps:
            return -1
        k = (p0['x'] - p1['x']) / (p0['y'] - p1['y'])
        b = p0['x'] - k * p0['y']
        return k * y + b

    def slice_cpoints(self, cpoints, im_height, pts_per_lane):
        start_pos = -1
        end_pos = -1
        slice_y_cpoints = []
        if len(cpoints) < 2:
            return slice_y_cpoints, start_pos, end_pos
        if cpoints[0]['y'] < cpoints[-1]['y']:
            cpoints.reverse()
        try:
            spline_cpoints = self.spline_interp_step(cpoints, 4)
        except:
            new_cpoints = [cpoints[0]]
            for i in range(1, len(cpoints)):
                if abs(cpoints[i]['x'] - cpoints[i - 1]['x']) + abs(
                        cpoints[i]['y'] - cpoints[i - 1]['y']) < 0.001:
                    continue
                new_cpoints.append(cpoints[i])
            spline_cpoints = self.spline_interp_step(new_cpoints, 4)
        start_y = spline_cpoints[0]['y']
        end_y = spline_cpoints[-1]['y']
        i_p = 0
        for idx in range(pts_per_lane):
            y = im_height - 1 - im_height / pts_per_lane * idx
            while i_p < len(spline_cpoints) - 2 and y < spline_cpoints[i_p +
                                                                       1]['y']:
                i_p += 1
            cross_x = self.calc_y_cross(spline_cpoints[i_p],
                                        spline_cpoints[i_p + 1], y)
            if y <= spline_cpoints[i_p]['y'] and start_pos < 0:
                start_pos = idx
            if y < spline_cpoints[i_p + 1]['y'] and end_pos < 0:
                end_pos = idx
            if (start_y - y) * (end_y - y) <= 0:
                slice_y_cpoints.append({"x": cross_x, "y": y})
        end_pos = start_pos + len(slice_y_cpoints)

        return slice_y_cpoints

    def get_intersection_gt(self, edge_gt_root, lane_gt_root):
        edge_gt = os.listdir(edge_gt_root)
        lane_gt = os.listdir(lane_gt_root)
        self.valid_gt_list = list(set(edge_gt) & set(lane_gt))

    def gen_x_with_ymax(self, line):
        point_gap = 1 if len(line) < 5 else 3
        is_reverse = -1 if line[0]['y'] < line[-1]['y'] else 1

        x1 = line[is_reverse * 1]['x']
        y1 = line[is_reverse * 1]['y']
        x2 = line[is_reverse * (1 + point_gap)]['x']
        y2 = line[is_reverse * (1 + point_gap)]['y']
        x = x1 + (self.ymax - y1) * (x1 - x2) / (y1 - y2)
        return x

    def gen_slope(self, line):
        point_gap = 1 if len(line) < 5 else 3
        is_reverse = -1 if line[0]['y'] < line[-1]['y'] else 1

        x1 = line[is_reverse * 1]['x']
        y1 = line[is_reverse * 1]['y']
        x2 = line[is_reverse * (1 + point_gap)]['x']
        y2 = line[is_reverse * (1 + point_gap)]['y']
        if x1 == x2:
            return 999
        else:
            return (y1 - y2) / (x1 - x2)

    def get_ego_index(self, line):
        for i in range(len(line) - 1):
            k1 = line[i][1]
            k2 = line[i + 1][1]
            if k1 * k2 < 0:
                return i, i + 1
        return -1, -1

    def get_ego_index_with_neighbor(self, line):
        for i in range(len(line) - 1):
            k1 = line[i][1]
            k2 = line[i + 1][1]
            if k1 * k2 < 0:
                l_i, r_i = i - 1, i + 2
                if (l_i < 0):
                    l_i = 999
                if (r_i > len(line) - 1):
                    r_i = 999
                return l_i, i, i + 1, r_i
        return -1, -1, -1, -1

    def convert_points_list_to_dic(self, lst):
        dic = {}
        for i in range(len(lst)):
            x = lst[i]['x']
            y = lst[i]['y']
            dic[int(y)] = x
        return dic

    def line_matcher(self, dt_points, gt_points):
        if not dt_points or not gt_points:
            return False
        min_dt_y = min(dt_points[-1]['y'], dt_points[0]['y'])
        max_dt_y = max(dt_points[-1]['y'], dt_points[0]['y'])
        min_gt_y = min(gt_points[-1]['y'], gt_points[0]['y'])
        max_gt_y = max(gt_points[-1]['y'], gt_points[0]['y'])
        min_y = int(max(min_dt_y, min_gt_y))
        max_y = int(min(max_dt_y, max_gt_y))

        if (max_y - min_y) / (
                max(max_dt_y, max_gt_y) - min(min_dt_y, min_gt_y)) < 0.3:
            # if self.get_lane_overlap_ratio(dt_points, gt_points) < 0.3:
            return False

        dt_dic = self.convert_points_list_to_dic(dt_points)
        gt_dic = self.convert_points_list_to_dic(gt_points)

        num = 0
        dis = 0.0
        for y in range(min_y, max_y + 1, 4):
            if y in dt_dic and y in gt_dic:
                dis += abs(dt_dic[y] - gt_dic[y])
                num += 1

        if num == 0:
            return False
        else:
            ave_dis = dis / num
            if ave_dis > self.match_thr:
                return False
            else:
                return True

    def line_matcher_y_min(self, dt_points, gt_points):
        if not dt_points or not gt_points:
            return False

        if dt_points[0]['y'] < dt_points[-1]['y']:
            dt_points.reverse()
        if gt_points[0]['y'] < gt_points[-1]['y']:
            gt_points.reverse()

        min_dt_y = min(dt_points[-1]['y'], dt_points[0]['y'])
        min_gt_y = min(gt_points[-1]['y'], gt_points[0]['y'])
        min_y = int(max(min_dt_y, min_gt_y))

        x1 = dt_points[-1]['x']
        y1 = dt_points[-1]['y']
        x2 = dt_points[-2]['x']
        y2 = dt_points[-2]['y']
        dt_x_min = x1 + (min_y - y1) * (x1 - x2) / (y1 - y2)

        x1 = gt_points[-1]['x']
        y1 = gt_points[-1]['y']
        x2 = gt_points[-2]['x']
        y2 = gt_points[-2]['y']
        gt_x_min = x1 + (min_y - y1) * (x1 - x2) / (y1 - y2)
        return dt_x_min, gt_x_min

    def judge_outmost_line(self, l_left_flag, left_flag, right_flag,
                           r_right_flag, dt_l_left_ego_index,
                           gt_l_left_ego_index, dt_r_right_ego_index,
                           gt_r_right_ego_index, dt, gt):
        if int(l_left_flag) + int(left_flag) + int(right_flag) + int(
                r_right_flag) == 3:
            is_left = (not l_left_flag) and (((dt_l_left_ego_index == 999) or
                                              (dt_l_left_ego_index == 0)) and
                                             ((gt_l_left_ego_index == 999) or
                                              (gt_l_left_ego_index == 0)))
            is_right = (not r_right_flag) and (
                ((dt_r_right_ego_index == 999) or
                 (dt_r_right_ego_index == len(dt) - 1)) and
                ((gt_r_right_ego_index == 999) or
                 (gt_r_right_ego_index == len(gt) - 1)))
            if is_left:
                if self.judge_left_conform_order_lines(
                        dt) and self.judge_left_conform_order_lines(gt):
                    return True
            if is_right:
                if self.judge_right_conform_order_lines(
                        dt) and self.judge_right_conform_order_lines(gt):
                    return True
        return False

    def judge_right_conform_order_lines(self, lines):
        x_inner, x_outer = self.line_matcher_y_min(lines[-2][-1],
                                                   lines[-1][-1])
        if x_inner > x_outer:
            return False
        return True

    def judge_left_conform_order_lines(self, lines):
        x_inner, x_outer = self.line_matcher_y_min(lines[1][-1], lines[0][-1])
        if x_inner < x_outer:
            return False
        return True

    def connect_points_to_line(self, lane, shape, thickness=None):
        """
        Gt line may be sparse, so it needs to be interpolated by cv.line and return. (points num is the pixel num in the
        interpolated line.
        """
        img_height, img_width = shape
        line_map = np.zeros([img_height, img_width], dtype=np.uint8)
        points = []
        for point in lane:
            points.append([point['x'], point['y']])
        if not points:
            return line_map
        for idx in range(len(points) - 1):
            src_point = tuple(list(map(int, points[idx])))
            dst_point = tuple(list(map(int, points[idx + 1])))
            cv2.line(
                line_map,
                src_point,
                dst_point,
                color=(1, 1, 1),
                thickness=thickness)
        return line_map

    def get_lane_overlap_ratio(self, dt_points, gt_points):
        judge_lanes_map_a = self.connect_points_to_line(
            dt_points, [576, 1024], 20)
        judge_lanes_map_b = self.connect_points_to_line(
            gt_points, [576, 1024], 20)
        judge_lanes_map = judge_lanes_map_a + judge_lanes_map_b
        return np.sum(judge_lanes_map >= 2) / len(judge_lanes_map_b >= 1)

    def show_result(self,
                    dt,
                    gt,
                    vis_path='vis_tmp',
                    lanes_hl={
                        'gt_hl': [[], [], [], []],
                        'dt_hl': [[], [], [], []]
                    },
                    flag=False):
        if True:
            im_dt = os.path.join('/share/public', dt['path'])
            im_gt = os.path.join('/share/public', gt['path'])
            im_dt_name = os.path.basename(im_dt)
            vis = os.path.join(vis_path, im_dt_name)
            im_dt = cv2.imread(im_dt)
            im_gt = cv2.imread(im_gt)

            im_dt = self.line_vis(im_dt, dt, (0, 255, 0), 'big')
            im_dt = self.hl_line_vis(im_dt, lanes_hl['dt_hl'], (0, 0, 255))

            im_gt = self.line_vis(im_gt, gt, (0, 255, 0), 'big_caffe')
            im_gt = self.hl_line_vis(im_gt, lanes_hl['gt_hl'], (0, 0, 255))
            self.im_save(im_dt, im_gt, vis, flag)

    def line_vis(self, im, lines, color, text=None):
        im = im.copy()
        if text:
            cv2.putText(im, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)

        height = lines['height']
        width = lines['width']
        scale_x = 1.0 * self.target_width / width
        scale_y = 1.0 * self.target_height / height

        for line in lines['Lanes']:
            for pts in line['cpoints']:
                pts['x'] = pts['x'] / scale_x
                pts['y'] = pts['y'] / scale_y

            line = line['cpoints']
            for i in range(len(line) - 1):
                p1 = line[i]
                p2 = line[i + 1]
                cv2.line(im, (int(p1['x']), int(p1['y'])),
                         (int(p2['x']), int(p2['y'])), color, 2)
        return im

    def hl_line_vis(self, im, lines, color):
        im = im.copy()
        for line in lines:
            if not line:
                continue
            line = line['cpoints']
            for i in range(len(line) - 1):
                p1 = line[i]
                p2 = line[i + 1]
                cv2.line(im, (int(p1['x']), int(p1['y'])),
                         (int(p2['x']), int(p2['y'])), color, 2)
        return im

    def im_save(self, im_dt, im_gt, vis, flag):

        im = np.concatenate((im_dt, im_gt), 1)
        if not flag:
            cv2.putText(im, 'Match', (900, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255), 4)
        else:
            cv2.putText(im, 'Not Match', (900, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 255), 4)
        cv2.imwrite(vis, im)
