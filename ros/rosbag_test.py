import rosbag
# from cv_bridge import CvBridge
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

"""
Not work for numpy 
"""


def bvCoord_tobvImg(points, bot_center_bv, bot_center_img, scale):
    points = [[(x[0] - bot_center_bv[0]) * scale, (x[1] - bot_center_bv[1]) * scale] for x in points]
    points = [[-x[1] + bot_center_img[0], -x[0] + bot_center_img[1]] for x in points]
    return points


def draw_lanes(xxx, yyy):
    height = 768
    img = np.ones((768, 400, 3)) * 0
    scale = height / 100.
    bot_center_bv = (-20, 0)
    bot_center_img = (200, height - 2)

    img = cv2.line(img, (200, 0), (200, height - 1), (255, 255, 255), 1)

    pts = [(0, -10), (0, 10)]
    pts = bvCoord_tobvImg(pts, bot_center_bv, bot_center_img, scale)
    pts = [(int(pt[0]), int(pt[1])) for pt in pts]
    img = cv2.line(img, pts[0], pts[1], (255, 255, 255), 1)
    for i in range(-1, 4):
        pts = [(i * 20, -200 * 100 / float(height))]
        pts = bvCoord_tobvImg(pts, bot_center_bv, bot_center_img, scale)
        pts1 = pts[0]
        pts1[0] -= 30
        pts2 = pts[0]
        cv2.line(img, (int(pts1[0]), int(pts1[1])), (int(pts2[0]), int(pts2[1])), (255, 255, 255), 1)
        cv2.putText(img, str(i * 20), (int(pts2[0]), int(pts2[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    points = [[x, y] for x, y in zip(xxx, yyy)]
    points = bvCoord_tobvImg(points, bot_center_bv, bot_center_img, scale)
    for i, pt in enumerate(points):
        x, y = pt
        x, y = int(x), int(y)
        img = cv2.circle(img, (x, y), 2, (255, 255, 255), -1)
    cv2.imshow("heh", img)
    cv2.waitKey()

    return img


def lane_poly_test(lane_msg):
    lanes = lane_msg.lane_perception.lanes
    for lane in lanes:
        coefficient_bv = lane.coefficient_bv
        p1 = np.poly1d(coefficient_bv[:-1])
        print(p1)
        xxx = np.arange(-30, 100, 10)
        yyy = p1(xxx)
        # draw_lanes(xxx, yyy)
        debug = 1
        # plt.plot(xxx, yyy, '*', label='original values')
        # plt.xlabel('x axis')
        # plt.ylabel('y axis')
        # plt.legend(loc=4)  # 指定legend在图中的位置，类似象限的位置
        # plt.title('polyfitting')
        # plt.show()


def get_msg_list(file, topics, msg_only=True):
    msg_list = []
    for topic, msg, t in file.read_messages(topics=topics):
        if msg_only:
            msg_list.append(msg)
        else:
            msg_list.append([topic, msg, t])
        print(topic)
    return msg_list


if __name__ == '__main__':
    bag = "/Users/yel/PythonProjects/momenta/bags/false_brake/PLAA72032_event_manual_recording_20220308-195645_0.bag"
    file = rosbag.Bag(bag, 'r')
    # bridge = CvBridge()
    # list = list(file.read_messages())
    topics = ["/perception/vision/lane"]
    # topics = ["/recorder/event_v2"]
    # topics = ["/perception/vision/object"]
    topics = ["/vehicle/body_report"]
    topics = ["/mla/egopose"]
    topics = ["/msd/planning/plan"]
    msg_list = get_msg_list(file, topics)[:50]

    # lane_poly_test(msg_list)
    for msg in msg_list:
        # /msd/planning/plan
        extra = msg.extra
        json_info = json.loads(extra.json)
        for obj in msg.objects:
            vehicle_state_info = obj.vehicle_state_info
            ego_lane_relation_info = vehicle_state_info.ego_lane_relation_info
            ego_lane_realtion = ego_lane_relation_info.ego_lane_relation.value
            print(ego_lane_realtion)

    debug = 1

    # import cv2
    # import numpy as np
    # from cv_bridge import CvBridge
    # br = CvBridge()
    # dtype, n_channels = br.encoding_as_cvtype2('8UC3')
    # im = np.ndarray(shape=(480, 640, n_channels), dtype=dtype)
    # msg = br.cv2_to_imgmsg(im)  # Convert the image to a message
    # im2 = br.imgmsg_to_cv2(msg)  # Convert the message to a new image
    # cmprsmsg = br.cv2_to_compressed_imgmsg(im)  # Convert the image to a compress message
    # im22 = br.compressed_imgmsg_to_cv2(msg)  # Convert the compress message to a new image
    # cv2.imwrite("this_was_a_message_briefly.png", im2)
