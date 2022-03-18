import rosbag
# from cv_bridge import CvBridge
import numpy as np
import cv2
"""
Not work for numpy 
"""

if __name__ == '__main__':
    bag = "PLSSV164_event_dbw_disabled_20220126-111743_0.bag"
    file = rosbag.Bag(bag, 'r')
    # bridge = CvBridge()
    # list = list(file.read_messages())
    for topic, msg, t in file.read_messages(topics=['/sensor/camera_front_mid/cylinder/image_raw/compressed']):
        header = msg.header
        header_seq = header.seq
        stamp_sec = header.stamp.secs
        stamp_nsec = header.stamp.nsecs
        data = msg.data
        img = np.frombuffer(data, dtype=np.uint8)
        img = img.reshape()
        cv2.imshow("hehe", img)
        cv2.waitKey()

        print(topic)
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
