

def get_short_bag_name(full_name):
    full_name_split = full_name.split('_')
    bag_base_name, bag_date_string = full_name_split[0], full_name_split[-2]
    bag_date_string, bag_id = bag_date_string.split('-')
    bag_date_string = bag_date_string[:4] + '-' + bag_date_string[4:6] + '-' + bag_date_string[-2:] + '-' + \
                      bag_id[:2] + '-' + bag_id[2:4] + '-' + bag_id[-2:]
    bag_name = bag_base_name + '-' + bag_date_string
    return bag_name





if __name__ == '__main__':
    from multi_process.get_id import always

    # name = "PLSIV189_recording_no_cam_RVIZ_011_20211031-163137_20211031-174428_0.bag"
    #
    # nam2 = "PLSIV189_recording_no_cam_RVIZ_007_20211031-095344_20211031-110501_0.bag"
    # print(get_short_bag_name(name))
    # print(get_short_bag_name(nam2))
    # import numpy as np
    #
    # s_list = [f"PLEF04399-2021-11-26-14-00-22/front_mid_cylinder/frame{i}.png" for i in np.random.randint(0, 100, 20)]
    # print(s_list)
    #
    # new_s =sorted(s_list, key=lambda x: int(x.split('/')[-1].split('frame')[-1].split('.')[0]))
    # print(new_s)