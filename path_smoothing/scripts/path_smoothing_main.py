#!/usr/bin/env python

import rospy
import pathsmoothinglib
import numpy as np
import time

from nav_msgs.msg import Path
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from dynamic_reconfigure.server import Server
from path_smoothing.cfg import ParamsConfig

def dyn_rec(config, level):
    global mean_angles_limit, std_angles_limit, path_length_limit, delta_width, num_of_iteratons, threshold, width_max_limit

    rospy.loginfo("""Reconfigure Request: {mean_angles_limit}, {std_angles_limit}, {path_length_limit}, {delta_width}, {num_of_iteratons}, {threshold}, {width_max_limit}""".format(**config))

    delta_width = config.delta_width
    num_of_iteratons = config.num_of_iteratons
    threshold = config.threshold
    width_max_limit = config.width_max_limit
    mean_angles_limit = config.mean_angles_limit
    std_angles_limit = config.std_angles_limit
    path_length_limit = config.path_length_limit

    return config

def costmap(msg):
    global costmap_array, cmap_update, map_res, costmap_update_array

    cmap_update = False

    data = msg.data

    map_x = msg.info.origin.position.x
    map_y = msg.info.origin.position.y
    map_height = msg.info.height
    map_width = msg.info.width
    map_res = msg.info.resolution

    # costmap array is transformed into a readable form
    data = np.reshape(data, (map_height, map_width))
    mat = data[::-1,::-1].T
    mat = np.transpose(mat)
    costmap_mat = np.flip(mat, 1)

    costmap_array = np.array([costmap_mat, map_x, map_y, map_height,
                             map_width, map_res])
    # costmap_update_array initialisation
    costmap_update_array = np.copy(costmap_array)

def costmap_update(msg):
    global costmap_update_array, cmap_update

    cmap_update = True

    data = msg.data

    map_x_update = msg.x
    map_y_update = msg.y
    map_width_update = msg.width
    map_height_update = msg.height

    # costmap array is transformed into a readable form
    data = np.reshape(data, (map_height_update, map_width_update))
    mat = data[::-1,::-1].T
    mat = np.transpose(mat)
    costmap_mat_update = np.flip(mat, 1)

    costmap_update_array = np.array([costmap_mat_update, map_x_update,
                                    map_y_update, map_height_update,
                                    map_width_update, map_res])

def path_initialisation(msg):
    global opt_param, param_check, path_init_flg

    if (param_init == False and param_check == False):
        PathSmooth.cancel_goal_movebase_client()
        rand_x, rand_y = \
            PathSmooth.get_random_point_map_costmap_check(1,
                                                          costmap_array,
                                                          costmap_update_array,
                                                          cmap_update)
        PathSmooth.set_goal_movebase_client(rand_x, rand_y)

        path_rough_init, check = PathSmooth.path_read(msg)
        PathSmooth.cancel_goal_movebase_client()

        path_rough_angles = PathSmooth.path_angles(path_rough_init)
        mean_angles = np.mean(path_rough_angles)
        std_angles = np.std(path_rough_angles)
        path_length, path_width = path_rough_init.shape

        # print("mean:", mean_angles, "len:", path_length, "std:", std_angles)

        if ((mean_angles >= mean_angles_limit) and
            (path_length >= path_length_limit) and
            (std_angles >= std_angles_limit) and
            (np.max(path_rough_angles) < 70)):

            # if condition for sufficiently "bad" path is met
            # paramter initialisation can begin
            
            rospy.loginfo("Parameter evaluation started.")
            # initial guesses (could be added to dynamic reconfigure)
            params0 = np.array([1000, 1])
            params1 = np.array([1000, 0.1])
            params2 = np.array([100, 1])
            params3 = np.array([100, 0.1])
            params4 = np.array([10, 1])
            params5 = np.array([10, 0.1])
            params = np.array([params0, params1, params2,
                              params3, params4, params5])
            opt_param = \
                PathSmooth.multistart_optimization(params,
                                                   path_rough_init,
                                                   costmap_array,
                                                   costmap_update_array,
                                                   cmap_update,
                                                   threshold,
                                                   delta_width,
                                                   width_max_limit,
                                                   NUM_OF_INBETWEEN_PTS,
                                                   True)
            
            rospy.loginfo("EVALUATION COMPLETE - optimal paramateres; mass: %f, stiffness: %f", opt_param[0], opt_param[1])
            rospy.loginfo("Parameter validation started.")

            param_check = True
            path_init_flg = path_rough_init
        else:
            # new random point
            rand_x, rand_y = \
                PathSmooth.get_random_point_map_costmap_check(1,
                                                              costmap_array,
                                                              costmap_update_array,
                                                              cmap_update)
            PathSmooth.set_goal_movebase_client(rand_x, rand_y)
    else:
        pass

def parameter_check(msg):
    # Parameter validation is done, because a marginal set of paramteres exist,
    # which preform very well on curved parts of the path but results in a zigzag
    # path where path to be smoothed is straight. Therefore parameter validation is done
    # on a relatively straight path.

    global param_check, param_init

    if (param_check == True and param_init == False):
        PathSmooth.cancel_goal_movebase_client()
        rand_x, rand_y = \
            PathSmooth.get_random_point_map_costmap_check(1,
                                                          costmap_array,
                                                          costmap_update_array,
                                                          cmap_update)
        PathSmooth.set_goal_movebase_client(rand_x, rand_y)

        path_rough_check, check = PathSmooth.path_read(msg)
        PathSmooth.cancel_goal_movebase_client()

        pth_ang = PathSmooth.path_angles(path_rough_check)
        pth_ang_ratio_5 = PathSmooth.no_angles_larger_than_ratio(pth_ang, 5)
        ANGLES_RATIO_LIMIT_5 = 0.2
        PATH_LENGTH_LIMIT = 10

        path_compare = path_rough_check != path_init_flg  # a new msg is required
        if ((len(path_rough_check[:, 0]) >= PATH_LENGTH_LIMIT) and
            (np.all(path_compare) == True) and
            (pth_ang_ratio_5 <= ANGLES_RATIO_LIMIT_5)):
            
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(path_rough_check[:, 0], path_rough_check[:, 1])
            # plt.show

            path_rough_seed, check = PathSmooth.path_seed(path_rough_check,
                                                          NUM_OF_INBETWEEN_PTS)
            path_width_check, check = \
                PathSmooth.inc_costmap_check(path_rough_seed,
                                             costmap_array,
                                             costmap_update_array,
                                             cmap_update,
                                             threshold,
                                             delta_width,
                                             width_max_limit)
            
            path_smooth_check, check = \
                PathSmooth.path_smooth(path_rough_seed,
                                       path_width_check,
                                       opt_param[0],
                                       opt_param[1],
                                       num_of_iteratons,
                                       True)

            #oznaka: r - rough, s - smooth
            path_angles_r = PathSmooth.path_angles(path_rough_check)
            mean_angles_r = np.mean(path_angles_r)
            std_angles_r = np.std(path_angles_r)
            # path_length_r, path_width_r = path_check_rough.shape

            path_angles_s = PathSmooth.path_angles(path_smooth_check)
            mean_angles_s = np.mean(path_angles_s)
            std_angles_s = np.std(path_angles_s)
            # path_length_s, path_width_s = path_check_smooth.shape

            # print("r:", mean_angles_r, std_angles_r)
            # print("s:", mean_angles_s, std_angles_s)

            UPPER_LIMIT = 0.95
            LOWER_LIMIT = 0.05

            if ((mean_angles_s < mean_angles_r*UPPER_LIMIT) and
                (std_angles_s < std_angles_r*UPPER_LIMIT) and
                (mean_angles_s > mean_angles_r*LOWER_LIMIT) and
                (std_angles_s > std_angles_r*LOWER_LIMIT)):

                rospy.loginfo("Good set of parameters found. Validation complete.")
                param_init = True
                param_check = False
            else:
                rospy.loginfo("Good set of parameters not found. Evaluation restarted.")
                param_init = False
                param_check = False

                rand_x, rand_y = \
                    PathSmooth.get_random_point_map_costmap_check(1,
                                                                  costmap_array,
                                                                  costmap_update_array,
                                                                  cmap_update)
                PathSmooth.set_goal_movebase_client(rand_x, rand_y)

        else:
            rand_x, rand_y = \
                PathSmooth.get_random_point_map_costmap_check(1,
                                                              costmap_array,
                                                              costmap_update_array,
                                                              cmap_update)
            PathSmooth.set_goal_movebase_client(rand_x, rand_y)
    else:
        pass


def path_smoothing(msg):
    global previous_msg_seq

    if (param_init == True and param_check == False):
        current_msg_seq = msg.header.seq
        seq_check = PathSmooth.msg_seq_check(previous_msg_seq, current_msg_seq)

        if (seq_check == True):
            path_read, check = PathSmooth.path_read(msg)
            path_seed, check = PathSmooth.path_seed(path_read,
                                                    NUM_OF_INBETWEEN_PTS)
            path_width, check = \
                PathSmooth.inc_costmap_check(path_seed,
                                             costmap_array,
                                             costmap_update_array,
                                             cmap_update,
                                             threshold,
                                             delta_width,
                                             width_max_limit)
            path_smooth, check = PathSmooth.path_smooth(path_seed,
                                                        path_width,
                                                        opt_param[0],
                                                        opt_param[1],
                                                        num_of_iteratons,
                                                        False)

            previous_msg_seq = current_msg_seq - 1
        else:
            pass

        pub_path_smooth.publish(PathSmooth.path_write(path_smooth,
                                                      path_seed,
                                                      msg))
        pub_zone_left.publish(PathSmooth.path_zone_write(path_seed,
                                                         path_width,
                                                         "left",
                                                         msg))
        pub_zone_right.publish(PathSmooth.path_zone_write(path_seed,
                                                          path_width,
                                                          "right",
                                                          msg))
    else:
        pass

if __name__ == '__main__':
    try:
        rospy.init_node('path_smoothing_node')
        # dynamic reconfigure
        srv = Server(ParamsConfig, dyn_rec)
        # publishers
        pub_path_smooth = rospy.Publisher('/vamos',
                                          Path,
                                          queue_size=10)
        pub_zone_left = rospy.Publisher('/vamos_zone_left',
                                        Path,
                                        queue_size=10)
        pub_zone_right = rospy.Publisher('/vamos_zone_right',
                                         Path,
                                         queue_size=10)
        # subscribers
        rospy.Subscriber('/move_base/TrajectoryPlannerROS/global_plan',
                         Path,
                         path_smoothing)
        rospy.Subscriber('/move_base/TrajectoryPlannerROS/global_plan',
                         Path,
                         path_initialisation)
        rospy.Subscriber('/move_base/TrajectoryPlannerROS/global_plan',
                         Path,
                         parameter_check)
        rospy.Subscriber('/move_base/global_costmap/costmap',
                         OccupancyGrid,
                         costmap)
        rospy.Subscriber('/move_base/global_costmap/costmap_updates',
                         OccupancyGridUpdate,
                         costmap_update)

        PathSmooth = pathsmoothinglib.PathSmoothing()

        NUM_OF_INBETWEEN_PTS = 1
        # if this paramter is changed, the initialisation porcess needs to be
        # redone, because the path smoothing process is a function of the
        # path (x, y points) dimensions (why: torque = RADIUS x force)

        # global variables initialisation
        previous_msg_seq = 0
        param_init = False
        param_check = False

        cntdwn = 10.0
        rospy.loginfo(str(cntdwn) + " seconds to set evaluation parameters.")
        # rospy.sleep(cntdwn)
        time.sleep(cntdwn)
        rospy.loginfo("Searching for an adequate path to start parameter evaluation.")
        PathSmooth.set_goal_movebase_client()

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
