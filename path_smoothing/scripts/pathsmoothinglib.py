#!/usr/bin/env python

from __future__ import division
import rospy
import numpy as np
import std_msgs.msg
import matplotlib.pyplot as plt
import actionlib
import time

from mpl_toolkits.mplot3d import Axes3D
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from move_base_msgs.msg import MoveBaseAction
from move_base_msgs.msg import MoveBaseGoal
from geometry_msgs.msg import Twist

# auxiliary functions
def perpendicular_at_point(path):
    """Returns direction vectors that are perpendicular to a path.
    
    Arguments:
        path {2D array} -- 2D array with first column of x points
        and second column of y points
    """

    dpos = np.diff(path, axis=0)
    ddist = np.sqrt(np.sum(dpos ** 2, axis=1))
    dpos_n = dpos / ddist[:, np.newaxis]
    perp = dpos_n[:, [1, 0]] * (-1, 1)
    perp_mid = perp[:-1] + perp[1:]
    perp_mid = perp_mid / np.sqrt(np.sum(perp_mid ** 2, axis=1))[:, np.newaxis]
    perp_at_point = np.zeros_like(path)
    if (len(perp_at_point) > 1):
        perp_at_point[0] = perp[0]
        perp_at_point[1:-1] = perp_mid
        perp_at_point[-1] = perp[-1]
    return perp_at_point

def costmap_mat_update_func(costmap_mat, costmap_mat_update, dx, dy):
    """Returns updated costmap matrix.
    
    Arguments:
        costmap_mat {2D array} -- non-updated costmap matrix which
        is published once
        costmap_mat_update {2D array} -- costmap matrix updates
        dx {Int} -- costmap matrix update x offset
        dy {Int} -- costmap matrix update y offset
    """

    len_cm_y = len(costmap_mat[:, 0])

    len_cmu_x = len(costmap_mat_update[0, :])
    len_cmu_y = len(costmap_mat_update[:, 0])

    for i in range(len_cmu_x):
        for j in range(len_cmu_y):
            costmap_mat[len_cm_y - len_cmu_y - dy + j, dx + i] = \
                        costmap_mat_update[j, i]
    return costmap_mat

# path smooting class
class PathSmoothing():

    def __init__(self):
        self.prev_msg_seq = None
        self.curr_msg_seq = None
        self.msg = None
        self.num_of_inbetween_points = 1
        self.costmap_array = np.zeros(5)
        self.costmap_array_update = np.zeros(5)
        self.cmap_update = False
        self.threshold = 50
        self.delta_t = 0.2
        self.tmax_limit = 2.0
        self.move = Path()

    def msg_seq_check(self, prev_msg_seq, curr_msg_seq):
        """Returns True for next callback message sequence.
        
        Arguments:
            prev_msg_seq {Int} -- Previous message sequence
            curr_msg_seq {Int} -- Current message sequnce
        
        Returns:
            [Bool] -- True if new message condition is met
        """

        if (prev_msg_seq != curr_msg_seq):
            return True
        else:
            return False

    def path_read(self, msg):
        """Transforms Path message into a 2D array
        where each row represents a point (x, y) of rough path.
        
        Arguments:
            msg {Path} -- nav_msgs/Path
        
        Returns:
            2D array -- 2D array of points that form the path
        """

        vecx = []
        vecy = []

        for pathmsg in msg.poses:
            vecx.append(pathmsg.pose.position.x)
            vecy.append(pathmsg.pose.position.y)
                
        pos = np.array((vecx, vecy)).T

        # sometimes a doubled point is published which causes troubles later on
        # in perpendicular_at_point (division by 0) -> these points are deleted
        
        pos_row, pos_col = pos.shape

        doubled_idx = []
        for i in range(pos_row - 1):
            if ((pos[i, 0] == pos[i + 1, 0]) and (pos[i, 1] == pos[i + 1, 1])):
                doubled_idx.append(i + 1)
        pos = np.delete(pos, doubled_idx, 0)

        return pos, True

    def path_seed(self, path, num_of_inbetween_points):
        """Adds extra points between two points that form the path. Points are
        added equidistantly. If points are added parameters should be
        re-evaluated.
        
        Arguments:
            path {2D array} -- 2D array with first column of x points
        and second column of y points
            num_of_inbetween_points {Int} -- Number of points to be added
            between two points of the original path
        
        Returns:
            [2D array] -- Path with added points
        """
                
        pos_x = path[:, 0]
        pos_y = path[:, 1]
                
        len_vec_rough = len(pos_x) - 1
                
        x_dn = np.zeros(len_vec_rough*num_of_inbetween_points + len_vec_rough + 1)
        y_dn = np.zeros(len_vec_rough*num_of_inbetween_points + len_vec_rough + 1)
                
        for i in range(len_vec_rough + 1):
            pos_vec1 = i*(num_of_inbetween_points + 1)
            x_dn[pos_vec1] = pos_x[i]
            y_dn[pos_vec1] = pos_y[i]
        
        for i in range(len_vec_rough):
            dxn = (pos_x[i + 1] - pos_x[i])/(num_of_inbetween_points + 1)
            dyn = (pos_y[i + 1] - pos_y[i])/(num_of_inbetween_points + 1)
                    
            for j in range(num_of_inbetween_points):
                pos_vec2 = i*num_of_inbetween_points + j + i
                x_dn[pos_vec2 + 1] = x_dn[pos_vec2] + dxn
                y_dn[pos_vec2 + 1] = y_dn[pos_vec2] + dyn
                
        path_seed = np.array((x_dn, y_dn)).T
                
        return path_seed, True

    def inc_costmap_check(self, path, costmap_array, costmap_update_array,
                          cmap_update, threshold, delta_t, tmax_limit):
        """Generates a 2D array of distances in the directions perpendicular
        to the path with respect to the costmap. These distances form the area
        where path is smoothed.
        
        Arguments:
            path {2D array} -- 2D array with first column of x points
        and second column of y points
            costmap_array {Array} -- Array with information about the costmap
            costmap_update_array {Array} -- Array with information about the
            costmap updates
            cmap_update {Bool} -- Indicates availability of
            costmap update information
            threshold {Int} -- Ranges form 0-100 and sets the limit of the
            smoothing area
            delta_t {Float} -- Smoothing area increment
            tmax_limit {Float} -- Maximum limit of smoothing area width
        
        Returns:
            [2D array] -- 2D array with first column of distances to the left
            of the path and the second column of distances to the right of the
            path
        """

        # costmap_array unpacking:
        costmap_mat = costmap_array[0]
        map_x = costmap_array[1]
        map_y = costmap_array[2]
        map_height = costmap_array[3]
        map_width = costmap_array[4]
        map_res = costmap_array[5]
        # costmap_update_array unpacking:
        costmap_mat_update = costmap_update_array[0]
        map_x_update = costmap_update_array[1]
        map_y_update = costmap_update_array[2]
            
        perp_at_point = perpendicular_at_point(path)

        # coordinate correction
        COORD_CORR_ROW = 0.5
        COORD_CORR_COLUMN = 0.5
                
        t_left = np.zeros_like(path[:, 0])
                
        # costmap_update
        if (cmap_update == True):
            costmap_mat = costmap_mat_update_func(
                costmap_mat,
                costmap_mat_update,
                map_x_update,
                map_y_update)
        else:
            pass

        for i in range(len(t_left)):
            left_border_cost = 0
            n_left = 1
                    
            while (left_border_cost < threshold):
                t_left[i] = delta_t*n_left

                left_border_point = path[i, :] + \
                    t_left[i] * perp_at_point[i, :]
                left_border_point_x = left_border_point[0]
                left_border_point_y = left_border_point[1]

                costmap_mat_row = map_height - (abs(map_y)/map_res) - \
                    (left_border_point_y/map_res) - COORD_CORR_ROW
                costmap_mat_column = (abs(map_x)/map_res) +  \
                    (left_border_point_x/map_res) - COORD_CORR_COLUMN
                                        
                if (costmap_mat_row >= map_height):
                    costmap_mat_row = map_height - 1
                if (costmap_mat_row < 0):
                    costmap_mat_row = 0
                            
                if (costmap_mat_column >= map_width):
                    costmap_mat_column = map_width - 1
                if (costmap_mat_column < 0):
                    costmap_mat_column = 0

                costmap_mat_row = int(round(costmap_mat_row))
                costmap_mat_column = int(round(costmap_mat_column))

                left_border_cost = \
                costmap_mat[costmap_mat_row, costmap_mat_column]
                n_left += 1
                    
                if (t_left[i] > tmax_limit):
                    t_left[i] = tmax_limit
                    break
                        
        t_right = np.zeros_like(path[:, 0])
                
        for i in range(len(t_right)):
            right_border_cost = 0
            n_right = 1
                    
            while (right_border_cost < threshold):
                t_right[i] = delta_t*n_right
                    
                right_border_point = path[i, :] - \
                    t_right[i] * perp_at_point[i, :]
                right_border_point_x = right_border_point[0]
                right_border_point_y = right_border_point[1]
                    
                costmap_mat_row = map_height - (abs(map_y)/map_res) - \
                    (right_border_point_y/map_res) - COORD_CORR_ROW
                costmap_mat_column = (abs(map_x)/map_res) + \
                    (right_border_point_x/map_res) - COORD_CORR_COLUMN
                                     
                if (costmap_mat_row >= map_height):
                    costmap_mat_row = map_height - 1
                if (costmap_mat_row < 0):
                    costmap_mat_row = 0
                            
                if (costmap_mat_column >= map_width):
                    costmap_mat_column = map_width - 1
                if (costmap_mat_column < 0):
                    costmap_mat_column = 0
                
                costmap_mat_row = int(round(costmap_mat_row))
                costmap_mat_column = int(round(costmap_mat_column))

                right_border_cost = \
                costmap_mat[costmap_mat_row, costmap_mat_column]
                n_right += 1
                    
                if (t_right[i] > tmax_limit):
                    t_right[i] = tmax_limit
                    break
                        
        t_left = t_left
        t_right = t_right
                        
        path_width = np.array((t_left, t_right)).T - delta_t

        return path_width, True
    
    def path_smooth(self, path, path_width, mass,
                    stiffness, num_of_iter, full_path=True):
        """Path smoothing algorithm where points are represented as mass
        points that are conneceted with massless, non-deformable beams. Beams
        are interconnected with torsional springs at the place of path (mass)
        points. A smooth path is achieved as a result of aforementioned
        mass-spring system in the state of minimal potential energy stored in
        the springs. Paramter values are a function of distances between path
        points (be careful if using path_seed()) and consequently a function
        of costmap resolution.
        Source: http://vamos.sourceforge.net/computer-controlled-cars/node2.html
        
        Arguments:
            path {2D array} -- 2D array with first column of x points
        and second column of y points
            path_width {2D array} -- 2D array with first column of distances
            to the left of the path and the second column of distances to the
            right of the path
            mass {Float} -- Mass of the mass point
            stiffness {[Float]} -- Spring stiffnes
            num_of_iter {[Int]} -- Number of algorith iterations
        
        Keyword Arguments:
            full_path {Bool} -- If True all points of the path are considered,
            full path is not needed with the last three points which are used
            to move an AGV to a final orientation (default: {True})
        
        Returns:
            [2D array] -- Smooth path
        """

        t_left = path_width[:, 0]
        t_right = path_width[:, 1]

        if ((full_path == False) and (len(path[:, 0]) > 5)):
            # the first and the last point of the path aren't considered
            path = path[1:, :]
            path = path[:-3, :]
            t_left = t_left[1:]
            t_left = t_left[:-3]
            t_right = t_right[1:]
            t_right = t_right[:-3]

        perp_at_point = perpendicular_at_point(path)
        dpos_path = np.diff(path, axis=0)
        if (len(perp_at_point[:, 0]) >= 2):
            perp_at_point[0, :] = dpos_path[0, :]
            perp_at_point[-1, :] = dpos_path[-1, :]
        else:
            pass
                
        poso = np.copy(path)
        ts = np.zeros(len(path))
        v = np.zeros(len(path))

        for iter in range(num_of_iter):
            if (mass == 0.0):
                # print("error - division by zero")
                break
            else:
                pass

            dpos = np.diff(poso, axis=0)
            ddist = np.sqrt(np.sum(dpos ** 2, axis=1))
            dpos_n = dpos / ddist[:, np.newaxis]

            cross_at_point = np.empty(len(path))
            cross_at_point[0] = 0
            cross_at_point[1:-1] = np.cross(dpos_n[1:], -dpos_n[:-1])
            cross_at_point[-1] = 0

            F = np.zeros(len(path))
            for i in range(1, len(path) - 1):
                F[i + 1] += -cross_at_point[i]
                F[i] += 2 * cross_at_point[i]
                F[i - 1] += -cross_at_point[i]

            v += F / mass - stiffness * v
            ts += v
                    
            for i in range(len(t_left)):
                ts[i] = np.clip(ts[i], -t_right[i], t_left[i])

            poso = path + ts[:, np.newaxis] * perp_at_point

        return poso, True

    def path_write(self, path_smooth, path_rough, msg):
        """A function used to generate nav_msgs/Path message.
        
        Arguments:
            path_smooth {2D array} -- 2D array of smooth path
            path_rough {2D array} -- 2D array of rough path
            msg {Path} -- nav_msgs/Path
        
        Returns:
            [Path] -- A message of type Path
        """
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        self.move.header.stamp = h.stamp
        self.move.header.frame_id = 'odom'
                
        path_smooth[0, :] = path_rough[0, :]
        path_smooth[-3, :] = path_rough[-3, :]
        path_smooth[-2, :] = path_rough[-2, :]
        path_smooth[-1, :] = path_rough[-1, :]
                
        m_poses = []
        posox = path_smooth[:, 0]
        posoy = path_smooth[:, 1]
                
        orientation_z = []
        orientation_w = []
                
        for pathmsg in msg.poses:
            orientation_z.append(pathmsg.pose.orientation.z)
            orientation_w.append(pathmsg.pose.orientation.w)
                
        for i in range(len(posox) - 1):
            m_poses.append(PoseStamped(self.move.header, Pose(Point(posox[i],
                           posoy[i], 0), Quaternion(0, 0, 0, 1))))
                
        m_poses.append(PoseStamped(self.move.header, Pose(Point(posox[-1],
                       posoy[-1], 0), Quaternion(0, 0, orientation_z[-1],
                                                 orientation_w[-1]))))
        self.move.poses = m_poses
                
        return self.move

    def path_zone_write(self, path_rough, path_width, path_side, msg):
        """A function used to generate nav_msgs/Path message for the area
        if which path smoothing is relevant. Used for visualisation.
        
        Arguments:
            path_rough {2D array} -- 2D array of smooth path
            path_width {2D array} -- 2D array with first column of distances
            to the left of the path and the second column of distances to
            the right of the path
            path_side {Str} -- "left" for left side of the zone and
            "right" for the right zone
            msg {Path} -- nav_msgs/Path
        
        Returns:
            [Path] -- A message of type Path
        """

        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        self.move.header.stamp = h.stamp
        self.move.header.frame_id = 'odom'
            
        zone = np.zeros_like(path_rough)
        perp_at_point = perpendicular_at_point(path_rough)

        if (path_side == "left"):
            t_left = path_width[:, 0]
            for i in range(len(t_left)):
                zone[i, :] = path_rough[i, :] + t_left[i] * perp_at_point[i, :]
        elif (path_side == "right"):
            t_right = path_width[:, 1]
            for i in range(len(t_right)):
                zone[i, :] = path_rough[i, :] - t_right[i] * perp_at_point[i, :]
                
        m_poses = []
        zone_x = zone[:, 0]
        zone_y = zone[:, 1]
                
        for i in range(len(zone_x)):
            m_poses.append(PoseStamped(self.move.header, Pose(Point(zone_x[i],
                           zone_y[i], 0), Quaternion(0, 0, 0, 1))))
            self.move.poses = m_poses
                    
        return self.move

    def set_goal_movebase_client(self, move_x=0.0, move_y=0.0):
        """Sets a goal which consequently returns a path.
        Used in parameter initialisation.
        
        Keyword Arguments:
            move_x {Float} -- x coordinate of goal (default: {0.0})
            move_y {Float} -- y coordinate of goal (default: {0.0})
        """

        goal_sent = False

        while(1):
            if (goal_sent == False):

                client = actionlib.SimpleActionClient('move_base',
                                                      MoveBaseAction)
                client.wait_for_server()

                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose.position.x = move_x
                goal.target_pose.pose.position.y = move_y
                goal.target_pose.pose.position.z = 0.0
                goal.target_pose.pose.orientation.y = 0.0
                goal.target_pose.pose.orientation.x = 0.0
                goal.target_pose.pose.orientation.y = 0.0
                goal.target_pose.pose.orientation.w = 1.0

                client.send_goal(goal)

                start_time = time.time()
                while(1):
                    end_time = time.time()
                    delta_time = end_time - start_time
                    if (delta_time > 0.1):
                        goal_sent = True
                        break
                    else:
                        pass
            else:
                break

    def cancel_goal_movebase_client(self):
        """Used for goal cancellation after a path is procured.
        """

        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client.cancel_all_goals()

    def get_random_point_map_costmap_check(self, threshold, costmap_array,
                                           costmap_update_array, cmap_update):
        """Gets a random point with respect to the costmap. Used with
        set_goal_movebase_client() in order to get a path used in
        parameter evaluation.
        
        Arguments:
            threshold {Int} -- Ranges form 0-100 and sets the limit of the
            smoothing area
            costmap_array {Array} -- Array with information about the costmap
            costmap_update_array {Array} -- Array with information about the
            costmap updates
            cmap_update {Bool} -- Indicates availability of
            costmap update information
        
        Returns:
            [Tuple] -- A point that doesn't lie on a costmap
        """
        # TODO: rand_x, rand_y -> doloci glede na dimenzije mapa
        #preverjanje ustreznosti tocke z ozirom na costmap
        #razlog: ce je goal poslan na costmap ni generacije poti

        # costmap_array unpacking:
        costmap_mat = costmap_array[0]
        map_x = costmap_array[1]
        map_y = costmap_array[2]
        map_height = costmap_array[3]
        map_width = costmap_array[4]
        map_res = costmap_array[5]
        # costmap_update_array unpacking:
        costmap_mat_update = costmap_update_array[0]
        map_x_update = costmap_update_array[1]
        map_y_update = costmap_update_array[2]

        rand_x = np.random.random_integers(-5, 5)
        rand_y = np.random.random_integers(-5, 5)
                
        COORD_CORR_ROW = 0.5
        COORD_CORR_COLUMN = 0.5
                
        if (cmap_update == True):
            costmap_mat = costmap_mat_update_func(costmap_mat,
                                                  costmap_mat_update,
                                                  map_x_update,
                                                  map_y_update)
        else:
            pass

        costmap_mat_row = map_height - (abs(map_y)/map_res) - \
            (rand_y/map_res) - COORD_CORR_ROW
        costmap_mat_column = (abs(map_x)/map_res) + \
            (rand_x/map_res) - COORD_CORR_COLUMN
                
        costmap_mat_row = int(round(costmap_mat_row))
        costmap_mat_column = int(round(costmap_mat_column))
                                 
        if (costmap_mat_row >= map_height):
            costmap_mat_row = map_height - 1
        if (costmap_mat_row < 0):
            costmap_mat_row = 0
                    
        if (costmap_mat_column >= map_width):
            costmap_mat_column = map_width - 1
        if (costmap_mat_column < 0):
            costmap_mat_column = 0

        point_costmap_value = costmap_mat[costmap_mat_row, costmap_mat_column]

        if (point_costmap_value <= threshold):
            return rand_x, rand_y
        else:
            return self.get_random_point_map_costmap_check(threshold,
                                                           costmap_array,
                                                           costmap_update_array,
                                                           cmap_update)

    # def path_angles(self, path, full_path=False):
    #     """Used for calculating angles between sections of the path.
    #     Uses atan2.
        
    #     Arguments:
    #         path {2D array} -- 2D array with first column of x points
    #     and second column of y points
        
    #     Keyword Arguments:
    #         full_path {Bool} -- If True full path is considered. False is used
    #         to ignore last three angles which are usualy large because of the
    #         orientation in the last point
    #          (default: {False})
        
    #     Returns:
    #         [Array] -- Array of absolute angles in degrees
    #     """

    #     # print("path shape:", path.shape)
    #     vec_mid = np.zeros_like(path)
    #     for i in range(len(path[:, 0]) - 1):
    #         vec_mid[i] = path[i, :] - path[i + 1, :]

    #     angles = np.zeros(len(path[:, 0]) - 2)
    #     for i in range(len(path[:, 0]) - 2):
    #         angles[i] = np.math.atan2(np.linalg.det([vec_mid[i],
    #                                   vec_mid[i + 1]]), np.dot(vec_mid[i],
    #                                   vec_mid[i + 1]))

    #     angles = np.degrees(angles)
    #     # print("ang shape:", angles.shape)
        
    #     if ((len(angles) > 3) and (full_path == False)):
    #         angles = angles[:-3]
    #     else:
    #         pass

    #     angles = np.abs(angles)

    #     return angles

    def path_angles(self, path, full_path=False):
        """Used for calculating angles between sections of the path.
        Uses atan2.
        
        Arguments:
            path {2D array} -- 2D array with first column of x points
        and second column of y points
        
        Keyword Arguments:
            full_path {Bool} -- If True full path is considered. False is used
            to ignore last three angles which are usualy large because of the
            orientation in the last point
             (default: {False})
        
        Returns:
            [Array] -- Array of absolute angles in degrees
        """

        # print("path shape:", path.shape)
        pth_row, pth_col = path.shape

        if pth_row > 1:
            vec_mid = np.zeros_like(path)
            for i in range(len(path[:, 0]) - 1):
                vec_mid[i] = path[i, :] - path[i + 1, :]

            angles = np.zeros(len(path[:, 0]) - 2)
            for i in range(len(path[:, 0]) - 2):
                angles[i] = np.math.atan2(np.linalg.det([vec_mid[i],
                                        vec_mid[i + 1]]), np.dot(vec_mid[i],
                                        vec_mid[i + 1]))

            angles = np.degrees(angles)
            # print("ang shape:", angles.shape)
            
            if ((len(angles) > 3) and (full_path == False)):
                angles = angles[:-3]
            else:
                pass

            angles = np.abs(angles)

            return angles
        else:
            angles = 0
            return angles

    def multistart_optimization(self, params, path, costmap_array,
                                costmap_update_array, cmap_update, threshold,
                                delta_t, tmax_limit, num_of_inbetween_points,
                                plot_flg=False):
        """Used to determine the optimal path smoothing paramteres (mass, stiffness).
        When a suitably "bad" (bad in the sense of absolute angles mean value
        and standard deviation) path is found, it can be used to determine path
        smoothing paramteres. Initial guesses must be aproximated and a more
        global optimization can be done. The algorithm used for optimization
        is Nelder-Mead or better known as Simplex optimization. It's main
        advantage is that it can minimize a function that isn't mathematically
        defined.
 
        Arguments:
            params {2D array} -- 2D array where fist column represents
            mass initial guesses and the second column represents
            stiffness initial guesses
            path {2D array} -- 2D array of a suitalby "bad" path
            costmap_array {Array} -- Array with information about the costmap
            costmap_update_array {Array} -- Array with information about the
            costmap updates
            cmap_update {Bool} -- Indicates availability of
            costmap update information
            threshold {Int} -- Ranges form 0-100 and sets the limit of the
            smoothing area
            delta_t {Float} -- Smoothing area increment
            tmax_limit {Float} -- Maximum limit of smoothing area width
            num_of_inbetween_points {Int} -- Number of points to be added
            between two points of the original path
        
        Keyword Arguments:
            plot_flg {Bool} -- True if you want the results plotted
            (default: {False})
        
        Returns:
            [Array] -- Array of optimal paramters, which are mass and stiffness
        """

        from scipy.optimize import minimize

        param_row, param_col = params.shape
        
        results = []
        res_params = []

        def criterion_func_mean(params):
            # TODO: currently an area of constant width is used
            mass, stiffness = params
            path_s, check = self.path_seed(path, num_of_inbetween_points)
            # path_width, check = self.inc_costmap_check(path_s,
            #                                            costmap_array,
            #                                            costmap_update_array,
            #                                            cmap_update,
            #                                            threshold,
            #                                            delta_t,
            #                                            tmax_limit)
            path_width = np.ones_like(path_s)*2  # area of constant width
            pth, check = self.path_smooth(path_s, path_width, mass, stiffness,
                                          threshold, True)
            abs_angles = self.path_angles(pth)
            abs_mean_angles = np.mean(abs_angles)
            return abs_mean_angles

        if plot_flg == True:
            plt.figure(figsize=(12, 12))
            plt.subplot(1, param_row + 1, 1)
            plt.plot(path[:, 0], path[:, 1], 'r')
            plt.axis('scaled')

        for i in range(param_row):
            res = minimize(criterion_func_mean, params[i],
                           method='nelder-mead',
                           options={'maxiter': 100, 'disp': True})
            results.append(res.fun)
            res_params.append(res.x)

            pars = res.x

            path_S, check = self.path_seed(path, num_of_inbetween_points)
            path_width_t, check = self.inc_costmap_check(path_S, costmap_array,
                                                         costmap_update_array,
                                                         cmap_update,
                                                         threshold,
                                                         delta_t,
                                                         tmax_limit)
            pth_t, check = self.path_smooth(path_S, path_width_t,
                                            pars[0], pars[1], 50, True)
            
            if plot_flg == True:
                plt.subplot(1, param_row + 1, i+2)
                plt.plot(pth_t[:, 0], pth_t[:, 1])
                plt.axis('scaled')

        if plot_flg == True:
            plt.show()

        res_params = np.array(res_params)
        
        # print('results:', results)
        # print('res_params:', res_params)

        opt_param_idx = np.argmin(results)
        opt_param = res_params[opt_param_idx]

        # optimisation often results in mass that is a bit too low
        # 50% increase in value greatly increases evualuation efficiency
        opt_param[0] = opt_param[0]*1.5

        return opt_param

    def no_angles_larger_than_ratio(self, angles, angle_limit):
        """Return the ratio of angles larger than selected enagle.
        Angles are in degrees.
        
        Arguments:
            angles {Array} -- Array of angles
            angle_limit {Float} -- Angle limit
        
        Returns:
            [Float] -- Ratio of angles larger than limit
        """

        if (angles.__class__ == np.ndarray):
            no_of_larger_angles = 0
            for i in range(len(angles)):
                if (angles[i] > angle_limit):
                    no_of_larger_angles += 1
                else:
                    pass
            if (no_of_larger_angles >= 1):
                angles_ratio = no_of_larger_angles/len(angles)
                return angles_ratio
            else:
                angles_ratio = 0
                return angles_ratio
        else:
            angles_ratio = 0
            return angles_ratio
