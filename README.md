# path_smoothing

In case that output path from navigation stack is rough (low resolution map) this node is used for path smoothing.

## Required dependencies
rospy
dynamic_reconfigure
numpy
std_msgs
nav_msgs
geometry_msgs

### Parameters
* `delta_width (double)` Track width increment used to determine smoothing area with regard to costmap
* `num_of_iteratons (int)` Number of path_smoothing algorithm iterations
* `threshold (int)` Cost threshold is used to limit the smoothing area
* `width_max_limit (double)` Limit of the smoothing area
* `mean_angles_limit (double)` When evaluating parameters a curved path is needed, which is characterised by mean value of absolute path angles in degrees
* `std_angles_limit (double)` When evaluating parameters a curved path is needed, which is characterised by standard deviation of absolute path angles in degrees
* `path_length_limit (int)` Minimal length of path needed for evaluation

### Subscribed topics
* `/move_base/TrajectoryPlannerROS/global_plan (nav_msgs::Path)` topic for receiving global path
* `/move_base/global_costmap/costmap (nav_msgs::OccupancyGrid)` topic for receiving costmap information

### Published topics
* `/vamos (nav_msgs::Path)` topic for publishing smooth path
* `/vamos_zone_left (nav_msgs::Path)` topic for publishing left border of smoothing track, used for visualisation
* `/vamos_zone_right (nav_msgs::Path)` topic for publishing right border of smoothing track, used for visualisation

### Services

###USAGE

