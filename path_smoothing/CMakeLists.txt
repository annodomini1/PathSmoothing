cmake_minimum_required(VERSION 2.8.3)
project(path_smoothing)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  roscpp
  rospy
)

generate_dynamic_reconfigure_options(
   cfg/Params.cfg
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
