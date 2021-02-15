cmake_minimum_required(VERSION 3.0.2)
project(jetbot_perception)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  jetbot_msgs
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_lib
  STATIC
  include/jetbot_detection/object_detection.hpp 
  src/jetbot_detection/object_detection.cpp 
)

add_executable(perception_detection src/perception_detection.cpp)
target_link_libraries(perception_detection
  ${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES} 
)

add_executable(perception_subscriber src/perception_subscriber.cpp)
target_link_libraries(perception_subscriber
  ${catkin_LIBRARIES}
)