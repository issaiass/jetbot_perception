#include <jetbot_detection/object_detection.hpp>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "Object_Detection");
  Object_Detection od;
  ros::spin();
  return 0;
}