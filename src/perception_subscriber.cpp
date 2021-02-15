#include <ros/ros.h>
#include <jetbot_detection/object_detection.hpp>

void objectListCallback(const jetbot_msgs::BoundingBoxes::ConstPtr& msg) {
    ROS_INFO( "***** New object list *****");
    for(int i=0; i<msg->bounding_boxes.size();i++)
    {
        ROS_INFO_STREAM( msg->bounding_boxes[i].Class
                         << " [class_id="
                         << msg->bounding_boxes[i].id
                         << "] - Pos. ["
                         << msg->bounding_boxes[i].x << ","
                         << msg->bounding_boxes[i].y << ","
                         << msg->bounding_boxes[i].w << ","                         
                         << msg->bounding_boxes[i].h << "] [pix]"
                         << " - Conf. "
                         << msg->bounding_boxes[i].probability
                         << "");
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "jetbot_object_detection_info");
    ros::NodeHandle nh;
    ros::Subscriber subObjList= nh.subscribe<jetbot_msgs::BoundingBoxes>("/detected_objects_info", 1, objectListCallback);
    ros::spin();
    return 0;
}