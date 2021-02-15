#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// jetbot_msgs
#include <jetbot_msgs/BoundingBox.h>
#include <jetbot_msgs/BoundingBoxes.h>


using namespace cv;
using namespace dnn;

class Object_Detection
{
public:
   Object_Detection();
  ~Object_Detection();

private:
  void imageCb(const sensor_msgs::ImageConstPtr& msg);
  cv::Mat object_detection(cv::Mat src);
  void object_labeling(cv::Mat &img, cv::Mat detections, float threshold = 0.25);
  void object_text(cv::Mat& img, std::string text, int x, int y);

  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  ros::Publisher boundingBoxesPublisher_;
  jetbot_msgs::BoundingBox boundingBox_;
  jetbot_msgs::BoundingBoxes boundingBoxesResults_;


  int cv_image_width_, cv_image_height_, cv_wait_key_;  
  bool cv_enable_floating_window_;
  std::string cv_window_name_;

  std::string pub_topic_;
  std::string pub_topic_info_;  
  std::string sub_topic_;

  std::string nn_configfile_;
  std::string nn_modelfile_;
  std::string nn_classfile_;
  std::vector<std::string> nn_classes;
  int nn_image_width_, nn_image_height_;
  double nn_scale_factor_;
  float nn_confidenceThreshold_;
  XmlRpc::XmlRpcValue nn_meanval_;
  double nn_meanvalr_, nn_meanvalg_, nn_meanvalb_;
  cv::Scalar nn_meanval__;
  std::string nn_classFile;
  Net net;
  std::string nn_frame_id_ = "detection_out";
};