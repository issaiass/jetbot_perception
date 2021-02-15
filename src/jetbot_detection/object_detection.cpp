#include <jetbot_detection/object_detection.hpp>


Object_Detection::Object_Detection(): nh_("~"), it_(nh_)
{
    // Load CV params
    nh_.param("opencv/enable_floating_window", cv_enable_floating_window_, bool(false));
    nh_.param("opencv/image_width", cv_image_width_, 320);
    nh_.param("opencv/image_height", cv_image_height_, 240);
    nh_.param("opencv/window_name", cv_window_name_, std::string("OPENCV_WINDOW"));
    nh_.param("opencv/wait_key", cv_wait_key_, int(3));

    // Load topic names
    nh_.param("topics/pub_topic", pub_topic_, std::string("/detected_objects"));
    nh_.param("topics/pub_topic_info", pub_topic_info_, std::string("/detected_objects_info"));
    nh_.param("topics/sub_topic", sub_topic_, std::string("/usb_cam/image_raw"));

    // Load NN params
    nh_.param("neuralnet/image_width", nn_image_width_, int(256));
    nh_.param("neuralnet/image_height", nn_image_height_, int(256));

    nh_.param("neuralnet/scale_factor", nn_scale_factor_, double(0.5));
    nh_.param("neuralnet/confidenceThreshold", nn_confidenceThreshold_, float(0.75));
    /*nh_.param("neuralnet/meanVal", nn_meanval_);
    nn_meanval__ = cv::Scalar(static_cast<double>(nn_meanval_[0]), 
                              static_cast<double>(nn_meanval_[1]), 
                              static_cast<double>(nn_meanval_[2]));*/    
    nh_.param("neuralnet/meanValR", nn_meanvalr_, double(127.5));
    nh_.param("neuralnet/meanValG", nn_meanvalg_, double(127.5));
    nh_.param("neuralnet/meanValB", nn_meanvalb_, double(127.5));    
    nn_meanval__ = cv::Scalar(nn_meanvalr_, nn_meanvalg_, nn_meanvalb_);
    nh_.param<std::string>("neuralnet/configfile", nn_configfile_,
    "/home/robot/jetbot_perception_ws/src/jetbot_perception0/models/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pbtxt");
    nh_.param<std::string>("neuralnet/modelfile", nn_modelfile_,
    "/home/robot/jetbot_perception_ws/src/jetbot_perception1/models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb");
    nh_.param<std::string>("neuralnet/classfile", nn_classfile_,
    "/home/robot/jetbot_perception_ws/src/jetbot_perception2/models/ssd_mobilenet_v2_coco_2018_03_29/coco_class_labels.txt");

    cv::namedWindow(cv_window_name_);

    image_sub_ = it_.subscribe(sub_topic_, 1, &Object_Detection::imageCb, this);
    image_pub_ = it_.advertise(pub_topic_, 1, false);
    boundingBoxesPublisher_ = nh_.advertise<jetbot_msgs::BoundingBoxes>(pub_topic_info_, 1, false);

    // Load NN
    net = readNetFromTensorflow(nn_modelfile_.c_str(), nn_configfile_.c_str());

    // Load file into vector
    std::ifstream ifs(nn_classfile_.c_str());
    std::string line;
    while (getline(ifs, line))
    {
      nn_classes.push_back(line);
    }

}

Object_Detection::~Object_Detection()
{
    destroyWindow(cv_window_name_);
}

cv::Mat Object_Detection::object_detection(cv::Mat src)
{
    cv::Mat inputBlob = cv::dnn::blobFromImage(src, nn_scale_factor_, cv::Size(nn_image_width_, nn_image_height_),
                                               nn_meanval__, true, false);
    net.setInput(inputBlob);
    try 
    {    
      cv::Mat detection = net.forward(nn_frame_id_);
      cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
      //cv::Mat detectionMat = detection.reshape(1, detection.size[2]);
      return detectionMat;      
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("exception: %s", e.what());
    }
}

void Object_Detection::imageCb(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  namespace enc = sensor_msgs::image_encodings;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat detections = object_detection(cv_ptr->image);
  object_labeling(cv_ptr->image, detections, nn_confidenceThreshold_);
  if (cv_enable_floating_window_) {
    //double size_factor = 0.5;
    //resize(src, src, Size(), size_factor, size_factor, INTER_LINEAR);
    cv::Mat dst;
    cv::resize(cv_ptr->image, dst, cv::Size(cv_image_width_, cv_image_height_), 0, 0, CV_INTER_LINEAR);
    cv::imshow(cv_window_name_, dst);
    cv::waitKey(cv_wait_key_);
  }    
  image_pub_.publish(cv_ptr->toImageMsg());
}

void Object_Detection::object_labeling(cv::Mat &img, cv::Mat detections, float threshold) 
{
    for (int i = 0; i < detections.rows; i++){
        int classId = detections.at<float>(i, 1);
        float score = detections.at<float>(i, 2);

        // Recover original cordinates from normalized coordinates
        int x = static_cast<int>(detections.at<float>(i, 3) * img.cols);
        int y = static_cast<int>(detections.at<float>(i, 4) * img.rows);
        int w = static_cast<int>(detections.at<float>(i, 5) * img.cols - x);
        int h = static_cast<int>(detections.at<float>(i, 6) * img.rows - y);

        // Check if the detection is of good quality
        if (score > threshold){
            object_text(img, nn_classes[classId].c_str(), x, y);
            cv::rectangle(img, cv::Point(x,y), cv::Point(x+w, y+h), cv::Scalar(127,127,127), 2);

            boundingBox_.probability = score;
            boundingBox_.x = x;
            boundingBox_.y = y;
            boundingBox_.w = w;
            boundingBox_.h = h;
            boundingBox_.id = classId;
            boundingBox_.Class = nn_classes[classId];
            boundingBoxesResults_.bounding_boxes.push_back(boundingBox_);
        }
    }
    boundingBoxesResults_.header.stamp = ros::Time::now();
    boundingBoxesResults_.header.frame_id = nn_frame_id_;    
    boundingBoxesPublisher_.publish(boundingBoxesResults_);
    boundingBoxesResults_.bounding_boxes.clear();
}

void Object_Detection::object_text(cv::Mat& img, std::string text, int x, int y)
{
    // Get text size
    int baseLine;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseLine);
    // Use text size to create a black rectangle
    rectangle(img, Point(x,y-textSize.height-baseLine), Point(x+textSize.width,y+baseLine),
             Scalar(0,0,0),-1);
    // Display text inside the rectangle
    putText(img, text, Point(x,y-5), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,255), 1, LINE_AA);
}