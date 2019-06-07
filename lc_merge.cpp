#include <ros/ros.h>
#include <math.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <fstream>

using namespace std;

#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 720

typedef pcl::PointXYZI  PointType;
pcl::PointCloud<PointType>::Ptr laserCloudIn;

ros::Publisher pub;

float r_t[16] = {       0.999049,  0.00912775  ,-0.0426374,    0.569389,
-0.00865455,    0.999899,   0.0112696,  -0.0721242,
   0.042736,  -0.0108898,    0.999027,  -0.0879596,
          0,           0,           0,           1};

float rot[9] = {       0.999049,  0.00912775  ,-0.0426374,
                        -0.00865455,    0.999899,   0.0112696,
                        0.042736,  -0.0108898,    0.999027};

float translation[3] = {0.569389, -0.0721242, -0.0879596};

float p_mat[12] = {647.419673, 0.000000, 517.989483,   0,
                0.000000, 647.419673, 310.535839,  0,
                0.000000, 0.000000, 1.000000,  0};

float cam_mat[9] = {647.419673, 0.000000, 517.989483,
                0.000000, 647.419673, 310.535839,
                0.000000, 0.000000, 1.000000};

float cam_mat_org[9] = {666.435662, 0.000000, 605.686260,
                    0.000000, 668.422861, 322.413398,
                    0.000000, 0.000000, 1.000000};

float dist_mat[5] = {-0.162663, 0.025235, -0.001717, 0.000203, 0.000000};

cv::Mat proj_matrix;
cv::Mat point_vector;
cv::Mat trans_matrix;
cv::Mat dist_matrix;
cv::Mat cam_matrix;
cv::Mat cam_matrix_org;
cv::Mat rvec;
cv::Mat tvec;

cv_bridge::CvImagePtr cv_ptr;
image_transport::Publisher lc_image_pub;
image_transport::Subscriber image_sub;

bool new_lidar_cloud = false;
bool new_image = false;
double frame_time = 0;
string frame_id;

uint32_t cloudSize = 0;

float board[] = {-46.3,-54.2,-5.0,-4.0,19.25};

pcl::PointCloud<PointType> transform(pcl::PointCloud<PointType> pc, float x, float y, float z, float rot_x, float rot_y, float rot_z)
{
	Eigen::Affine3f transf = pcl::getTransformation(x, y, z, rot_x, rot_y, rot_z);
	pcl::PointCloud<PointType> new_cloud;
	pcl::transformPointCloud(pc, new_cloud, transf);
	return new_cloud;
}

void project_lidar_points(){
    if(new_lidar_cloud == false || new_image == false){
        return;
    }

    new_lidar_cloud = false;
    new_image = false;

    int circle_cntr = 0;
    cloudSize = laserCloudIn->points.size();

    cv::Mat outputImage;
    cv::Mat proj_pts;
    cv::Mat rvec_vec;
    cv::Rodrigues(rvec, rvec_vec);
    cv::undistort(cv_ptr->image, outputImage, cam_matrix, dist_matrix);

    for( size_t i=0; i<cloudSize; i++){

        if(laserCloudIn->points[i].z < 1 || laserCloudIn->points[i].z > 2) continue;

        std::vector<cv::Point3f> lid_pts;
        cv::Point3f lid_pt(laserCloudIn->points[i].x, laserCloudIn->points[i].y, laserCloudIn->points[i].z);
        lid_pts.push_back(lid_pt);
        cv::projectPoints(lid_pts, rvec_vec, tvec, cam_matrix, dist_matrix, proj_pts);
        short x_ = short( proj_pts.at<float>(0,0) );
        short y_ = short( proj_pts.at<float>(0,1) );

        if(y_ >= IMAGE_HEIGHT || x_ >= IMAGE_WIDTH || x_ < 0 || y_ < 0){
            continue;
        }
        cv::circle(outputImage, cv::Point(x_,y_), 4, cv::Scalar(0,255,0), 1, 8, 0);
        circle_cntr++;
    }

    std::vector<cv::Point3f> poi;
    cv::Point3f pt(-0.373926, 0.0987597, 1.77497);
    poi.push_back(pt);
    cv::projectPoints(poi, rvec_vec, tvec, cam_matrix, dist_matrix, proj_pts);

    short x_ = short( proj_pts.at<float>(0,0) );
    short y_ = short( proj_pts.at<float>(0,1) );
    if(y_ >= IMAGE_HEIGHT || x_ >= IMAGE_WIDTH || x_ < 0 || y_ < 0){
        ;
    }
    else{
        cv::circle(outputImage, cv::Point(x_,y_), 4, cv::Scalar(255,0,0), 4, 8, 0);
    }

    std::vector<cv::Point3f> cam_poi;
    cv::Point3f cam_pt(-0.257861, -0.350891, 1.66645);
    cam_poi.push_back(cam_pt);
    cv::Rodrigues(cv::Mat::eye(3,3,CV_32FC1), rvec_vec);
    cv::projectPoints(cam_poi, rvec_vec, cv::Mat::zeros(3,1,CV_32FC1), cam_matrix_org, dist_matrix, proj_pts);

    x_ = short( proj_pts.at<float>(0,0) );
    y_ = short( proj_pts.at<float>(0,1) );
    if(y_ >= IMAGE_HEIGHT || x_ >= IMAGE_WIDTH || x_ < 0 || y_ < 0){
        ;
    }
    else{
        cv::circle(outputImage, cv::Point(x_,y_), 4, cv::Scalar(255,255,0), 4, 8, 0);
    }

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_ptr->image).toImageMsg();
    lc_image_pub.publish(msg);
    ROS_INFO("cicle cntr: %d\n",circle_cntr);
    cv::imshow("view", outputImage);
    cv::waitKey(30);
}

void get_lidar_cloud(const sensor_msgs::PointCloud2ConstPtr& msg){
    laserCloudIn->clear();
    frame_time = msg->header.stamp.toSec();
    frame_id = msg->header.frame_id;
    pcl::fromROSMsg(*msg, *laserCloudIn);
    *laserCloudIn = transform(*laserCloudIn, 0, 0, 0, 1.57, -1.57, 0);
    new_lidar_cloud = true;
}

void imageCb(const sensor_msgs::ImageConstPtr& msg){
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    new_image = true;
}

int main(int argc, char **argv){

    ros::init(argc, argv, "Lidar_on_Image");

    ros::NodeHandle n;
    image_transport::ImageTransport it(n);

    ros::Subscriber sub_lidar = n.subscribe("velodyne_points",40,get_lidar_cloud);
    image_sub = it.subscribe("zed/zed_node/left_raw/image_raw_color", 1,imageCb);
    lc_image_pub = it.advertise("lc_image",1);

    proj_matrix = cv::Mat(3, 4, CV_32FC1, p_mat);
    point_vector = cv::Mat(4, 1, CV_32FC1, cv::Scalar::all(0));
    trans_matrix = cv::Mat(4, 4, CV_32FC1, r_t);
    dist_matrix = cv::Mat(1,5, CV_32FC1, dist_mat);
    cam_matrix = cv::Mat(3,3, CV_32FC1, cam_mat);
    cam_matrix_org = cv::Mat(3,3, CV_32FC1, cam_mat_org);
    rvec = cv::Mat(3,3, CV_32FC1, rot);
    tvec = cv::Mat(3,1, CV_32FC1, translation);

    cout << "P mat" << endl << proj_matrix << endl;
    cout << "T mat" << endl << trans_matrix << endl;

    cv_ptr = cv_bridge::CvImagePtr(new cv_bridge::CvImage);

    laserCloudIn.reset(new pcl::PointCloud<PointType>());
    cv::namedWindow("view");
    cv::startWindowThread();
    ros::Rate loop_rate(20);

    ROS_INFO("\nLC_merge node started\n");

    while(ros::ok()){

        project_lidar_points();
        ros::spinOnce();

        loop_rate.sleep();
    }

    cv::destroyWindow("view");
    return 0;
}