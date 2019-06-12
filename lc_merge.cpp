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

/*
    input: Image, trans, rot, projection, distortion, points to be projected, image dim, color
 */
void overlay_points_on_image(   cv::Mat &image,
                                cv::Mat rot,
                                cv::Mat trans,
                                cv::Mat p_mat,
                                cv::Mat dist_mat,
                                std::vector<cv::Point3f> pts,
                                unsigned int img_height,
                                unsigned int img_width,
                                cv::Scalar color,
                                int thickness){

    cv::Mat proj_pts;
    cv::Mat rvec;
    cv::Rodrigues(rot, rvec);
    cv::projectPoints(pts, rvec, trans, p_mat, dist_mat, proj_pts);

    short x_,y_=0;
    for(size_t i=0; i < proj_pts.rows; i++){

        x_ = short( proj_pts.at<float>(i,0) );
        y_ = short( proj_pts.at<float>(i,1) );

        if(y_ < img_height && x_ < img_width && x_ > 0 && y_ > 0){
            cv::circle(image, cv::Point(x_,y_), 4, color, thickness, 8, 0);
        }
    }
}

void project_lidar_points(){
    if(new_lidar_cloud == false || new_image == false){
        return;
    }

    new_lidar_cloud = false;
    new_image = false;

    cloudSize = laserCloudIn->points.size();

    cv::Mat outputImage;

    cv::undistort(cv_ptr->image, outputImage, cam_matrix, dist_matrix);

    std::vector<cv::Point3f> lid_pts;

    for( size_t i=0; i<cloudSize; i++){
        /*
            Keep the points within a certain z distance
         */
        if(laserCloudIn->points[i].z < 1 || laserCloudIn->points[i].z > 2) continue;

        lid_pts.push_back( cv::Point3f(laserCloudIn->points[i].x, laserCloudIn->points[i].y, laserCloudIn->points[i].z) );
    }

    overlay_points_on_image(   outputImage,
                                rvec,
                                tvec,
                                cam_matrix,
                                dist_matrix,
                                lid_pts,
                                720,
                                1280,
                                cv::Scalar(0,255,0),
                                1 );

    std::vector<cv::Point3f> poi;
    poi.push_back(cv::Point3f(-0.725787, -0.300817, 1.75655));
    poi.push_back(cv::Point3f(-0.373926, 0.0987597, 1.77497));
    poi.push_back(cv::Point3f(-0.706638, 0.407004, 1.74148));
    poi.push_back(cv::Point3f(-1.08684, 0.00340891, 1.74881));
    poi.push_back(cv::Point3f(0.113457, -0.31275, 1.84271));
    poi.push_back(cv::Point3f(0.468701, 0.0889406, 1.90563));
    poi.push_back(cv::Point3f(0.137314, 0.406434, 1.81652));
    poi.push_back(cv::Point3f(-0.234975, -0.0053591, 1.76959));

    overlay_points_on_image(   outputImage,
                                rvec,
                                tvec,
                                cam_matrix,
                                dist_matrix,
                                poi,
                                720,
                                1280,
                                cv::Scalar(255,0,0),
                                4 );

    std::vector<cv::Point3f> cam_poi;
    cam_poi.push_back(cv::Point3f(-0.257861, -0.350891, 1.66645));
    cam_poi.push_back(cv::Point3f(0.111404, 0.0422095 ,1.6397));
    cam_poi.push_back(cv::Point3f(-0.223684, 0.35374 ,1.59211));
    cam_poi.push_back(cv::Point3f(-0.592948, -0.0393605 ,1.61885));
    cam_poi.push_back(cv::Point3f(0.614391 ,-0.356681 ,1.73359));
    cam_poi.push_back(cv::Point3f(0.970656, 0.0406231 ,1.8162));
    cam_poi.push_back(cv::Point3f(0.633394, 0.351058 ,1.77768));
    cam_poi.push_back(cv::Point3f(0.277129, -0.0462458 ,1.69507));

    overlay_points_on_image(   outputImage,
                                cv::Mat::eye(3,3,CV_32FC1),
                                cv::Mat::zeros(3,1,CV_32FC1),
                                cam_matrix,
                                dist_matrix,
                                cam_poi,
                                720,
                                1280,
                                cv::Scalar(255,255,0),
                                4 );


    /*  TO DO:
        * check the intensity value of that corresponding point on image
        * if its not a road point push it to another vector
        * publish vector
     */

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_ptr->image).toImageMsg();
    lc_image_pub.publish(msg);

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
    try{
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e){
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