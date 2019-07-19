#include <lc_merge.h>

using namespace std;

#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 720

typedef pcl::PointXYZI  PointType;
pcl::PointCloud<PointType>::Ptr laserCloudIn;

ros::Publisher pub;

cv::Mat *proj_matrix;
cv::Mat *camera_intrinsic;
cv::Mat *dist_matrix;
cv::Mat R_rect_mat;
cv::Size *image_size;

cv::Mat *rot_mat;
cv::Mat *t_mat;
cv::Mat T_mat;

cv_bridge::CvImagePtr cv_ptr;
image_transport::Publisher lc_image_pub;
image_transport::Subscriber image_sub;

bool new_lidar_cloud = false;
bool new_image = false;
double frame_time = 0;
string frame_id;

uint32_t cloudSize = 0;

double R_rect[] = {9.998817e-01, 1.511453e-02, -2.841595e-03, 0, -1.511724e-02, 9.998853e-01, -9.338510e-04,0, 2.827154e-03, 9.766976e-04, 9.999955e-01,0,0,0,0,1};

pcl::PointCloud<PointType> transform(pcl::PointCloud<PointType> pc, float x, float y, float z, float rot_x, float rot_y, float rot_z)
{
	Eigen::Affine3f transf = pcl::getTransformation(x, y, z, rot_x, rot_y, rot_z);
	pcl::PointCloud<PointType> new_cloud;
	pcl::transformPointCloud(pc, new_cloud, transf);
	return new_cloud;
}

bool parse_calibration_file(std::string calib_filename){

    sensor_msgs::CameraInfo camera_calibration_data;
    std::string camera_name = "camera";

    camera_calibration_parsers::readCalibrationIni(calib_filename, camera_name, camera_calibration_data);

    // Alocation of memory for calibration data
    camera_intrinsic    = new(cv::Mat)(3, 3, CV_64FC1);
    proj_matrix         = new(cv::Mat)(3, 4, CV_64FC1);
    dist_matrix         = new(cv::Mat)(5, 1, CV_64FC1);
    image_size          = new(cv::Size);

    image_size->width = camera_calibration_data.width;
    image_size->height = camera_calibration_data.height;

    for(size_t i = 0; i < 3; i++)
        for(size_t j = 0; j < 3; j++)
            camera_intrinsic->at<double>(i,j) = camera_calibration_data.K.at(3*i+j);

    for(size_t i = 0; i < 3; i++)
        for(size_t j = 0; j < 4; j++)
            proj_matrix->at<double>(i,j) = camera_calibration_data.P.at(4*i+j);

    for(size_t i = 0; i < 5; i++)
        dist_matrix->at<double>(i,0) = camera_calibration_data.D.at(i);

    ROS_DEBUG_STREAM("Image width: " << image_size->width);
    ROS_DEBUG_STREAM("Image height: " << image_size->height);
    ROS_DEBUG_STREAM("camera_intrinsic:" << std::endl << *camera_intrinsic);
    ROS_DEBUG_STREAM("camera_intrinsic:" << std::endl << *proj_matrix);
    ROS_DEBUG_STREAM("Distortion: " << *dist_matrix);

    //Simple check if calibration data meets expected values
    if ((camera_intrinsic->at<double>(2,2) == 1) && (dist_matrix->at<double>(0,4) == 0)){
        ROS_INFO_STREAM("Calibration data loaded successfully");
        return true;
    }
    else{
        ROS_WARN("Wrong calibration data, check calibration file and filepath");
        return false;
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

    // cv::undistort(cv_ptr->image, outputImage, *proj_matrix, *dist_matrix);
    outputImage = cv_ptr->image;

    std::vector<cv::Point3f> lid_pts;
    cv::Mat x(4,1, CV_64FC1);
    cv::Mat y(3,1, CV_64FC1);
    x.at<double>(3,0) = 1;

    short x_,y_=0;

    for( size_t i=0; i<cloudSize; i++){
        x.at<double>(0,0) = laserCloudIn->points[i].x;
        x.at<double>(1,0) = laserCloudIn->points[i].y;
        x.at<double>(2,0) = laserCloudIn->points[i].z;

        // consider only the lidar points in front
        if(x.at<double>(0,0) < 0 ) continue;

        y = *proj_matrix * R_rect_mat * T_mat * x;
        x_ = short( y.at<double>(0,0)/y.at<double>(0,2) );
        y_ = short( y.at<double>(0,1)/y.at<double>(0,2) );

        if(y_ < outputImage.rows && x_ < outputImage.cols && x_ > 0 && y_ > 0){
            cv::circle(outputImage, cv::Point(x_,y_), 1, cv::Scalar(0,255,0), 1, 8, 0);
        }
    }

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
    image_sub = it.subscribe("image_raw", 1,imageCb);
    lc_image_pub = it.advertise("lc_image",1);

    std::string calibration_filename;
    n.getParam("/project_lidar_on_image_node/calibration_file", calibration_filename);
    parse_calibration_file(calibration_filename);

    cout << "P mat" << endl << *proj_matrix << endl;
    cout << "Dist mat" << endl << *dist_matrix << endl;

    std::string trans_filename;
    n.getParam("/project_lidar_on_image_node/transformation_matrix", trans_filename);

    rot_mat = new(cv::Mat)(3,3, CV_64FC1);
    t_mat = new(cv::Mat)(3,1, CV_64FC1);
    T_mat = cv::Mat(4,4,CV_64FC1,cv::Scalar(0));

    R_rect_mat = cv::Mat(4,4,CV_64FC1,R_rect);
    cout << "R_rect mat" << endl << R_rect_mat << endl;

    ifstream f(trans_filename.c_str());

    for(int i=0;i<3;i++){
        for(int j=0;j<4;j++){
            if(j != 3){
                f >> rot_mat->at<double>(i,j);
                T_mat.at<double>(i,j) = rot_mat->at<double>(i,j);
            }
            else{
                f >> t_mat->at<double>(i,0);
                T_mat.at<double>(i,j) = t_mat->at<double>(i,0);
            }
        }
    }

    T_mat.at<double>(3,3) = 1;

    cout << "Rot_mat" << endl << *rot_mat << endl;
    cout << "translation_mat" << endl << *t_mat << endl;
    cout << "Transformation_mat" << endl << T_mat << endl;

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