#pragma once

#include <iostream>
#include <list>
#include <vector>
#include <mutex>
#include <thread>

#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "kinect_slam_freenect.h"

namespace kinect_slam {

	//////////////////////////////////////////////////////////////////
	// CFeatureTrack : data structure that contains a reconstructed
	//   point cloud and the camera pose for a registered frame.
	//////////////////////////////////////////////////////////////////
	struct CFeatureTrack {
		cv::Point2f base_position;
		cv::Point2f active_position;
		cv::Mat1f descriptor;
		int missed_frames;
	};

	//////////////////////////////////////////////////////////////////
	// CTrackedView: data structure that contains a reconstructed
	//   point cloud and the camera pose for a registered frame.
	//////////////////////////////////////////////////////////////////

	struct CTrackedView {
		pcl::PointCloud<pcl::PointXYZRGB> cloud;
		cv::Matx33f R;
		cv::Matx31f T;
	};

	//////////////////////////////////////////////////////////////////
	// CTrackingSharedData: data structure that contains shared
	//   variables between the tracking module and other modules.
	//////////////////////////////////////////////////////////////////
	struct CTrackingSharedData {
		std::mutex m_mutex;
		//Commands
		bool is_data_new;   //True if this class has new data that should be rendered
		bool is_tracking_enabled = true; //True if the tracking thread should process images

		cv::Matx33f base_R;
		cv::Matx31f base_T;
		cv::Mat3b   base_rgb;//Base image
		cv::Mat3f   base_pointmap;

		cv::Mat3b  active_rgb;//Last tracked image
		cv::Mat1s  active_depth;

		//Model
		std::list<CFeatureTrack> tracks; //Tracked features since last base frame
		std::vector<CTrackedView> views; //All registered views

		CTrackingSharedData();
		~CTrackingSharedData() = default;
	};

	//////////////////////////////////////////////////////////////////
	// CTrackingModule: Executes the SLAM algorithm. Extracts 2D 
	//   features, builds point clouds, and registers the point clouds.
	//////////////////////////////////////////////////////////////////
	class CTrackingModule
	{
	public:
		CTrackingSharedData shared;
		CFreenectSharedData* m_freenect_data;
		
		CTrackingModule()  = default;
		~CTrackingModule() = default;
		void stop() { running = false; m_thread.join(); }
		void run();
		int  thread_entry();

	private:
		bool running = true;
		std::thread m_thread;
		cv::Mat rgb_buffer;
		cv::Mat depth_buffer;

		void get_cloud_and_pointmap(const cv::Mat4b &rgb, const cv::Mat1f &depth,
			cv::Mat3f &pointmap,
			pcl::PointCloud<pcl::PointXYZRGB> &cloud);

		void match_features(const cv::Mat1f &new_descriptors,
			std::vector<int> &match_idx);

		static bool is_track_stale(const CFeatureTrack &track);

		void update_tracks(const std::vector<cv::KeyPoint> &feature_points,
			const cv::Mat1f &feature_descriptors,
			const std::vector<int> &match_idx);

		float get_median_feature_movement();

		void absolute_orientation(cv::Mat1f   &X, cv::Mat1f   &Y,
			cv::Matx33f &R, cv::Matx31f &T);

		void ransac_orientation(const cv::Mat1f &X, const cv::Mat1f &Y,
			cv::Matx33f &R, cv::Matx31f &T);

		void transformation_from_tracks(const cv::Mat3f &active_pointmap,
			cv::Matx33f &R, cv::Matx31f &T);
	};
}
