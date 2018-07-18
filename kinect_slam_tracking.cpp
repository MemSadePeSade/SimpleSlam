#include <cstdlib>
#include <iostream>
#include <algorithm>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

#include "kinect_slam_tracking.h"

namespace kinect_slam {

	template <class T1, class T2>
	T1 round(T2 v) { return static_cast<T1>(v + static_cast<T2>(0.5)); }

	CTrackingSharedData::CTrackingSharedData() :base_R(1, 0, 0, 0, 1, 0, 0, 0, 1),
		base_T(0.0, 0.0, 0.0) {
		base_pointmap = cv::Mat3f(424, 512, cv::Vec3f(0, 0, 0));
		base_rgb = cv::Mat3b(424, 512);
		active_rgb = cv::Mat3b(424, 512);
		active_depth = cv::Mat1s(424, 512);
	}
	
	void CTrackingModule::run() {
		m_thread = std::thread(&CTrackingModule::thread_entry, this);
	}

	void CTrackingModule::get_cloud_and_pointmap(const cv::Mat4b &rgb, const cv::Mat1f &depth,
		cv::Mat3f &pointmap,
		pcl::PointCloud<pcl::PointXYZRGB> &cloud) {
		libfreenect2::Frame rgb_tmp(rgb.cols, rgb.rows, 4, rgb.data);
		libfreenect2::Frame depth_tmp(depth.cols, depth.rows, 4, depth.data);

		cloud.resize(depth.rows*depth.cols);
		auto point = cloud.begin();
		float x, y, z, rgb_val;

		for (size_t i = 0; i < depth.rows; ++i)
		{
			for (size_t j = 0; j < depth.cols; j++)
			{
				m_freenect_data->m_registration->getPointXYZRGB(&depth_tmp,
					&rgb_tmp,
					i, j,
					x, y, z, rgb_val);
				point->x = x;
				point->y = y;
				point->z = z;
				pointmap(i, j) = cv::Vec3f(x, y, z);
				const uint8_t *p = reinterpret_cast<uint8_t*>(&rgb_val);
				point->r = p[0];
				point->g = p[1];
				point->b = p[2];
				++point;
			}
		}
	}

	void CTrackingModule::match_features(const cv::Mat1f &new_descriptors,
		std::vector<int> &match_idx) {
		cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM_L1, true));
		std::vector<cv::Mat> train_vector;
		std::vector<std::vector<cv::DMatch>> matches;
		std::vector<cv::DMatch> matchess;

		match_idx.resize(shared.tracks.size());

		std::list<CFeatureTrack>::iterator track_it;
		int i;
		for (i = 0, track_it = shared.tracks.begin();
			track_it != shared.tracks.end();
			i++, track_it++) {
			matcher->match(track_it->descriptor, new_descriptors, matchess);
			float best_dist = matchess[0].distance;
			if (best_dist < 0.8)
				match_idx[i] = matchess[0].trainIdx;
			else
				match_idx[i] = -1;
		}
	}

	bool CTrackingModule::is_track_stale(const CFeatureTrack &track) {
		return track.missed_frames > 20;
	}

	void CTrackingModule::update_tracks(const std::vector<cv::KeyPoint> &feature_points,
		const cv::Mat1f &feature_descriptors,
		const std::vector<int> &match_idx) {
		int i; std::list<CFeatureTrack>::iterator track_it;
		for (i = 0, track_it = shared.tracks.begin();
			track_it != shared.tracks.end();
			i++, track_it++)
		{
			int j = match_idx[i];
			if (j == -1)
				track_it->missed_frames++;
			else
			{
				track_it->missed_frames = 0;
				track_it->active_position = feature_points[j].pt;
				memcpy(track_it->descriptor.data, &feature_descriptors(j, 0),
					sizeof(float)*feature_descriptors.cols);
			}
		}
		//Delete tracks
		shared.tracks.remove_if(is_track_stale);
	}

	float CTrackingModule::get_median_feature_movement() {
		std::vector<float> vals;
		for (const auto& track : shared.tracks) {
			if (track.missed_frames == 0)
				vals.push_back(fabs(track.base_position.x - track.active_position.x)
					+
					fabs(track.base_position.y - track.active_position.y));
		}

		if (vals.empty())
			return 0;
		else
		{
			// Calculate median
			int n = vals.size() / 2;
			std::nth_element(vals.begin(), vals.begin() + n, vals.end());
			return vals[n];
		}
	}

	void CTrackingModule::absolute_orientation(cv::Mat1f   &X, cv::Mat1f   &Y,
		cv::Matx33f &R, cv::Matx31f &T) {
		cv::Matx31f meanX(0, 0, 0), meanY(0, 0, 0);
		int point_count = X.rows;

		//Calculate mean
		for (int i = 0; i < point_count; i++) {
			meanX(0) += X(i, 0);
			meanX(1) += X(i, 1);
			meanX(2) += X(i, 2);
			meanY(0) += Y(i, 0);
			meanY(1) += Y(i, 1);
			meanY(2) += Y(i, 2);
		}
		meanX *= 1.0f / point_count;
		meanY *= 1.0f / point_count;

		//Subtract mean
		for (int i = 0; i < point_count; i++) {
			X(i, 0) -= meanX(0);
			X(i, 1) -= meanX(1);
			X(i, 2) -= meanX(2);
			Y(i, 0) -= meanY(0);
			Y(i, 1) -= meanY(1);
			Y(i, 2) -= meanY(2);
		}

		//Rotation
		cv::Mat1f A;
		A = Y.t() * X;

		cv::SVD svd(A);

		cv::Mat1f Rmat;
		Rmat = svd.vt.t() * svd.u.t();
		Rmat.copyTo(R);

		//Translation
		T = meanX - R * meanY;
	}

	void CTrackingModule::ransac_orientation(const cv::Mat1f &X, const cv::Mat1f &Y,
		cv::Matx33f &R, cv::Matx31f &T) {
		const int max_iterations = 300;
		const int min_support = 6;
		const float inlier_error_threshold = 0.1f;

		const int pcount = X.rows;
		cv::RNG rng;
		cv::Mat1f Xk(min_support, 3), Yk(min_support, 3);
		cv::Matx33f Rk;
		cv::Matx31f Tk;
		std::vector<int> best_inliers;

		for (int k = 0; k < max_iterations; k++) {
			//Select random points
			for (int i = 0; i < min_support; i++) {
				int idx = rng(pcount);
				Xk(i, 0) = X(idx, 0);
				Xk(i, 1) = X(idx, 1);
				Xk(i, 2) = X(idx, 2);
				Yk(i, 0) = Y(idx, 0);
				Yk(i, 1) = Y(idx, 1);
				Yk(i, 2) = Y(idx, 2);
			}

			//Get orientation
			absolute_orientation(Xk, Yk, Rk, Tk);

			//Get error
			std::vector<int> inliers;
			for (int i = 0; i < pcount; i++) {
				float a, b, c, errori;
				cv::Matx31f py, pyy;
				py(0) = Y(i, 0);
				py(1) = Y(i, 1);
				py(2) = Y(i, 2);
				pyy = Rk * py + T;
				a = pyy(0) - X(i, 0);
				b = pyy(1) - X(i, 1);
				c = pyy(2) - X(i, 2);
				errori = sqrt(a*a + b * b + c * c);
				if (errori < inlier_error_threshold) {
					inliers.push_back(i);
				}
			}

			if (inliers.size() > best_inliers.size()) {
				best_inliers = inliers;
			}
		}
		std::cout << "Inlier count: " << best_inliers.size() << "/" << pcount << "\n";

		//Do final estimation with inliers
		Xk.resize(best_inliers.size());
		Yk.resize(best_inliers.size());
		for (unsigned int i = 0; i < best_inliers.size(); i++) {
			int idx = best_inliers[i];
			Xk(i, 0) = X(idx, 0);
			Xk(i, 1) = X(idx, 1);
			Xk(i, 2) = X(idx, 2);
			Yk(i, 0) = Y(idx, 0);
			Yk(i, 1) = Y(idx, 1);
			Yk(i, 2) = Y(idx, 2);
		}
		absolute_orientation(Xk, Yk, R, T);
	}

	void CTrackingModule::transformation_from_tracks(const cv::Mat3f &active_pointmap,
		cv::Matx33f &R, cv::Matx31f &T) {
		cv::Mat1f X(0, 3), Y(0, 3);
		X.reserve(shared.tracks.size());
		Y.reserve(shared.tracks.size());

		for (const auto& track : shared.tracks) {
			if (track.missed_frames != 0)
				continue;

			int ub = round<int>(track.base_position.x),
				vb = round<int>(track.base_position.y);
			cv::Vec3f &base_point = shared.base_pointmap(vb, ub);
			if (base_point(2) == 0 || std::isnan(base_point(2)))
				continue;

			int ua = round<int>(track.active_position.x),
				va = round<int>(track.active_position.y);
			const cv::Vec3f &active_point = active_pointmap(va, ua);
			if (active_point(2) == 0 || std::isnan(active_point(2)))
				continue;

			//Add to matrices
			int i = X.rows;
			X.resize(i + 1);
			X(i, 0) = base_point(0);
			X(i, 1) = base_point(1);
			X(i, 2) = base_point(2);

			Y.resize(i + 1);
			Y(i, 0) = active_point(0);
			Y(i, 1) = active_point(1);
			Y(i, 2) = active_point(2);
		}
		ransac_orientation(X, Y, R, T);
	}

	using namespace cv::xfeatures2d;
	int CTrackingModule::thread_entry() {
		int frame_count = 0;
		bool do_tracking = false;

		while (running) {
			std::unique_lock<std::mutex> lock(m_freenect_data->m_mutex);
			m_freenect_data->m_data_ready_cond.wait(lock,
				[&] {return m_freenect_data->got_rgbd > 0; });

			frame_count++;
			m_freenect_data->got_rgbd = 0;

			rgb_buffer = m_freenect_data->rgb_mid.clone();
			depth_buffer = m_freenect_data->depth_mid.clone();

			do_tracking = frame_count > 10 && shared.is_tracking_enabled;
			lock.unlock();

			if (!do_tracking) {
				//Update only images and skip
				std::unique_lock<std::mutex> lock2(shared.m_mutex);
				shared.active_depth = depth_buffer;
				cv::cvtColor(rgb_buffer, shared.active_rgb, CV_RGBA2RGB);
				shared.is_data_new = true;
				continue;
			}

			cv::Mat gray_img;
			//Convert the image to gray level
			cv::cvtColor(rgb_buffer, gray_img, CV_RGB2GRAY, 1);

			//Extract SURF features
			std::vector<cv::KeyPoint> feature_points; //Extracted feature points
			cv::Mat1f feature_descriptors_mat;  //Descriptor data returned by SURF
			int minHessian = 100;

			cv::Ptr<SURF> detector = SURF::create(100, 4, 1, false, false);
			detector->detect(gray_img, feature_points);
			cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = SURF::create(100, 4, 1, false, false);
			descriptorExtractor->compute(gray_img, feature_points, feature_descriptors_mat);

			//Match features
			std::vector<int> match_idx;
			match_features(feature_descriptors_mat, match_idx);

			//Update images & tracks
			std::unique_lock<std::mutex> lock2(shared.m_mutex);
			shared.active_depth = depth_buffer;
			cv::cvtColor(rgb_buffer, shared.active_rgb, CV_RGBA2RGB);
			update_tracks(feature_points, feature_descriptors_mat, match_idx);
			lock2.unlock();

			//Create new view
			bool add_view = false;
			CTrackedView new_view;
			cv::Mat3f pointmap = cv::Mat3f(424, 512, cv::Vec3f(0, 0, 0));
			float movement = get_median_feature_movement();
			std::cout << "Movement is " << movement << " do it!" << std::endl;
			if (shared.views.empty())
			{
				new_view.R = shared.base_R;
				new_view.T = shared.base_T;
				get_cloud_and_pointmap(rgb_buffer, depth_buffer,
					pointmap, new_view.cloud);
				add_view = true;
			}
			else
			{
				if (movement > 100) {
					get_cloud_and_pointmap(rgb_buffer, depth_buffer,
						pointmap, new_view.cloud);
					cv::Matx33f stepR;
					cv::Matx31f stepT;
					transformation_from_tracks(pointmap, stepR, stepT);
					new_view.R = shared.base_R * stepR;
					new_view.T = shared.base_T + shared.base_R*stepT;
					add_view = true;
				}
			}

			//Update shared data
			std::unique_lock<std::mutex> lock3(shared.m_mutex);
			if (add_view) {
				shared.views.push_back(new_view);
				shared.active_rgb.copyTo(shared.base_rgb);
				pointmap.copyTo(shared.base_pointmap);

				shared.base_R = new_view.R;
				shared.base_T = new_view.T;

				shared.tracks.clear();
				for (unsigned int i = 0; i < feature_points.size(); i++) {
					CFeatureTrack track;
					track.base_position = feature_points[i].pt;
					track.active_position = track.base_position;
					track.descriptor.create(1, feature_descriptors_mat.cols);
					memcpy(track.descriptor.data, &feature_descriptors_mat(i, 0),
						sizeof(float)*feature_descriptors_mat.cols);
					track.missed_frames = 0;
					shared.tracks.push_back(track);
				}
			}
			shared.is_data_new = true;
			lock3.unlock();
		}
		return 0;
	}
}