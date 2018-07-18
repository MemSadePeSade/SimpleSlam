#pragma once

#include<thread>
#include<condition_variable>
#include<mutex>

#include <libfreenect2.hpp>
#include <frame_listener_impl.h>
#include <packet_pipeline.h>
#include <registration.h>
#include <logger.h>
#include <opencv2/opencv.hpp>

namespace kinect_slam {
	//////////////////////////////////////////////////////////////////
	// CFreenectSharedData: data structure that contains shared
	//   variables between the freenect module and other modules.
	//////////////////////////////////////////////////////////////////
	struct CFreenectSharedData{
		libfreenect2::Registration* m_registration;
		std::mutex m_mutex;
		std::condition_variable  m_data_ready_cond;
		cv::Mat depth_mid; //Newest depth image
		cv::Mat rgb_mid;   //Newest color image
		int got_rgbd = 0; //Indicate the number of frames obtained (1 if no frames were dropped)
		
		CFreenectSharedData() = default;
		~CFreenectSharedData() { delete m_registration; }
	};

	//////////////////////////////////////////////////////////////////
	// CFreenectModule: Handles all communication with the Freenect 
	//   library. Acquires color and depth images from the Kinect and 
	//   offers them to the other modules.
	//////////////////////////////////////////////////////////////////
	class CFreenectModule {
	public:
		CFreenectSharedData m_buffers;
		
		CFreenectModule();
		~CFreenectModule() = default;
		void run();
		void stop();
	private:
		libfreenect2::PacketPipeline* m_pipeline = 0;
		libfreenect2::Freenect2 m_freenect2;
		libfreenect2::Freenect2Device* m_dev = 0;
		libfreenect2::SyncMultiFrameListener m_listener;
		libfreenect2::FrameMap m_frames;

		bool running;
		std::thread m_thread;
	
		int  thread_entry();
		void rgbd_callback(libfreenect2::Frame* rgb, libfreenect2::Frame* depth);
	};
}