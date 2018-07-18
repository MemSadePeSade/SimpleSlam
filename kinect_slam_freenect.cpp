#include<iostream>

#include "kinect_slam_freenect.h"

namespace kinect_slam {
	CFreenectModule::CFreenectModule() :
		m_listener(libfreenect2::Frame::Color |
			libfreenect2::Frame::Ir |
			libfreenect2::Frame::Depth)
	{
		if (m_freenect2.enumerateDevices() == 0)
			std::cout << "no kinect2 connected!" << std::endl;
		auto serial = m_freenect2.getDefaultDeviceSerialNumber();
		m_pipeline = new libfreenect2::CpuPacketPipeline();
		m_dev = m_freenect2.openDevice(serial, m_pipeline);
		m_dev->setColorFrameListener(&m_listener);
		m_dev->setIrAndDepthFrameListener(&m_listener);
	}

	void CFreenectModule::stop() {
		running = false;
		m_thread.join();
		m_dev->stop();
		m_dev->close();
	}

	void CFreenectModule::run() {
		m_dev->start();
		m_buffers.m_registration = new libfreenect2::Registration(m_dev->getIrCameraParams(),
			m_dev->getColorCameraParams());
		running = true;
		m_thread = std::thread(&CFreenectModule::thread_entry, this);
	}

	int CFreenectModule::thread_entry() {
		while (running) {
			if (!m_listener.waitForNewFrame(m_frames, 10 * 1000)) {
				std::cout << "timeout!" << std::endl;
				return -1;
			}
			libfreenect2::Frame *rgb = m_frames[libfreenect2::Frame::Color];
			libfreenect2::Frame *ir = m_frames[libfreenect2::Frame::Ir];
			libfreenect2::Frame *depth = m_frames[libfreenect2::Frame::Depth];
			libfreenect2::Frame undistorted(depth->width, depth->height, 4);
			libfreenect2::Frame registered(depth->width, depth->height, 4);

			std::unique_lock<std::mutex> lock(m_buffers.m_mutex);
			m_buffers.m_registration->apply(rgb, depth,
				&undistorted, &registered);
			rgbd_callback(&registered, &undistorted);
			lock.unlock();
			m_buffers.m_data_ready_cond.notify_one();
			m_listener.release(m_frames);
		}
		return 0;
	}

	void inline CFreenectModule::rgbd_callback(libfreenect2::Frame* rgb,
		libfreenect2::Frame* depth) {
		m_buffers.depth_mid = cv::Mat(
			depth->height, depth->width,
			CV_32FC1, depth->data).clone();
		m_buffers.rgb_mid = cv::Mat(
			rgb->height, rgb->width,
			CV_8UC4, rgb->data).clone();
		m_buffers.got_rgbd++;
	}
}