#include"kinect_slam_freenect.h"
#include"kinect_slam_tracking.h"
#include"kinect_slam_gl.h"

using namespace kinect_slam;
int main(int argc, char** argv) {
	CFreenectModule module1;
	CTrackingModule module2;
	module2.m_freenect_data = &module1.m_buffers;

	kinect_slam::CUIModule::inst.shared = &module2.shared;
	kinect_slam::CUIModule::inst.freenect = &module1;
	kinect_slam::CUIModule::inst.tracking = &module2;

	kinect_slam::CUIModule::inst.run(argc, argv);
	return 0;
}