cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(kinect_slam)

set(CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS} -std=c++11 -pthread") 
set(CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS} -march=native")

option(WITH_PCL "adds pcl cloud support" ON)
if(${WITH_PCL})
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_PCL")
	find_package(PCL REQUIRED)
	link_directories(${PCL_LIBRARY_DIRS})
	add_definitions(${PCL_DEFINITIONS})
	include_directories(${PCL_INCLUDE_DIRS})
endif()

find_package(Threads    REQUIRED)
find_package(OpenCV     REQUIRED)
find_package(Freenect2  REQUIRED)
find_package(GLUT       REQUIRED)

include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${FREENECT2_INCLUDE_DIRS})
include_directories(${INCLUDE})
include_directories(C:/builds/vcpkg/packages/libfreenect2_x64-windows/include/libfreenect2)

set(SRCS 
    main.cpp
	kinect_slam_freenect.cpp 
	kinect_slam_tracking.cpp 
	kinect_slam_gl.cpp
)

add_executable (kinect_slam ${SRCS})
target_link_libraries (kinect_slam 
	${freenect2_LIBRARIES}
	${CMAKE_THREAD_LIBS_INIT} 
	${OPENGL_LIBRARIES} 
	${GLUT_LIBRARIES} 
	${LIBUSB_1_LIBRARIES} 
	${OpenCV_LIBS} 
	${PCL_LIBRARIES}
	)