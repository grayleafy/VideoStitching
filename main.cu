#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<opencv2/cudaimgproc.hpp>
#include<opencv2/cudafeatures2d.hpp>
#include<opencv2/cudawarping.hpp>
#include<opencv2/cudacodec.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/xfeatures2d/cuda.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/calib3d/calib3d.hpp>
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include<opencv2/stitching.hpp>
#include <cstdio>
#include<algorithm>
#include<iostream>
#include<cmath>
#include<ctime>
#include<thread>
#include<mutex>
#include<condition_variable>
#include<windows.h>

#include"ransac.h"
#include"ThreadPool.h"
#include"videostitch.h"
#include"utils.h"
using namespace cv;
using namespace std;




//鱼眼摄像头畸变矫正器
class Corrector {
	Mat mapx;
	Mat mapy;
	Mat R;
	Size image_size;
	Mat intrinsic_matrix;
	Mat distortion_coeffs;
	cuda::GpuMat mapx_gpu, mapy_gpu;
public:
	//输入标定数据文件目录，初始化矫正mapx和mapy
	Corrector(string filename) {
		FileStorage fs;
		if (!fs.open(filename, FileStorage::READ)) {
			cout << "打开标定数据失败\n";
			return;
		}
		fs["intrinsic_matrix"] >> intrinsic_matrix;
		fs["distortion_coeffs"] >> distortion_coeffs;
		fs["image_size"] >> image_size;
		fs.release();

		mapx = Mat(image_size, CV_32FC1);
		mapy = Mat(image_size, CV_32FC1);
		R = Mat::eye(3, 3, CV_32F);

		fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);
		mapx_gpu.upload(mapx);
		mapy_gpu.upload(mapy);
		cout << "初始化mapx, mapy完成\n";
	}

	//矫正cpu图像
	void FishRemap(Mat &inputArray, Mat &outputArray) {
		cuda::GpuMat src_gpu, img_gpu;
		src_gpu.upload(inputArray);
		cuda::remap(src_gpu, img_gpu, mapx_gpu, mapy_gpu, INTER_LINEAR);
		img_gpu.download(outputArray);
	}

	//矫正gpu图像
	void FishRemap(cuda::GpuMat &inputArray, cuda::GpuMat &outputArray) {
		cuda::remap(inputArray, outputArray, mapx_gpu, mapy_gpu, INTER_LINEAR);
	}
};



//读取视频流线程
void read_thread(Ptr<cudacodec::VideoReader> &cap, cuda::GpuMat &d_frame, mutex &mtx) {
	int flag = 1;
	//int last_t = clock();
	while (flag) {
		unique_lock<mutex> m(mtx);
		//int t1 = clock();
		flag = cap->nextFrame(d_frame);
		//int t2 = clock();
		//cout << "haoshi:" << t2 - t1 << endl;
		m.unlock();
		Sleep(30);
		//int t = clock();
		//cout << "读取每帧：" << t - last_t << endl;
		//last_t = t;
	}
}

//rtsp视频流
class Video {
	
	string url;
	Ptr<cudacodec::VideoReader> cap;
	cuda::GpuMat d_frame;
	Mat cpu_frame;

public:
	mutex mtx;
	Video(string url) :url(url) {
		cap = cudacodec::createVideoReader(url);
	}

	//在子线程中不断读取实时视频
	void read_run() {
		thread t(read_thread, ref(cap), ref(d_frame), ref(mtx));
		t.detach();
	}

	cuda::GpuMat getGpuMat() {
		return d_frame;
	}

	void show() {
		while (1) {
			unique_lock<mutex> m(mtx);
			d_frame.download(cpu_frame);
			m.unlock();
			imshow("img", cpu_frame);
			waitKey(1);
		}
	}

	//获取最新的gpu图像
	void read_gpu(cuda::GpuMat &outputArray) {
		unique_lock<mutex> m(mtx);
		outputArray = d_frame;
		m.unlock();
		return;
	}

	//返回互斥锁
	mutex &getMutex() {
		return mtx;
	}
};


mutex mtx;
condition_variable condtion_var;
bool ready_show = 0;
Mat img_show;
int thread_last_time;

//图片拼接流水线线程
void stitch_thread(cuda::GpuMat left_gpu, cuda::GpuMat right_gpu, int img_time, VideoStitcher &stitcher) {
	Mat res;
	if (stitcher.stitch(left_gpu, right_gpu, res) != 0) {
		cout << "拼接失败\n";
		return;
	}
	int now_time = clock();
	if (now_time < thread_last_time)	return;
	unique_lock<mutex> lck(mtx);
	thread_last_time = img_time;
	res.copyTo(img_show);
	ready_show = 1;	
	condtion_var.notify_one();
	lck.unlock();
	return;
}

//显示结果图片线程
void show_thread() {
	int sum = 0;
	int cnt = 0;
	int last_t = clock();
	while (1) {
		unique_lock<mutex> lck(mtx);
		while (ready_show == 0)	condtion_var.wait(lck);
		ready_show = 0;
		//resize(img_show, img_show, Size(0, 0), 0.5, 0.5, INTER_LINEAR);
		imshow("res", img_show);
		waitKey(1);
		int t = clock();
		sum += t - last_t;
		cnt++;
		cout << "流水线******最终单帧耗时:" << t - last_t << " ,平均每帧耗时:" << sum / cnt << endl;
		last_t = t;
		
		//Sleep(10);
	}
	
}



int main()
{
	string url_left = "rtsp://192.168.31.12:554/user=admin&password=&channel=1&stream=0.sdp?real_stream";
	string url_right = "rtsp://192.168.31.24:554/user=admin&password=&channel=1&stream=0.sdp?real_stream";
	string filename = "../../鱼眼摄像头标定/鱼眼摄像头标定/intrinsics.xml";
	Corrector corrector(filename);
	int opt = 3;

	
	//靖世九柱，流水线
	if (opt == 2) {
		int last_t = clock();
		thread_last_time = clock();
		

		Mat left = imread("left.jpg");
		Mat right = imread("right.jpg");
		resize(left, left, Size(0, 0), 0.5, 0.5, INTER_LINEAR);
		resize(right, right, Size(0, 0), 0.5, 0.5, INTER_LINEAR);
		cuda::GpuMat left_gpu, right_gpu;
		left_gpu.upload(left);
		right_gpu.upload(right);

		thread showThread(show_thread);
		ThreadPool pool(32);
		while (1) {
			Sleep(10);
			int now = clock();
//			pool.enqueue(stitch_thread, right_gpu, left_gpu, now);
			
	

			int t = clock();
		//	cout << "\n\n每帧耗时:" << t - last_t << endl << endl << endl;
			last_t = t;
		}
		
		showThread.join();
	}
	//读入实验室视频，流水线
	else if (opt == 3) {
		Ptr<cudacodec::VideoReader> cap_left = cudacodec::createVideoReader(string("day left.mp4"));
		Ptr<cudacodec::VideoReader> cap_right = cudacodec::createVideoReader(string("day right.mp4"));
		cuda::GpuMat left_gpu, right_gpu;

		if (!cap_left->nextFrame(left_gpu))		return -1;
		if (!cap_right->nextFrame(right_gpu))	return -1;
		corrector.FishRemap(left_gpu, left_gpu);
		corrector.FishRemap(right_gpu, right_gpu);
		VideoStitcher stitcher;
		stitcher.init(left_gpu, right_gpu);

		//显示结果线程
		thread showThread(show_thread);
		ThreadPool pool(32);

		int last_t = clock();
		int tt = 1;
		while (1) {
			if (!cap_left->nextFrame(left_gpu))		break;
			if (!cap_right->nextFrame(right_gpu))	break;
			
			//cuda::cvtColor(src_left_gpu, src_left_gpu, COLOR_BGRA2BGR);
			//cuda::cvtColor(src_right_gpu, src_right_gpu, COLOR_BGRA2BGR);
			//cout << "width:" << src_left_gpu.size() << endl;

			corrector.FishRemap(left_gpu, left_gpu);
			corrector.FishRemap(right_gpu, right_gpu);

			//cuda::resize(src_left_gpu, src_left_gpu, Size(0, 0), 0.5, 0.5);
			//cuda::resize(src_right_gpu, src_right_gpu, Size(0, 0), 0.5, 0.5);

			//Mat src_left, src_right;
			//src_left_gpu.download(src_left);
			//src_right_gpu.download(src_right);
			//imshow("left", src_left);
			//imshow("right", src_right);
			//imwrite("day left.jpg", src_left);
			//imwrite("day right.jpg", src_right);
			//waitKey(1);
			int now = clock();
			pool.enqueue(stitch_thread, left_gpu, right_gpu, now, stitcher);
			Sleep(10);
		}

		showThread.join();
	}
	//固定摄像头，读入实验室视频
	else if (opt == 4){
		VideoStitcher video_stitcher;
		Ptr<cudacodec::VideoReader> cap_left = cudacodec::createVideoReader(string("day left.mp4"));
		Ptr<cudacodec::VideoReader> cap_right = cudacodec::createVideoReader(string("day right.mp4"));
		cuda::GpuMat left_gpu, right_gpu;

		if (!cap_left->nextFrame(left_gpu))		return -1;
		if (!cap_right->nextFrame(right_gpu))	return -1;	
		corrector.FishRemap(left_gpu, left_gpu);
		corrector.FishRemap(right_gpu, right_gpu);
		//cuda::resize(left_gpu, left_gpu, Size(0, 0), 0.6, 0.6);
		//cuda::resize(right_gpu, right_gpu, Size(0, 0), 0.6, 0.6);
		cout << "left size:" << left_gpu.size() << endl;
		cout << "right size:" << right_gpu.size() << endl;
		if (video_stitcher.init(left_gpu, right_gpu) != 0)	return -1;
		cout << "初始化成功\n";

		int last_t = clock();
		while (1) {
			if (!cap_left->nextFrame(left_gpu))		break;
			if (!cap_right->nextFrame(right_gpu))	break;

			corrector.FishRemap(left_gpu, left_gpu);
			corrector.FishRemap(right_gpu, right_gpu);

			//cuda::resize(left_gpu, left_gpu, Size(0, 0), 0.6, 0.6);
			//cuda::resize(right_gpu, right_gpu, Size(0, 0), 0.6, 0.6);

			Mat result;
			if (video_stitcher.stitch(left_gpu, right_gpu, result) != 0) {
				cout << "拼接失败\n";
				continue;
			}

			//resize(result, result, Size(0, 0), 0.3, 0.3);
			imshow("res", result);
			waitKey(1);
			int t = clock();
			cout << "每帧耗时:" << t - last_t << endl;
			last_t = t;
			//waitKey(0);
		}
	}
	//靖世九柱，固定摄像头
	else if (opt == 5){
		Mat left = imread("left.jpg");
		Mat right = imread("right.jpg");
		resize(left, left, Size(0, 0), 0.5, 0.5, INTER_LINEAR);
		resize(right, right, Size(0, 0), 0.5, 0.5, INTER_LINEAR);
		cout << "left size:" << left.size() << endl;
		cout << "right size:" << right.size() << endl;
		cuda::GpuMat left_gpu, right_gpu;
		left_gpu.upload(left);
		right_gpu.upload(right);
		cuda::cvtColor(left_gpu, left_gpu, COLOR_BGR2BGRA);
		cuda::cvtColor(right_gpu, right_gpu, COLOR_BGR2BGRA);

		VideoStitcher video_stitcher;
		if (video_stitcher.init(left_gpu, right_gpu) != 0)	return -1;
		cout << "初始化成功\n";

		int last_t = clock();
		while (1) {
			Mat result;
			if (video_stitcher.stitch(left_gpu, right_gpu, result) != 0) {
				cout << "拼接失败\n";
				continue;
			}

			//resize(result, result, Size(0, 0), 0.6, 0.6);
			imshow("res", result);
			waitKey(1);
			int t = clock();
			cout << "每帧耗时:" << t - last_t << endl;
			last_t = t;
		}
	}
	

}

// Helper function for using CUDA to add vectors in parallel.

