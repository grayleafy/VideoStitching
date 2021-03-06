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

#include"ransac.h"
#include"utils.h"
using namespace cv;
using namespace std;


//图像拼接
class MyStitcher {
	Point2f corner_left_up, corner_left_down, corner_right_up, corner_right_down;
public:


	//计算透视变换后的4个角点位置
	void CalCorner(Mat src, Mat H) {
		int width = src.size().width;
		int height = src.size().height;
		Mat temp, res;

		//左上
		temp = (Mat_<double>(3, 1) << 0, 0, 1);
		res = H * temp;
		corner_left_up.x = res.at<double>(0, 0) / res.at<double>(2, 0);
		corner_left_up.y = res.at<double>(1, 0) / res.at<double>(2, 0);

		//左下
		temp = (Mat_<double>(3, 1) << 0, height, 1);
		res = H * temp;
		corner_left_down.x = res.at<double>(0, 0) / res.at<double>(2, 0);
		corner_left_down.y = res.at<double>(1, 0) / res.at<double>(2, 0);

		//右上
		temp = (Mat_<double>(3, 1) << width, 0, 1);
		res = H * temp;
		corner_right_up.x = res.at<double>(0, 0) / res.at<double>(2, 0);
		corner_right_up.y = res.at<double>(1, 0) / res.at<double>(2, 0);

		//右下
		temp = (Mat_<double>(3, 1) << width, height, 1);
		res = H * temp;
		corner_right_down.x = res.at<double>(0, 0) / res.at<double>(2, 0);
		corner_right_down.y = res.at<double>(1, 0) / res.at<double>(2, 0);
		return;
	}

	//优化两图的连接处，使得拼接自然
	void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
	{
		int h = dst.size().height;
		int w = dst.size().width;
		Mat left = Mat::zeros(Size(w, h), CV_8UC3);
		Mat right = Mat::zeros(Size(w, h), CV_8UC3);
		img1.copyTo(left(Rect(0, 0, img1.size().width, img1.size().height)));
		trans.copyTo(right(Rect(0, 0, trans.size().width, trans.size().height)));

		int start = min(corner_left_up.x, corner_left_down.x);//开始位置，即重叠区域的左边界  
	//	start = max(start, 0);
		double processWidth = img1.size().width - start;//重叠区域的宽度  
		//int rows = dst.rows;
		//int cols = img1.size().width; //注意，是列数*通道数
		double alpha = 1;//img1中像素的权重  
		for (int i = 0; i < h; i++)
		{
			uchar* p = left.ptr<uchar>(i);  //获取第i行的首地址
			uchar* t = right.ptr<uchar>(i);
			uchar* d = dst.ptr<uchar>(i);
			for (int j = max(start, 0); j < w; j++)
			{
				///*
				//3通道
				//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
				if (t[j * 3] <= 0 && t[j * 3 + 1] <= 0 && t[j * 3 + 2] <= 0)
				{
					alpha = 1;
				}
				else if (p[j * 3] == 0 && p[j * 3 + 1] == 0 && p[j * 3 + 2] == 0) {
					alpha = 0;
				}
				else
				{
					//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
					alpha = (processWidth - (j - start)) / processWidth;
				}
				d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
				d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
				d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
				//*/

				/*
				//单通道
				//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
				if (t[j] == 0)
				{
					alpha = 1;
				}
				else if (p[j] == 0) {
					alpha = 0;
				}
				else
				{
					//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好
					alpha = (processWidth - (j - start)) / processWidth;
				}
				d[j] = p[j] * alpha + t[j] * (1 - alpha);
				*/
			}
		}

	}

	//拼接图片，src1在右
	int stitch(cuda::GpuMat &src1_gpu, cuda::GpuMat src2_gpu, Mat &dst, int method = 1) {
		//计时开始
		int start_time = clock();

		//转化为灰度图
		cuda::GpuMat src1_gray_gpu, src2_gray_gpu; //gpu灰度图
		try
		{
			cuda::cvtColor(src1_gpu, src1_gray_gpu, COLOR_BGR2GRAY);
			cuda::cvtColor(src2_gpu, src2_gray_gpu, COLOR_BGR2GRAY);
		}
		catch (const Exception err)
		{
			cout << err.msg << endl;
			return -1;
		}

		Mat src1, src2;
		src1_gpu.download(src1);
		src2_gpu.download(src2);
		int upload_init_time = clock();
		cout << "gpu初始化时间:" << upload_init_time - start_time << endl;

		int upload_time = clock();
		cout << "加载时间:" << upload_time - upload_init_time << endl;

		//提取特征点，计算特征向量
		int minHessian = 400;
		Ptr<cuda::DescriptorMatcher> matcher;
		Mat descriptors1_cpu, descriptors2_cpu;
		cuda::GpuMat descriptors1, descriptors2;
		cuda::GpuMat keypoint1_gpu, keypoint2_gpu;
		vector<KeyPoint> keypoint1, keypoint2;
		int compute_time;
		if (method == 0) {		/*ORB*/
			Ptr<cuda::ORB> gpu_orb = cuda::ORB::create(minHessian);
			gpu_orb->detectAndCompute(src1_gray_gpu, cuda::GpuMat(), keypoint1, descriptors1);
			gpu_orb->detectAndCompute(src2_gray_gpu, cuda::GpuMat(), keypoint2, descriptors2);
			compute_time = clock();
			cout << "提取特征点时间:" << compute_time - upload_time << endl;

			matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING); //orb
		}
		else if (method == 1) { /*SURF*/
			cuda::SURF_CUDA gpu_surf;
			try
			{
				gpu_surf(src1_gray_gpu, cuda::GpuMat(), keypoint1, descriptors1);
			}
			catch (const Exception err)
			{
				cout << err.msg << endl;
				return -1;
			}
			try
			{
				gpu_surf(src2_gray_gpu, cuda::GpuMat(), keypoint2, descriptors2);
			}
			catch (const Exception err)
			{
				cout << err.msg << endl;
				return -1;
			}

			compute_time = clock();
			cout << "surf提取特征点时间:" << compute_time - upload_time << endl;

			matcher = cuda::DescriptorMatcher::createBFMatcher(); //surf
		}
		else if (method == 2) { /*sift*/
			Ptr<SIFT> sift = SIFT::create(minHessian); //????
			sift->detectAndCompute(src1, Mat(), keypoint1, descriptors1_cpu);
			sift->detectAndCompute(src2, Mat(), keypoint2, descriptors2_cpu);
			descriptors1.upload(descriptors1_cpu);
			descriptors2.upload(descriptors2_cpu);

			compute_time = clock();
			cout << "sift提取特征点时间:" << compute_time - upload_time << endl;

			matcher = cuda::DescriptorMatcher::createBFMatcher(); //sift
		}

		//descriptors1.download(descriptors1_cpu);
		//imshow("descriptors1", descriptors1_cpu);
		//descriptors2.download(descriptors2_cpu);

		//特征点匹配	
		vector<DMatch> matches;
		vector<vector<DMatch> > matches2;
		try {
			matcher->knnMatch(descriptors1, descriptors2, matches2, 2);
			//matcher->match(descriptors1, descriptors2, matches);
		}
		catch (const Exception err) {
			cout << "err:" << err.err << endl;
			cout << "msg:" << err.msg << endl;
			return -1;
		}
		int match_time = clock();
		cout << "匹配耗时:" << match_time - compute_time << endl;


		/*
		//提取好的匹配
		double max_dist = 0, min_dist = inf;
		for (int i = 0; i < matches.size(); i++) {
			double dist = matches[i].distance;
			if (dist < min_dist)	min_dist = dist;
			if (dist > max_dist)	max_dist = dist;
		}
		vector<DMatch> good_matches;
		for (int i = 0; i < matches.size(); i++) {
			if (matches[i].distance < min_dist * 3)	good_matches.push_back(matches[i]);
		}
		*/

		//Low's 算法
		vector<DMatch> good_matches;
		for (int i = 0; i < matches2.size(); i++) {
			if (matches2[i][0].distance < 0.5 * matches2[i][1].distance) {
				good_matches.push_back(matches2[i][0]);
			}
		}

		//绘制匹配图
		Mat img_matches;
		drawMatches(src1, keypoint1, src2, keypoint2, good_matches, img_matches);
		//imwrite("match_img.jpg", img_matches);
		//resize(img_matches, img_matches, Size(0, 0), 0.4, 0.4, INTER_LINEAR);
		imshow("img_matches", img_matches);
		waitKey(0);


		//构建变换矩阵
		vector<Point2f> p1, p2;
		for (int i = 0; i < good_matches.size(); i++) {
			p1.push_back(keypoint1[good_matches[i].queryIdx].pt);
			p2.push_back(keypoint2[good_matches[i].trainIdx].pt);
		}
		int add_time = clock();
		cout << "筛选特征点时间:" << add_time - match_time << ",匹配点对数量：" << p1.size() << endl;
		if (p1.size() < 4) {
			cout << "特征点匹配数量不足\n";
			return -1;
		}
		Mat match_mask;
		Mat H;
		vector<int> inlier_id;
		H = findHomography_gpu(p1, p2, 3, inlier_id);
		//H = findHomography(p1, p2, RHO);
		//H = findHomography(p1, p2);
		cout << "H:\n" << H << endl;
		int findh_time = clock();
		cout << "构建单应性矩阵耗时:" << findh_time - add_time << endl;
		int flag = 0;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (i == 2 && j == 2)	break;
				if (!isZero(H.at<double>(i, j)))	flag = 1;
			}
		}
		if (flag == 0) {
			cout << "构建单应性矩阵失败\n";
			return -1;
		}


		//输出匹配点坐标
		for (int i = 0; i < good_matches.size(); i++) {
			Point2f a = keypoint1[good_matches[i].queryIdx].pt;
			Point2f b = keypoint2[good_matches[i].trainIdx].pt;
			cout << "a = " << a << ", b = " << b << endl;
			Point2d p = mul(a, H);
			cout << "p = " << p << endl;
		}
		Point2f right_up(src1.size().width, 0);
		Point2d p = mul(right_up, H);
		cout << "右上:" << right_up << ", p = " << p << endl;

		//透视变换
		CalCorner(src1, H);
		cout << "边界点:" << corner_right_down << endl;
		cout << "边界点:" << corner_right_up << endl;
		int width = max(corner_right_up.x, corner_right_down.x);
		width = max(width, 0);
		int height = max(corner_left_down.y, corner_right_down.y);
		height = max(height, 0);
		cuda::GpuMat src1_trans_gpu;
		Mat src1_trans;
		imshow("src1", src1);
		try {
			cuda::warpPerspective(src1_gpu, src1_trans_gpu, H, Size(2920, 720));
			//warpPerspective(src1, src1_trans, H, Size(width, height), INTER_LINEAR);
			//cuda::warpAffine
		}
		catch (const Exception err) {
			cout << err.msg << endl;
			return -1;
		}
		src1_trans_gpu.download(src1_trans);
		int warp_time = clock();
		cout << "透视变换耗时:" << warp_time - findh_time << endl;
		//resize(src1_trans, src1_trans, Size(0, 0), 0.5, 0.5);
		imshow("src1_trans", src1_trans);
		waitKey(0);

		//图像拼接
		int w = max(src2.size().width, src1_trans.size().width), h = max(src2.size().height, src1_trans.size().height);
		Mat img_res = Mat::zeros(Size(w, h), CV_8UC3);	//单通道，改一下
		//cout << "img_res:" << img_res.size << endl;
		//cout << "trans:" << src1_trans.size << endl;
		//cout << "trans_channals:" << src1_trans.channels() << endl;
		//Mat img_res(1000, 2080, CV_8UC1);
		src1_trans.copyTo(img_res(Rect(0, 0, src1_trans.size().width, src1_trans.size().height)));
		src2.copyTo(img_res(Rect(0, 0, src2.size().width, src2.size().height)));
		int unite_time = clock();
		cout << "合并耗时:" << unite_time - warp_time << endl;


		//图像融合
		try {
			OptimizeSeam(src2, src1_trans, img_res);
		}
		catch (const Exception err) {
			cout << "图像融合失败";
			return -1;
		}
		int optimize_time = clock();
		cout << "图像融合耗时:" << optimize_time - unite_time << endl;

		//释放显卡内存？

		//imshow("res", img_res);

		//waitKey(0);
		img_res.copyTo(dst);
		int end_time = clock();
		cout << "拼接总耗时：" << end_time - start_time << endl;
		return 0;
	}
};

/*
int run() {
	int opt = 1;
	if (opt == 1) {
		Video video_left(url_left);
		video_left.read_run();

		Video video_right(url_right);
		video_right.read_run();


		int last_t = clock();
		while (1) {
			cout << "\n\nstart:\n";
			int read_t1 = clock();
			cuda::GpuMat d_src_left, d_src_right;
			video_left.read_gpu(d_src_left);
			video_right.read_gpu(d_src_right);
			int read_t2 = clock();
			cout << "读取视频耗时:" << read_t2 - read_t1 << endl;
			cuda::cvtColor(d_src_left, d_src_left, COLOR_BGRA2BGR);
			cuda::cvtColor(d_src_right, d_src_right, COLOR_BGRA2BGR);

			int correct_t1 = clock();
			corrector.FishRemap(d_src_left, d_src_left);
			corrector.FishRemap(d_src_right, d_src_right);
			int corrrct_t2 = clock();
			cout << "矫正耗时:" << corrrct_t2 - correct_t1 << endl;

			int other_t1 = clock();
			Mat src_left, src_right;
			d_src_left.download(src_left);
			d_src_right.download(src_right);
			cuda::resize(d_src_left, d_src_left, Size(0, 0), 0.3, 0.3, INTER_LINEAR);
			cuda::resize(d_src_right, d_src_right, Size(0, 0), 0.3, 0.3, INTER_LINEAR);
			//resize(src_left, src_left, Size(0, 0), 0.3, 0.3, INTER_LINEAR);
			//resize(src_right, src_right, Size(0, 0), 0.3, 0.3, INTER_LINEAR);
			//imshow("left", src_left);
			//imshow("right", src_right);
			//waitKey(1);
			int other_t2 = clock();
			cout << "变换及下载图片耗时:" << other_t2 - other_t1 << endl;

			//cout << "src channals : " << src_left.channels() << endl;
			//cout << "src type:\n" << src_left.type() << endl;

			Mat res;
			if (stitcher.stitch(d_src_left, d_src_right, res, 2) != 0) {
				cout << "拼接失败\n";
				continue;
			}
			imshow("res", res);
			waitKey(1);

			int t = clock();
			cout << "每帧总耗时:" << t - last_t << endl;
			last_t = t;
		}
	}
}
*/

#pragma once
