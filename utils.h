#include<opencv2/cudaimgproc.hpp>
#include<opencv2/cudafeatures2d.hpp>
#include<opencv2/cudawarping.hpp>
#include<opencv2/cudacodec.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc.hpp>

#include <cstdio>
#include<algorithm>
#include<iostream>
#include<cmath>
#include<ctime>
using namespace std;
using namespace cv;
static const double eps = 1e-8;

#define gpu_rate 1590000

//�жϸ������Ƿ�Լ����0
bool isZero(double x) {
	return fabs(x) <= eps;
}

//�������任
Point2d mul(Point2d p, Mat H) {
	Mat res_mat;
	Mat temp = (Mat_<double>(3, 1) << p.x, p.y, 1);
	res_mat = H * temp;
	Point2d res;
	res.x = res_mat.at<double>(0, 0) / res_mat.at<double>(2, 0);
	res.y = res_mat.at<double>(1, 0) / res_mat.at<double>(2, 0);
	return res;
}

//�жϵ㾭���任��x���귽���Ƿ�ת
int judge_reverse(Point2d p, Mat H) {
	double x = p.x;
	double y = p.y;
	double temp = x * H.at<double>(2, 0) + y * H.at<double>(2, 1) + H.at<double>(2, 2);
	return temp <= 0;
}

//��ȡGPUƵ��
int get_GPU_Rate() {
	cudaDeviceProp deviceProp;//CUDA����Ĵ洢GPU���ԵĽṹ��
	cudaGetDeviceProperties(&deviceProp, 0);//CUDA���庯��
	return deviceProp.clockRate;
}

//��ӡgpu����������Ϣ
int getGPUThreadNum()
{
	cudaDeviceProp prop;
	int count;

	cudaGetDeviceCount(&count);
	printf("gpu num %d\n", count);
	cudaGetDeviceProperties(&prop, 0);
	printf("max thread num: %d\n", prop.maxThreadsPerBlock);
	printf("max grid dimensions: %d, %d, %d)\n",
		prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	return prop.maxThreadsPerBlock;
}

#pragma once
