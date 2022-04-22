#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <thrust/extrema.h>	/*CUDA数学工具*/

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
#include <cstdio>
#include<algorithm>
#include<iostream>
#include<cmath>
#include<ctime>
#include"utils.h"
using namespace cv;
using namespace std;
#define inf 1000000000.0
//#define d_fabs(a)	a > 0 ? a : -a


//清除透视变换后会反转的多余部分
__global__ void ClearSurplus_8UC4(cuda::PtrStepSz<uchar4> inputArray, cuda::PtrStepSz<double> H) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= inputArray.cols || y >= inputArray.rows)	return;
	double z = H(2, 0) * (double)x + H(2, 1) * (double)y + H(2, 2);
	if (z < 0) {
		inputArray(y, x).x = 0;
		inputArray(y, x).y = 0;
		inputArray(y, x).z = 0;
		inputArray(y, x).w = 0;
	}
	return;
}

//CUDA曝光差异矫正
__global__ void GainCompensation_8UC4(cuda::PtrStepSz<uchar4> inputArray, double alpha_blue, double beta_blue, double alpha_green, double beta_green, double alpha_red, double beta_red) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= inputArray.cols || y >= inputArray.rows)	return;
	if (inputArray(y, x).w == 0)	return;
	//int threshold = 0;
	//if (inputArray(y, x).x < threshold && inputArray(y, x).y < threshold && inputArray(y, x).z < threshold)	return;
	//if (inputArray(y, x).x >= threshold)	
	inputArray(y, x).x = alpha_blue * (double)inputArray(y, x).x + beta_blue;
	//if (inputArray(y, x).y >= threshold)	
	inputArray(y, x).y = alpha_green * (double)inputArray(y, x).y + beta_green;
	//if (inputArray(y, x).z >= threshold)	
	inputArray(y, x).z = alpha_red * (double)inputArray(y, x).z + beta_red;
	return;
}

//根据亚像素坐标，获取亚像素灰度, 仅适用于CV_8UC4
__device__ double GetScale(cuda::PtrStepSz<uchar4> image, Point2f point, int channel) {
	int x_l = point.x;
	int x_r = x_l + 1;
	int y_u = point.y;
	int y_d = y_u + 1;
	double x = point.x; /*亚像素点x坐标*/
	double y = point.y;	/*亚像素点y坐标*/

	double s1, s2, s3, s4; /*分别为左上，右上，左下，右下的灰度值*/
	if (channel == 1) {
		s1 = image(y_u, x_l).x;
		s2 = image(y_u, x_r).x;
		s3 = image(y_d, x_l).x;
		s4 = image(y_d, x_r).x;
	}
	else if (channel == 2) {
		s1 = image(y_u, x_l).y;
		s2 = image(y_u, x_r).y;
		s3 = image(y_d, x_l).y;
		s4 = image(y_d, x_r).y;
	}
	else if (channel == 3) {
		s1 = image(y_u, x_l).z;
		s2 = image(y_u, x_r).z;
		s3 = image(y_d, x_l).z;
		s4 = image(y_d, x_r).z;
	}
	else if (channel == 4) {
		s1 = image(y_u, x_l).w;
		s2 = image(y_u, x_r).w;
		s3 = image(y_d, x_l).w;
		s4 = image(y_d, x_r).w;
	}

	double ans_l = s1 * ((double)y_d - y) + s3 * (y - (double)y_u);
	double ans_r = s2 * ((double)y_d - y) + s4 * (y - (double)y_u);
	double ans = ans_l * ((double)x_r - x) + ans_r * (x - (double)x_l);
	return ans;
}

//构造求解曝光补偿的方程
__global__ void ConstructA_8UC4(cuda::PtrStepSz<uchar4> left_gpu, Point2f *inliers_left_gpu, cuda::PtrStepSz<uchar4> right_gpu, Point2f *inliers_right_gpu, int inliers_siz, double *A_red_gpu, double *b_red_gpu, double *A_green_gpu, double *b_green_gpu, double *A_blue_gpu, double *b_blue_gpu) {
	for (int i = 0; i < inliers_siz; i++) {
		double scale_left, scale_right;

		//蓝色通道
		scale_left = GetScale(left_gpu, inliers_left_gpu[i], 1);
		scale_right = GetScale(right_gpu, inliers_right_gpu[i], 1);
		A_blue_gpu[i] = scale_right;
		A_blue_gpu[inliers_siz + i] = 1.0;
		b_blue_gpu[i] = scale_left;

		//绿色通道
		scale_left = GetScale(left_gpu, inliers_left_gpu[i], 2);
		scale_right = GetScale(right_gpu, inliers_right_gpu[i], 2);
		A_green_gpu[i] = scale_right;
		A_green_gpu[inliers_siz + i] = 1.0;
		b_green_gpu[i] = scale_left;

		//红色通道
		scale_left = GetScale(left_gpu, inliers_left_gpu[i], 3);
		scale_right = GetScale(right_gpu, inliers_right_gpu[i], 3);
		A_red_gpu[i] = scale_right;
		A_red_gpu[inliers_siz + i] = 1.0;
		b_red_gpu[i] = scale_left;
	}
}

//指数函数，设备函数
__device__ inline double d_pow(double x, int n) {
	double res = 1;
	while (n) {
		res *= x;
		n--;
	}
	return res;
}

//绝对值，设备函数
__device__ inline double d_fabs(double x) {
	if (x > 0)	return x;
	return -x;
}

//BGRA图像，求(x, y)点x方向梯度，设备函数，仅适用于CV_8UC4
__device__ double GetGrand_x_BGRA(cuda::PtrStepSz<uchar4> src, int x, int y) {
	double res = 0;
	if (x - 1 < 0 || x + 1 >= src.cols || y - 1 < 0 || y + 1 >= src.rows)	return res;
	
	//下
	res += 1.0 * src(y + 1, x + 1).x;
	res += 1.0 * src(y + 1, x + 1).y;
	res += 1.0 * src(y + 1, x + 1).z;
	res -= 1.0 * src(y + 1, x - 1).x;
	res -= 1.0 * src(y + 1, x - 1).y;
	res -= 1.0 * src(y + 1, x - 1).z;
	//中
	res += 2.0 * src(y, x + 1).x;
	res += 2.0 * src(y, x + 1).y;
	res += 2.0 * src(y, x + 1).z;
	res -= 2.0 * src(y, x - 1).x;
	res -= 2.0 * src(y, x - 1).y;
	res -= 2.0 * src(y, x - 1).z;
	//上
	res += 1.0 * src(y - 1, x + 1).x;
	res += 1.0 * src(y - 1, x + 1).y;
	res += 1.0 * src(y - 1, x + 1).z;
	res -= 1.0 * src(y - 1, x - 1).x;
	res -= 1.0 * src(y - 1, x - 1).y;
	res -= 1.0 * src(y - 1, x - 1).z;

	res /= 3.0;
	return res;
}

//BGRA图像，求(x, y)点y方向梯度，设备函数，仅适用于CV_8UC4
__device__ double GetGrand_y_BGRA(cuda::PtrStepSz<uchar4> src, int x, int y) {
	double res = 0;
	if (x - 1 < 0 || x + 1 >= src.cols || y - 1 < 0 || y + 1 >= src.rows)	return res;

	//左
	res += 1.0 * src(y + 1, x - 1).x;
	res += 1.0 * src(y + 1, x - 1).y;
	res += 1.0 * src(y + 1, x - 1).z;
	res -= 1.0 * src(y - 1, x - 1).x;
	res -= 1.0 * src(y - 1, x - 1).y;
	res -= 1.0 * src(y - 1, x - 1).z;
	//中
	res += 2.0 * src(y + 1, x).x;
	res += 2.0 * src(y + 1, x).y;
	res += 2.0 * src(y + 1, x).z;
	res -= 2.0 * src(y - 1, x).x;
	res -= 2.0 * src(y - 1, x).y;
	res -= 2.0 * src(y - 1, x).z;
	//右
	res += 1.0 * src(y + 1, x + 1).x;
	res += 1.0 * src(y + 1, x + 1).y;
	res += 1.0 * src(y + 1, x + 1).z;
	res -= 1.0 * src(y - 1, x + 1).x;
	res -= 1.0 * src(y - 1, x + 1).y;
	res -= 1.0 * src(y - 1, x + 1).z;

	res /= 3.0;
	return res;
}

//灰度图像，求(x, y)点x方向梯度，包含周围4个点的插值。设备函数
__device__ inline double GetGrand_x(cuda::PtrStepSz<uchar> src, int x, int y) {
	double res = 0;
	if (x - 1 < 0 || x + 1 >= src.cols || y - 1 < 0 || y + 1 >= src.rows)	return res;
	else {
		//下
		res += 2.0 * src(y + 1, x + 1);
		res -= 2.0 * src(y + 1, x - 1);
		//中
		res += 1.0 * src(y, x + 1);
		res -= 1.0 * src(y, x - 1);
		//上
		res += 2.0 * src(y - 1, x + 1);
		res -= 2.0 * src(y - 1, x - 1);

		return res;
	}	
}

//灰度图像，求(x, y)点y方向梯度，包含周围4个点的插值。设备函数
__device__ inline double GetGrand_y(cuda::PtrStepSz<uchar> src, int x, int y) {
	double res = 0;
	if (x - 1 < 0 || x + 1 >= src.cols || y - 1 < 0 || y + 1 >= src.rows)	return res;
	else {
		//左
		res += 2.0 * src(y + 1, x - 1);
		res -= 2.0 * src(y - 1, x - 1);
		//中
		res += 1.0 * src(y + 1, x);
		res -= 1.0 * src(y - 1, x);
		//右
		res += 2.0 * src(y + 1, x + 1);
		res -= 2.0 * src(y - 1, x + 1);

		return res;
	}
}

//邻域灰度值差平均值
__device__ inline double GetGrayDifference(const cuda::PtrStepSz<uchar> left, const cuda::PtrStepSz<uchar> right, const int x, const int y, int n = 1) {
	double res = 0;
	if (x - n < 0 || x + n >= left.cols || y - n < 0 || y + n >= left.rows)	return res;
	for (int i = -n; i <= n; i++) {
		for (int j = -n; j <= n; j++) {
			res += d_fabs((double)left(y + j, x + i) - right(y + j, x + i));
		}
	}
	res /= (2.0 * n + 1.0) * (2.0 * n + 1.0);
	return res;
}


//CUDA计算最佳缝合线能量函数，仅适用于CV_GRAY
__global__ void CalE(cuda::PtrStepSz<uchar> left, cuda::PtrStepSz<uchar> right, cuda::PtrStepSz<double> E) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= left.cols || y >= left.rows)	return;
	E(y, x) = 0;

	//无关部分
	if (left(y, x) == 0 && right(y, x) == 0) {
		E(y, x) = 0;
		return;
	}
	//边界点
	if (x - 1 < 0 || x + 1 >= left.cols || y - 1 < 0 || y + 1 >= left.rows) {
		E(y, x) = 2000.0;
		return;
	}

	//颜色差异
	double temp;
	temp = 0;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			double val = (double)left(y + j, x + i) - right(y + j, x + i);
			if (val > 0)	temp += val;
			else			temp -= val;
		}
	}
	temp /= 9.0;
	E(y, x) += temp;



	//梯度差异
	double res = 0;
	res = 0;
	//下
	res += 2.0 * left(y + 1, x + 1);
	res -= 2.0 * left(y + 1, x - 1);
	//中
	res += 1.0 * left(y, x + 1);
	res -= 1.0 * left(y, x - 1);
	//上
	res += 2.0 * left(y - 1, x + 1);
	res -= 2.0 * left(y - 1, x - 1);
	double grand_left_x = res; /*left,x方向*/

	res = 0;
	//左
	res += 2.0 * left(y + 1, x - 1);
	res -= 2.0 * left(y - 1, x - 1);
	//中
	res += 1.0 * left(y + 1, x);
	res -= 1.0 * left(y - 1, x);
	//右
	res += 2.0 * left(y + 1, x + 1);
	res -= 2.0 * left(y - 1, x + 1);
	double grand_left_y = res;

	res = 0;
	//下
	res += 2.0 * right(y + 1, x + 1);
	res -= 2.0 * right(y + 1, x - 1);
	//中
	res += 1.0 * right(y, x + 1);
	res -= 1.0 * right(y, x - 1);
	//上
	res += 2.0 * right(y - 1, x + 1);
	res -= 2.0 * right(y - 1, x - 1);
	double grand_right_x = res;

	res = 0;
	//左
	res += 2.0 * right(y + 1, x - 1);
	res -= 2.0 * right(y - 1, x - 1);
	//中
	res += 1.0 * right(y + 1, x);
	res -= 1.0 * right(y - 1, x);
	//右
	res += 2.0 * right(y + 1, x + 1);
	res -= 2.0 * right(y - 1, x + 1);
	double grand_right_y = res;

	E(y, x) += d_fabs((grand_left_x - grand_right_x) * (grand_left_y - grand_right_y));
	return;
}

//CUDA动态规划计算最佳缝合线，x坐标跨度小于等于1024，既一个block中最大线程数
__global__ void CalPath_DP(cuda::PtrStepSz<double1> E, cuda::PtrStepSz<double2> path, int from) {
	const int rows = E.rows;
	const int cols = E.cols;
	int x = blockDim.x * blockIdx.x + threadIdx.x + from;
	if (x >= cols)	return;
	path(rows - 1, x).y = E(rows - 1, x).x;
	__syncthreads();
	
	for (int y = rows - 2; y >= 0; y--) {
		path(y, x).y = path(y + 1, x).y;
		path(y, x).x = x;

		if (x - 1 >= 0 && path(y + 1, x - 1).y < path(y, x).y) {
			path(y, x).y = path(y + 1, x - 1).y;
			path(y, x).x = x - 1;
		}
		if (x + 1 < cols && path(y + 1, x + 1).y < path(y, x).y) {
			path(y, x).y = path(y + 1, x + 1).y;
			path(y, x).x = x + 1;
		}
		path(y, x).y = path(y,x).y + E(y, x).x;
		//if (x == 4735 || x == 4736) {
			//printf("nei   y = %d, x = %d, path = %lf\n", y, x, path(y, x).y);
		//}
		__syncthreads();
		//不会选择同一行
	}
}

//CUDA动态规划计算最佳缝合线，每次计算同一y坐标的一层，x坐标跨度大于1024，既一个block中最大线程数
__global__ void CalPath_DP_SingleRow(cuda::PtrStepSz<double1> E, cuda::PtrStepSz<double2> path, int from, int y) {
	const int rows = E.rows;
	const int cols = E.cols;
	int x = blockDim.x * blockIdx.x + threadIdx.x + from;
	if (x >= cols)	return;
	if (y == rows - 1) {
		path(y, x).y = E(y, x).x;
		return;
	}

	
	path(y, x).y = path(y + 1, x).y;
	path(y, x).x = x;

	if (x - 1 >= 0 && path(y + 1, x - 1).y < path(y, x).y) {
		path(y, x).y = path(y + 1, x - 1).y;
		path(y, x).x = x - 1;
	}
	if (x + 1 < cols && path(y + 1, x + 1).y < path(y, x).y) {
		path(y, x).y = path(y + 1, x + 1).y;
		path(y, x).x = x + 1;
	}
	path(y, x).y = path(y, x).y + E(y, x).x;
	//不会选择同一行
	return;
}

//根据路径回溯，计算缝合线x坐标
__global__ void CalSeam(cuda::PtrStepSz<int> seam, cuda::PtrStepSz<double2> path) {
	double res = inf;
	const int cols = path.cols;
	const int rows = path.rows;

	for (int x = 0; x < cols; x = x + 1) {
		if (path(0, x).y <= res) {
			res = path(0, x).y;
			seam(0, 0) = x;
		}
	}
	

	int x = seam(0, 0);
	for (int y = 0; y < rows - 1; y++) {
		x = seam(y + 1, 0) = path(y, x).x;
	}
}

//CUDA求解最佳缝合线，left和right的大小需相等，,返回0则成功。仅适用于CV_8UC4和CV_GRAY
int GetOptimalSeam(cuda::GpuMat left, cuda::GpuMat right, cuda::GpuMat &seam, int overlap_left, int overlap_right) {
	if (left.size() != right.size()) {
		cout << "图片大小不一致，求解最佳缝合线失败\n";
		return -1;
	}
	if (left.type() == CV_8UC4) cuda::cvtColor(left, left, COLOR_BGRA2GRAY);
	if (right.type() == CV_8UC4) cuda::cvtColor(right, right, COLOR_BGRA2GRAY);
	int rows = left.rows;
	int cols = left.cols;
	seam = cuda::GpuMat(rows, 1, CV_32SC1);
	cuda::GpuMat E(rows, cols, CV_64FC1);

	//计算能量函数
	dim3 block_siz(32, 32);
	dim3 block_num((cols + block_siz.x - 1) / block_siz.x, (rows + block_siz.y - 1) / block_siz.y);
	CalE << <block_num, block_siz >> > (left, right, E);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	//显示能量函数
	/*
	{
		Mat e(E.size(), CV_8UC1);
		Mat temp;
		E.download(temp);
		for (int y = 0; y < temp.rows; y++) {
			for (int x = 0; x < temp.cols; x++) {
				//if (y == 0) printf("x = %d, y = %d, e = %lf\n", x, y, temp.at<double>(y, x));
				e.at<uchar>(y, x) = 255 * temp.at<double>(y, x) / 2000.0;
			}
		}
		resize(e, e, Size(0, 0), 0.3, 0.3);
		imshow("E", e);
		waitKey(1);
	}
	*/
	
	//动态规划计算最佳缝合线路径，分两种情况
	cuda::GpuMat path(rows, cols, CV_64FC2, Scalar(inf, inf));
	{
		//int t1 = clock();
		if (overlap_right - overlap_left + 1 <= 1024) {
			CalPath_DP << <1, overlap_right - overlap_left + 1 >> > (E, path, overlap_left);
			CHECK(cudaGetLastError());
			CHECK(cudaDeviceSynchronize());
		}
		else {
			for (int y = rows - 1; y >= 0; y--) {
				CalPath_DP_SingleRow << <(overlap_right - overlap_left + 1 + 1024 - 1) / 1024, 1024 >> > (E, path, overlap_left, y);
				CHECK(cudaGetLastError());
				CHECK(cudaDeviceSynchronize());
			}
		}
		//int t2 = clock();
		//cout << "动态规划耗时:" << t2 - t1 << endl;
	}

	//回溯得到缝合线x坐标,gpu
	{
		//int t1 = clock();
		CalSeam << <1, 1 >> > (seam, path);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());
		//int t2 = clock();
		//cout << "回溯耗时:" << t2 - t1 << endl;
	}

	//path show
	/*
	{
		Mat e(path.size(), CV_8UC1);
		Mat temp;
		path.download(temp);
		for (int y = 0; y < temp.rows; y++) {
			for (int x = 0; x < temp.cols; x++) {
				if (y == 0) {
					printf("x = %d, y = %d,  path = %lf\n", x, y, temp.at<Vec2d>(y, x)[1]);
				}
				e.at<uchar>(y, x) = (double)255 * temp.at<Vec2d>(y, x)[1] / 20000.0;
			}
		}
		resize(e, e, Size(0, 0), 0.3, 0.3);
		imshow("path", e);
		waitKey(1);
	}
	*/

	return 0;
}

//CUDA计算最佳缝合线能量函数，仅适用于CV_GRAY，只计算重叠区域
__global__ void CalE(cuda::PtrStepSz<uchar> left, cuda::PtrStepSz<uchar> right, cuda::PtrStepSz<double> E, int from) {
	int x = blockIdx.x * blockDim.x + threadIdx.x + from;
	int ex = x - from;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= left.cols || y >= left.rows)	return;
	E(y, ex) = 0;

	//无关部分
	if (left(y, x) == 0 && right(y, x) == 0) {
		E(y, ex) = 0;
		return;
	}
	//边界点
	if (x - 1 < 0 || x + 1 >= left.cols || y - 1 < 0 || y + 1 >= left.rows) {
		E(y, ex) = 2000.0;
		return;
	}

	//颜色差异
	double temp;
	temp = 0;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			double val = (double)left(y + j, x + i) - right(y + j, x + i);
			if (val > 0)	temp += val;
			else			temp -= val;
		}
	}
	temp /= 9.0;
	E(y, ex) += temp;



	//梯度差异
	double res = 0;
	res = 0;
	//下
	res += 2.0 * left(y + 1, x + 1);
	res -= 2.0 * left(y + 1, x - 1);
	//中
	res += 1.0 * left(y, x + 1);
	res -= 1.0 * left(y, x - 1);
	//上
	res += 2.0 * left(y - 1, x + 1);
	res -= 2.0 * left(y - 1, x - 1);
	double grand_left_x = res; /*left,x方向*/

	res = 0;
	//左
	res += 2.0 * left(y + 1, x - 1);
	res -= 2.0 * left(y - 1, x - 1);
	//中
	res += 1.0 * left(y + 1, x);
	res -= 1.0 * left(y - 1, x);
	//右
	res += 2.0 * left(y + 1, x + 1);
	res -= 2.0 * left(y - 1, x + 1);
	double grand_left_y = res;

	res = 0;
	//下
	res += 2.0 * right(y + 1, x + 1);
	res -= 2.0 * right(y + 1, x - 1);
	//中
	res += 1.0 * right(y, x + 1);
	res -= 1.0 * right(y, x - 1);
	//上
	res += 2.0 * right(y - 1, x + 1);
	res -= 2.0 * right(y - 1, x - 1);
	double grand_right_x = res;

	res = 0;
	//左
	res += 2.0 * right(y + 1, x - 1);
	res -= 2.0 * right(y - 1, x - 1);
	//中
	res += 1.0 * right(y + 1, x);
	res -= 1.0 * right(y - 1, x);
	//右
	res += 2.0 * right(y + 1, x + 1);
	res -= 2.0 * right(y - 1, x + 1);
	double grand_right_y = res;

	E(y, ex) += d_fabs((grand_left_x - grand_right_x) * (grand_left_y - grand_right_y));
	return;
}

//CUDA和GPU求解最佳缝合线，left和right的大小需相等，,返回0则成功。仅适用于CV_8UC4和CV_GRAY
int GetOptimalSeam_CPU(cuda::GpuMat left, cuda::GpuMat right, cuda::GpuMat &seam, int overlap_left, int overlap_right) {
	if (left.size() != right.size()) {
		cout << "图片大小不一致，求解最佳缝合线失败\n";
		return -1;
	}
	if (left.type() == CV_8UC4) cuda::cvtColor(left, left, COLOR_BGRA2GRAY);
	if (right.type() == CV_8UC4) cuda::cvtColor(right, right, COLOR_BGRA2GRAY);
	int rows = left.rows;
	int cols = left.cols;
	int overlap_len = overlap_right - overlap_left + 1;
	
	

	//计算能量函数
	cuda::GpuMat E(rows, overlap_len, CV_64FC1);
	dim3 block_siz(32, 32);
	dim3 block_num((overlap_len + block_siz.x - 1) / block_siz.x, (rows + block_siz.y - 1) / block_siz.y);
	CalE << <block_num, block_siz >> > (left, right, E, overlap_left);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	//动态规划求最佳路径
	Mat E_cpu;
	E.download(E_cpu);
	Mat path(rows, overlap_len, CV_64FC2, Scalar(inf, inf));
	//最后一列初始化
	double *p_path = (double*)path.data + (rows - 1) * path.step1();
	double *p_E = (double*)E_cpu.data + (rows - 1) * E_cpu.step1();
	for (int x = 0; x < overlap_len; x++) {
		p_path[0] = p_E[0];
		p_path += 2;
		p_E++;
	}
	//向上更新
	for (int y = rows - 2; y >= 0; y--) {
		double *p_path = (double*)path.data + y * path.step1();
		double *p_path_last = (double*)path.data + (y + 1) * path.step1();
		double *p_E = (double*)E_cpu.data + y * E_cpu.step1();
		for (int x = 0; x < overlap_len; x++) {
			p_path[2 * x + 1] = x;
			p_path[2 * x] = p_path_last[2 * x];

			if (x - 1 >= 0 && p_path_last[2 * (x - 1)] < p_path[2 * x]) {
				p_path[2 * x] = p_path_last[2 * (x - 1)];
				p_path[2 * x + 1] = x - 1;
			}

			if (x + 1 < overlap_len && p_path_last[2 * (x + 1)] < p_path[2 * x]) {
				p_path[2 * x] = p_path_last[2 * (x + 1)];
				p_path[2 * x + 1] = x + 1;
			}

			p_path[2 * x] += p_E[x];
			p_path[2 * x + 1] += overlap_left;
		}
	}



	//回溯得到缝合线x坐标,gpu
	{
		Mat seam_cpu(rows, 1, CV_32S);
		//从第一行找最小
		seam_cpu.at<int>(0, 0) = 0;
		double temp = inf;
		double *p_path = (double*)path.data;
		for (int x = 0; x < overlap_len; x++) {
			if (p_path[2 * x] < temp) {
				temp = p_path[2 * x];
				seam_cpu.at<int>(0, 0) = x + overlap_left;
			}
		}
		//向下更新
		for (int y = 1; y < rows; y++) {
			seam_cpu.at<int>(y, 0) = path.at<Vec2d>(y - 1, seam_cpu.at<int>(y - 1, 0) - overlap_left)[1];
		}
		seam.upload(seam_cpu);
	}

	//path show
	/*
	{
		Mat e(path.size(), CV_8UC1);
		Mat temp;
		path.download(temp);
		for (int y = 0; y < temp.rows; y++) {
			for (int x = 0; x < temp.cols; x++) {
				if (y == 0) {
					printf("x = %d, y = %d,  path = %lf\n", x, y, temp.at<Vec2d>(y, x)[1]);
				}
				e.at<uchar>(y, x) = (double)255 * temp.at<Vec2d>(y, x)[1] / 20000.0;
			}
		}
		resize(e, e, Size(0, 0), 0.3, 0.3);
		imshow("path", e);
		waitKey(1);
	}
	*/

	return 0;
}

//绘制缝合线
__global__ void DrawSeam(cuda::PtrStepSz<uchar4> image, cuda::PtrStepSz<int> seam) {
	for (int y = 0; y < image.rows; y++) {
		int x = seam(y, 0);
		image(y, x).x = 0;
		image(y, x).y = 0;
		image(y, x).z = 255;
		x--;
		if (x >= 0) {
			image(y, x).x = 0;
			image(y, x).y = 0;
			image(y, x).z = 255;
		}
		x += 2;
		if (x < image.cols) {
			image(y, x).x = 0;
			image(y, x).y = 0;
			image(y, x).z = 255;
		}
	}
}

//最佳缝合线图像融合, 仅适用于CV_8UC4
__global__ void ImageBlend(cuda::PtrStepSz<uchar4> left, cuda::PtrStepSz<uchar4> right, cuda::PtrStepSz<int> seam, cuda::PtrStepSz<uchar4> result, int len) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= result.cols || y >= result.rows)	return;

	if (x < seam(y, 0) - len) {
		result(y, x).x = left(y, x).x;
		result(y, x).y = left(y, x).y;
		result(y, x).z = left(y, x).z;
		result(y, x).w = left(y, x).w;
	}
	else if (x > seam(y, 0) + len) {
		result(y, x).x = right(y, x).x;
		result(y, x).y = right(y, x).y;
		result(y, x).z = right(y, x).z;
		result(y, x).w = right(y, x).w;
	}
	else {
		double alpha = len + seam(y, 0) - x;
		alpha = alpha / (double)(2.0 * len);
		result(y, x).x = (1.0 - alpha) * right(y, x).x + alpha * left(y, x).x;
		result(y, x).y = (1.0 - alpha) * right(y, x).y + alpha * left(y, x).y;
		result(y, x).z = (1.0 - alpha) * right(y, x).z + alpha * left(y, x).z;
		result(y, x).w = (1.0 - alpha) * right(y, x).w + alpha * left(y, x).w;
	}
}

//视频拼接器
class VideoStitcher {
	Mat H;
	cuda::GpuMat H_gpu;
	Size left_siz, right_siz;
	Size result_siz;
	double alpha_blue, alpha_green, alpha_red;
	double beta_blue, beta_green, beta_red;
	int overlap_left, overlap_right; /*c重叠区域左右x坐标，左闭右闭*/
public:
	//初始化图像大小，单应性矩阵
	int init(cuda::GpuMat left_gpu, cuda::GpuMat right_gpu) {
		cout << "初始化中......\n";
		//如果图片为空则退出
		if (left_gpu.empty() || right_gpu.empty()) {
			cout << "图片为空，初始化失败\n";
			return -1;
		}

		//如果图片不是4通道则退出
		if (left_gpu.type() != CV_8UC4) {
			cout << "图片不是CV_8UC4类型，初始化失败\n";
			return -1;
		}

		//计算大小参数
		left_siz = left_gpu.size();
		right_siz = right_gpu.size();
		result_siz = Size(2.5 * left_siz.width, 1.5 * left_siz.height);

		//下载图片，低效
		Mat left, right;
		left_gpu.download(left);
		right_gpu.download(right);

		//高斯模糊，低效
		GaussianBlur(left, left, Size(3, 3), 3, 3);
		GaussianBlur(right, right, Size(3, 3), 3, 3);
		left_gpu.upload(left);
		right_gpu.upload(right);

		//SIFT提取特征点，计算特征向量
		int minHessian = 400;
		Mat descriptors1, descriptors2;
		vector<KeyPoint> keypoint1, keypoint2;
		Ptr<SIFT> sift = SIFT::create();   /*可选400*/
		sift->detectAndCompute(left, Mat(), keypoint1, descriptors1);
		sift->detectAndCompute(right, Mat(), keypoint2, descriptors2);

		//特征点匹配
		vector<vector<DMatch> > matches;
		cuda::GpuMat descriptors1_gpu, descriptors2_gpu;
		descriptors1_gpu.upload(descriptors1);
		descriptors2_gpu.upload(descriptors2);
		Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L1);
		matcher->knnMatch(descriptors1_gpu, descriptors2_gpu, matches, 2);

		//Low's 算法剔除错误匹配
		vector<DMatch> good_matches;
		for (int i = 0; i < matches.size(); i++) {
			if (matches[i][0].distance < 0.5 * matches[i][1].distance) {
				good_matches.push_back(matches[i][0]);
			}
		}

		//选取匹配点对建立方程
		vector<Point2f> p1, p2;
		for (int i = 0; i < good_matches.size(); i++) {
			p1.push_back(keypoint1[good_matches[i].queryIdx].pt);
			p2.push_back(keypoint2[good_matches[i].trainIdx].pt);
		}
		if (p1.size() < 4) {
			cout << "匹配点数量不足，初始化失败\n";
			return -1;
		}

		//调用findHomography求解单应性方程，并保存内点
		vector<int> inlier_id;
		Mat temp_H = findHomography_gpu(p2, p1, 3.0, inlier_id);
		int flag = 0;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (i == 2 && j == 2)	break;
				if (!isZero(temp_H.at<double>(i, j)))	flag = 1;
			}
		}
		if (flag == 0) {
			cout << "构建单应性矩阵失败\n";
			return -1;
		}

		//保存单应性矩阵结果
		temp_H.copyTo(H);
		H_gpu.upload(H);

		//计算重叠区域的左右x坐标，反转的情况？
		{
			overlap_right = left_gpu.cols - 1;
			overlap_left = left_gpu.cols - 1;
			Point2d temp1 = Point2d(0, 0);
			Point2d res = mul(temp1, H);	/*使用到了H*/
			overlap_left = min(overlap_left, (int)res.x);
			Point2d temp2 = Point2d(0, right_gpu.rows - 1);
			res = mul(temp2, H);	/*使用到了H*/
			overlap_left = min(overlap_left, (int)res.x);
			overlap_left = max(overlap_left, 0);
			if (judge_reverse(temp1, H) || judge_reverse(temp2, H)) {
				overlap_left = 0;
			}
			cout << "左边界:" << overlap_left << endl;
			cout << "右边界:" << overlap_right << endl;
		}
		

		//构造曝光补偿方程, CV_8UC4
		int inlier_siz = inlier_id.size();
		vector<double> A_red(2 * inlier_siz), A_green(2 * inlier_siz), A_blue(2 * inlier_siz);	/*方程A矩阵*/
		vector<double> b_blue(inlier_siz), b_green(inlier_siz), b_red(inlier_siz);	/*方程b向量*/
		double *A_red_gpu, *A_green_gpu, *A_blue_gpu;	/*方程A矩阵的gpu内存*/
		double *b_blue_gpu, *b_green_gpu, *b_red_gpu;	/*方程b向量的gpu内存*/
		CHECK(cudaMalloc(&A_red_gpu, sizeof(double) * 2 * inlier_siz));
		CHECK(cudaMalloc(&A_green_gpu, sizeof(double) * 2 * inlier_siz));
		CHECK(cudaMalloc(&A_blue_gpu, sizeof(double) * 2 * inlier_siz));
		CHECK(cudaMalloc(&b_red_gpu, sizeof(double) * inlier_siz));
		CHECK(cudaMalloc(&b_green_gpu, sizeof(double) * inlier_siz));
		CHECK(cudaMalloc(&b_blue_gpu, sizeof(double) * inlier_siz));
		vector<Point2f> inliers_left, inliers_right;
		for (int i = 0; i < inlier_siz; i++) {
			inliers_left.push_back(p1[inlier_id[i]]);
			inliers_right.push_back(p2[inlier_id[i]]);
		}
		Point2f *inliers_left_gpu, *inliers_right_gpu;
		CHECK(cudaMalloc(&inliers_left_gpu, sizeof(Point2f) * inlier_siz));
		CHECK(cudaMalloc(&inliers_right_gpu, sizeof(Point2f) * inlier_siz));
		CHECK(cudaMemcpy(inliers_left_gpu, &inliers_left[0], sizeof(Point2f) * inlier_siz, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(inliers_right_gpu, &inliers_right[0], sizeof(Point2f) * inlier_siz, cudaMemcpyHostToDevice));
		ConstructA_8UC4 << <1, 1 >> > (left_gpu, inliers_left_gpu, right_gpu, inliers_right_gpu, inlier_siz, A_red_gpu, b_red_gpu, A_green_gpu, b_green_gpu, A_blue_gpu, b_blue_gpu);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(&A_blue[0], A_blue_gpu, sizeof(double) * inlier_siz * 2, cudaMemcpyDeviceToHost));	/*数据下载到cpu*/
		CHECK(cudaMemcpy(&A_green[0], A_green_gpu, sizeof(double) * inlier_siz * 2, cudaMemcpyDeviceToHost));	/*数据下载到cpu*/
		CHECK(cudaMemcpy(&A_red[0], A_red_gpu, sizeof(double) * inlier_siz * 2, cudaMemcpyDeviceToHost));	/*数据下载到cpu*/
		CHECK(cudaMemcpy(&b_blue[0], b_blue_gpu, sizeof(double) * inlier_siz, cudaMemcpyDeviceToHost));	/*数据下载到cpu*/
		CHECK(cudaMemcpy(&b_green[0], b_green_gpu, sizeof(double) * inlier_siz, cudaMemcpyDeviceToHost));	/*数据下载到cpu*/
		CHECK(cudaMemcpy(&b_red[0], b_red_gpu, sizeof(double) * inlier_siz, cudaMemcpyDeviceToHost));	/*数据下载到cpu*/
		CHECK(cudaFree(A_blue_gpu));	/*gpu内存释放*/
		CHECK(cudaFree(A_green_gpu));	/*gpu内存释放*/
		CHECK(cudaFree(A_red_gpu));	/*gpu内存释放*/
		CHECK(cudaFree(b_blue_gpu));	/*gpu内存释放*/
		CHECK(cudaFree(b_green_gpu));	/*gpu内存释放*/
		CHECK(cudaFree(b_red_gpu));	/*gpu内存释放*/
		CHECK(cudaFree(inliers_left_gpu));	/*gpu内点数据释放*/
		CHECK(cudaFree(inliers_right_gpu));	/*gpu内点数据释放*/

		//解超定方程组，求解曝光补偿参数
		vector<double> temp;
		SolveBySVD(inlier_siz, 2, A_blue, b_blue, temp);
		alpha_blue = temp[0];
		beta_blue = temp[1];
		SolveBySVD(inlier_siz, 2, A_green, b_green, temp);
		alpha_green = temp[0];
		beta_green = temp[1];
		SolveBySVD(inlier_siz, 2, A_red, b_red, temp);
		alpha_red = temp[0];
		beta_red = temp[1];

		

		cout << "初始化完成\n";
		return 0;
	}

	//使用现有的H拼接两幅图片
	int stitch(cuda::GpuMat left_gpu, cuda::GpuMat right_gpu, Mat &result) {
		//剔除右图多余部分
		cuda::GpuMat right_temp_gpu;
		right_gpu.copyTo(right_temp_gpu);
		dim3 block_siz(32, 32);
		dim3 block_num((right_gpu.cols + block_siz.x - 1) / block_siz.x, (right_gpu.rows + block_siz.y - 1) / block_siz.y);
		if (right_gpu.type() == CV_8UC4) {
			ClearSurplus_8UC4 << <block_num, block_siz >> > (right_temp_gpu, H_gpu);
			CHECK(cudaGetLastError());
			CHECK(cudaDeviceSynchronize());
		}

		//右图进行透视变换
		cuda::warpPerspective(right_temp_gpu, right_temp_gpu, H, result_siz);

		//右图高斯模糊,低效
		//Mat temp;
		//right_temp_gpu.download(temp);
		//GaussianBlur(temp, temp, Size(3, 3), 3, 3);
		//right_temp_gpu.upload(temp);

		//右图曝光差异矫正
		//int t1 = clock();
		block_num = dim3((right_temp_gpu.cols + block_siz.x - 1) / block_siz.x, (right_temp_gpu.rows + block_siz.y - 1) / block_siz.y);
		GainCompensation_8UC4 << <block_num, block_siz >> > (right_temp_gpu, alpha_blue, beta_blue, alpha_green, beta_green, alpha_red, beta_red);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());
		//int t2 = clock();
		//cout << "gain: " << t2 - t1 << endl;

		//求最佳缝合线
		cuda::GpuMat seam_gpu(result_siz.height, 1, CV_32SC1, Scalar(1800));
		cuda::GpuMat left_temp_gpu(result_siz, CV_8UC4);
		left_gpu.copyTo(left_temp_gpu(Rect(0, 0, left_gpu.cols, left_gpu.rows)));
		GetOptimalSeam_CPU(left_temp_gpu, right_temp_gpu, seam_gpu, overlap_left, overlap_right);

		//最佳缝合线拼接
		cuda::GpuMat result_gpu(result_siz, CV_8UC4);	
		{
			cuda::GpuMat left_temp_gpu(result_siz, CV_8UC4, Scalar(0, 0, 0, 0));
			left_gpu.copyTo(left_temp_gpu(Rect(0, 0, left_gpu.cols, left_gpu.rows)));
			dim3 block_siz(32, 32);
			dim3 block_num((result_gpu.cols + block_siz.x - 1) / block_siz.x, (result_gpu.rows + block_siz.y - 1) / block_siz.y);
			ImageBlend << <block_num, block_siz >> > (left_temp_gpu, right_temp_gpu, seam_gpu, result_gpu, 20);
			CHECK(cudaGetLastError());
			CHECK(cudaDeviceSynchronize());
		}
		

		//绘制缝合线
		//DrawSeam << <1, 1 >> > (result_gpu, seam_gpu);
		//CHECK(cudaGetLastError());
		//CHECK(cudaDeviceSynchronize());

		//下载最终结果
		cuda::resize(result_gpu, result_gpu, Size(0, 0), 0.3, 0.3);
		result_gpu.download(result);
		return 0;
	}
};
#pragma once
