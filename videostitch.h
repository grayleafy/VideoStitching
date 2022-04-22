#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <thrust/extrema.h>	/*CUDA��ѧ����*/

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


//���͸�ӱ任��ᷴת�Ķ��ಿ��
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

//CUDA�ع�������
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

//�������������꣬��ȡ�����ػҶ�, ��������CV_8UC4
__device__ double GetScale(cuda::PtrStepSz<uchar4> image, Point2f point, int channel) {
	int x_l = point.x;
	int x_r = x_l + 1;
	int y_u = point.y;
	int y_d = y_u + 1;
	double x = point.x; /*�����ص�x����*/
	double y = point.y;	/*�����ص�y����*/

	double s1, s2, s3, s4; /*�ֱ�Ϊ���ϣ����ϣ����£����µĻҶ�ֵ*/
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

//��������عⲹ���ķ���
__global__ void ConstructA_8UC4(cuda::PtrStepSz<uchar4> left_gpu, Point2f *inliers_left_gpu, cuda::PtrStepSz<uchar4> right_gpu, Point2f *inliers_right_gpu, int inliers_siz, double *A_red_gpu, double *b_red_gpu, double *A_green_gpu, double *b_green_gpu, double *A_blue_gpu, double *b_blue_gpu) {
	for (int i = 0; i < inliers_siz; i++) {
		double scale_left, scale_right;

		//��ɫͨ��
		scale_left = GetScale(left_gpu, inliers_left_gpu[i], 1);
		scale_right = GetScale(right_gpu, inliers_right_gpu[i], 1);
		A_blue_gpu[i] = scale_right;
		A_blue_gpu[inliers_siz + i] = 1.0;
		b_blue_gpu[i] = scale_left;

		//��ɫͨ��
		scale_left = GetScale(left_gpu, inliers_left_gpu[i], 2);
		scale_right = GetScale(right_gpu, inliers_right_gpu[i], 2);
		A_green_gpu[i] = scale_right;
		A_green_gpu[inliers_siz + i] = 1.0;
		b_green_gpu[i] = scale_left;

		//��ɫͨ��
		scale_left = GetScale(left_gpu, inliers_left_gpu[i], 3);
		scale_right = GetScale(right_gpu, inliers_right_gpu[i], 3);
		A_red_gpu[i] = scale_right;
		A_red_gpu[inliers_siz + i] = 1.0;
		b_red_gpu[i] = scale_left;
	}
}

//ָ���������豸����
__device__ inline double d_pow(double x, int n) {
	double res = 1;
	while (n) {
		res *= x;
		n--;
	}
	return res;
}

//����ֵ���豸����
__device__ inline double d_fabs(double x) {
	if (x > 0)	return x;
	return -x;
}

//BGRAͼ����(x, y)��x�����ݶȣ��豸��������������CV_8UC4
__device__ double GetGrand_x_BGRA(cuda::PtrStepSz<uchar4> src, int x, int y) {
	double res = 0;
	if (x - 1 < 0 || x + 1 >= src.cols || y - 1 < 0 || y + 1 >= src.rows)	return res;
	
	//��
	res += 1.0 * src(y + 1, x + 1).x;
	res += 1.0 * src(y + 1, x + 1).y;
	res += 1.0 * src(y + 1, x + 1).z;
	res -= 1.0 * src(y + 1, x - 1).x;
	res -= 1.0 * src(y + 1, x - 1).y;
	res -= 1.0 * src(y + 1, x - 1).z;
	//��
	res += 2.0 * src(y, x + 1).x;
	res += 2.0 * src(y, x + 1).y;
	res += 2.0 * src(y, x + 1).z;
	res -= 2.0 * src(y, x - 1).x;
	res -= 2.0 * src(y, x - 1).y;
	res -= 2.0 * src(y, x - 1).z;
	//��
	res += 1.0 * src(y - 1, x + 1).x;
	res += 1.0 * src(y - 1, x + 1).y;
	res += 1.0 * src(y - 1, x + 1).z;
	res -= 1.0 * src(y - 1, x - 1).x;
	res -= 1.0 * src(y - 1, x - 1).y;
	res -= 1.0 * src(y - 1, x - 1).z;

	res /= 3.0;
	return res;
}

//BGRAͼ����(x, y)��y�����ݶȣ��豸��������������CV_8UC4
__device__ double GetGrand_y_BGRA(cuda::PtrStepSz<uchar4> src, int x, int y) {
	double res = 0;
	if (x - 1 < 0 || x + 1 >= src.cols || y - 1 < 0 || y + 1 >= src.rows)	return res;

	//��
	res += 1.0 * src(y + 1, x - 1).x;
	res += 1.0 * src(y + 1, x - 1).y;
	res += 1.0 * src(y + 1, x - 1).z;
	res -= 1.0 * src(y - 1, x - 1).x;
	res -= 1.0 * src(y - 1, x - 1).y;
	res -= 1.0 * src(y - 1, x - 1).z;
	//��
	res += 2.0 * src(y + 1, x).x;
	res += 2.0 * src(y + 1, x).y;
	res += 2.0 * src(y + 1, x).z;
	res -= 2.0 * src(y - 1, x).x;
	res -= 2.0 * src(y - 1, x).y;
	res -= 2.0 * src(y - 1, x).z;
	//��
	res += 1.0 * src(y + 1, x + 1).x;
	res += 1.0 * src(y + 1, x + 1).y;
	res += 1.0 * src(y + 1, x + 1).z;
	res -= 1.0 * src(y - 1, x + 1).x;
	res -= 1.0 * src(y - 1, x + 1).y;
	res -= 1.0 * src(y - 1, x + 1).z;

	res /= 3.0;
	return res;
}

//�Ҷ�ͼ����(x, y)��x�����ݶȣ�������Χ4����Ĳ�ֵ���豸����
__device__ inline double GetGrand_x(cuda::PtrStepSz<uchar> src, int x, int y) {
	double res = 0;
	if (x - 1 < 0 || x + 1 >= src.cols || y - 1 < 0 || y + 1 >= src.rows)	return res;
	else {
		//��
		res += 2.0 * src(y + 1, x + 1);
		res -= 2.0 * src(y + 1, x - 1);
		//��
		res += 1.0 * src(y, x + 1);
		res -= 1.0 * src(y, x - 1);
		//��
		res += 2.0 * src(y - 1, x + 1);
		res -= 2.0 * src(y - 1, x - 1);

		return res;
	}	
}

//�Ҷ�ͼ����(x, y)��y�����ݶȣ�������Χ4����Ĳ�ֵ���豸����
__device__ inline double GetGrand_y(cuda::PtrStepSz<uchar> src, int x, int y) {
	double res = 0;
	if (x - 1 < 0 || x + 1 >= src.cols || y - 1 < 0 || y + 1 >= src.rows)	return res;
	else {
		//��
		res += 2.0 * src(y + 1, x - 1);
		res -= 2.0 * src(y - 1, x - 1);
		//��
		res += 1.0 * src(y + 1, x);
		res -= 1.0 * src(y - 1, x);
		//��
		res += 2.0 * src(y + 1, x + 1);
		res -= 2.0 * src(y - 1, x + 1);

		return res;
	}
}

//����Ҷ�ֵ��ƽ��ֵ
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


//CUDA������ѷ����������������������CV_GRAY
__global__ void CalE(cuda::PtrStepSz<uchar> left, cuda::PtrStepSz<uchar> right, cuda::PtrStepSz<double> E) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= left.cols || y >= left.rows)	return;
	E(y, x) = 0;

	//�޹ز���
	if (left(y, x) == 0 && right(y, x) == 0) {
		E(y, x) = 0;
		return;
	}
	//�߽��
	if (x - 1 < 0 || x + 1 >= left.cols || y - 1 < 0 || y + 1 >= left.rows) {
		E(y, x) = 2000.0;
		return;
	}

	//��ɫ����
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



	//�ݶȲ���
	double res = 0;
	res = 0;
	//��
	res += 2.0 * left(y + 1, x + 1);
	res -= 2.0 * left(y + 1, x - 1);
	//��
	res += 1.0 * left(y, x + 1);
	res -= 1.0 * left(y, x - 1);
	//��
	res += 2.0 * left(y - 1, x + 1);
	res -= 2.0 * left(y - 1, x - 1);
	double grand_left_x = res; /*left,x����*/

	res = 0;
	//��
	res += 2.0 * left(y + 1, x - 1);
	res -= 2.0 * left(y - 1, x - 1);
	//��
	res += 1.0 * left(y + 1, x);
	res -= 1.0 * left(y - 1, x);
	//��
	res += 2.0 * left(y + 1, x + 1);
	res -= 2.0 * left(y - 1, x + 1);
	double grand_left_y = res;

	res = 0;
	//��
	res += 2.0 * right(y + 1, x + 1);
	res -= 2.0 * right(y + 1, x - 1);
	//��
	res += 1.0 * right(y, x + 1);
	res -= 1.0 * right(y, x - 1);
	//��
	res += 2.0 * right(y - 1, x + 1);
	res -= 2.0 * right(y - 1, x - 1);
	double grand_right_x = res;

	res = 0;
	//��
	res += 2.0 * right(y + 1, x - 1);
	res -= 2.0 * right(y - 1, x - 1);
	//��
	res += 1.0 * right(y + 1, x);
	res -= 1.0 * right(y - 1, x);
	//��
	res += 2.0 * right(y + 1, x + 1);
	res -= 2.0 * right(y - 1, x + 1);
	double grand_right_y = res;

	E(y, x) += d_fabs((grand_left_x - grand_right_x) * (grand_left_y - grand_right_y));
	return;
}

//CUDA��̬�滮������ѷ���ߣ�x������С�ڵ���1024����һ��block������߳���
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
		//����ѡ��ͬһ��
	}
}

//CUDA��̬�滮������ѷ���ߣ�ÿ�μ���ͬһy�����һ�㣬x�����ȴ���1024����һ��block������߳���
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
	//����ѡ��ͬһ��
	return;
}

//����·�����ݣ���������x����
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

//CUDA�����ѷ���ߣ�left��right�Ĵ�С����ȣ�,����0��ɹ�����������CV_8UC4��CV_GRAY
int GetOptimalSeam(cuda::GpuMat left, cuda::GpuMat right, cuda::GpuMat &seam, int overlap_left, int overlap_right) {
	if (left.size() != right.size()) {
		cout << "ͼƬ��С��һ�£������ѷ����ʧ��\n";
		return -1;
	}
	if (left.type() == CV_8UC4) cuda::cvtColor(left, left, COLOR_BGRA2GRAY);
	if (right.type() == CV_8UC4) cuda::cvtColor(right, right, COLOR_BGRA2GRAY);
	int rows = left.rows;
	int cols = left.cols;
	seam = cuda::GpuMat(rows, 1, CV_32SC1);
	cuda::GpuMat E(rows, cols, CV_64FC1);

	//������������
	dim3 block_siz(32, 32);
	dim3 block_num((cols + block_siz.x - 1) / block_siz.x, (rows + block_siz.y - 1) / block_siz.y);
	CalE << <block_num, block_siz >> > (left, right, E);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	//��ʾ��������
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
	
	//��̬�滮������ѷ����·�������������
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
		//cout << "��̬�滮��ʱ:" << t2 - t1 << endl;
	}

	//���ݵõ������x����,gpu
	{
		//int t1 = clock();
		CalSeam << <1, 1 >> > (seam, path);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());
		//int t2 = clock();
		//cout << "���ݺ�ʱ:" << t2 - t1 << endl;
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

//CUDA������ѷ����������������������CV_GRAY��ֻ�����ص�����
__global__ void CalE(cuda::PtrStepSz<uchar> left, cuda::PtrStepSz<uchar> right, cuda::PtrStepSz<double> E, int from) {
	int x = blockIdx.x * blockDim.x + threadIdx.x + from;
	int ex = x - from;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= left.cols || y >= left.rows)	return;
	E(y, ex) = 0;

	//�޹ز���
	if (left(y, x) == 0 && right(y, x) == 0) {
		E(y, ex) = 0;
		return;
	}
	//�߽��
	if (x - 1 < 0 || x + 1 >= left.cols || y - 1 < 0 || y + 1 >= left.rows) {
		E(y, ex) = 2000.0;
		return;
	}

	//��ɫ����
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



	//�ݶȲ���
	double res = 0;
	res = 0;
	//��
	res += 2.0 * left(y + 1, x + 1);
	res -= 2.0 * left(y + 1, x - 1);
	//��
	res += 1.0 * left(y, x + 1);
	res -= 1.0 * left(y, x - 1);
	//��
	res += 2.0 * left(y - 1, x + 1);
	res -= 2.0 * left(y - 1, x - 1);
	double grand_left_x = res; /*left,x����*/

	res = 0;
	//��
	res += 2.0 * left(y + 1, x - 1);
	res -= 2.0 * left(y - 1, x - 1);
	//��
	res += 1.0 * left(y + 1, x);
	res -= 1.0 * left(y - 1, x);
	//��
	res += 2.0 * left(y + 1, x + 1);
	res -= 2.0 * left(y - 1, x + 1);
	double grand_left_y = res;

	res = 0;
	//��
	res += 2.0 * right(y + 1, x + 1);
	res -= 2.0 * right(y + 1, x - 1);
	//��
	res += 1.0 * right(y, x + 1);
	res -= 1.0 * right(y, x - 1);
	//��
	res += 2.0 * right(y - 1, x + 1);
	res -= 2.0 * right(y - 1, x - 1);
	double grand_right_x = res;

	res = 0;
	//��
	res += 2.0 * right(y + 1, x - 1);
	res -= 2.0 * right(y - 1, x - 1);
	//��
	res += 1.0 * right(y + 1, x);
	res -= 1.0 * right(y - 1, x);
	//��
	res += 2.0 * right(y + 1, x + 1);
	res -= 2.0 * right(y - 1, x + 1);
	double grand_right_y = res;

	E(y, ex) += d_fabs((grand_left_x - grand_right_x) * (grand_left_y - grand_right_y));
	return;
}

//CUDA��GPU�����ѷ���ߣ�left��right�Ĵ�С����ȣ�,����0��ɹ�����������CV_8UC4��CV_GRAY
int GetOptimalSeam_CPU(cuda::GpuMat left, cuda::GpuMat right, cuda::GpuMat &seam, int overlap_left, int overlap_right) {
	if (left.size() != right.size()) {
		cout << "ͼƬ��С��һ�£������ѷ����ʧ��\n";
		return -1;
	}
	if (left.type() == CV_8UC4) cuda::cvtColor(left, left, COLOR_BGRA2GRAY);
	if (right.type() == CV_8UC4) cuda::cvtColor(right, right, COLOR_BGRA2GRAY);
	int rows = left.rows;
	int cols = left.cols;
	int overlap_len = overlap_right - overlap_left + 1;
	
	

	//������������
	cuda::GpuMat E(rows, overlap_len, CV_64FC1);
	dim3 block_siz(32, 32);
	dim3 block_num((overlap_len + block_siz.x - 1) / block_siz.x, (rows + block_siz.y - 1) / block_siz.y);
	CalE << <block_num, block_siz >> > (left, right, E, overlap_left);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	//��̬�滮�����·��
	Mat E_cpu;
	E.download(E_cpu);
	Mat path(rows, overlap_len, CV_64FC2, Scalar(inf, inf));
	//���һ�г�ʼ��
	double *p_path = (double*)path.data + (rows - 1) * path.step1();
	double *p_E = (double*)E_cpu.data + (rows - 1) * E_cpu.step1();
	for (int x = 0; x < overlap_len; x++) {
		p_path[0] = p_E[0];
		p_path += 2;
		p_E++;
	}
	//���ϸ���
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



	//���ݵõ������x����,gpu
	{
		Mat seam_cpu(rows, 1, CV_32S);
		//�ӵ�һ������С
		seam_cpu.at<int>(0, 0) = 0;
		double temp = inf;
		double *p_path = (double*)path.data;
		for (int x = 0; x < overlap_len; x++) {
			if (p_path[2 * x] < temp) {
				temp = p_path[2 * x];
				seam_cpu.at<int>(0, 0) = x + overlap_left;
			}
		}
		//���¸���
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

//���Ʒ����
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

//��ѷ����ͼ���ں�, ��������CV_8UC4
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

//��Ƶƴ����
class VideoStitcher {
	Mat H;
	cuda::GpuMat H_gpu;
	Size left_siz, right_siz;
	Size result_siz;
	double alpha_blue, alpha_green, alpha_red;
	double beta_blue, beta_green, beta_red;
	int overlap_left, overlap_right; /*c�ص���������x���꣬����ұ�*/
public:
	//��ʼ��ͼ���С����Ӧ�Ծ���
	int init(cuda::GpuMat left_gpu, cuda::GpuMat right_gpu) {
		cout << "��ʼ����......\n";
		//���ͼƬΪ�����˳�
		if (left_gpu.empty() || right_gpu.empty()) {
			cout << "ͼƬΪ�գ���ʼ��ʧ��\n";
			return -1;
		}

		//���ͼƬ����4ͨ�����˳�
		if (left_gpu.type() != CV_8UC4) {
			cout << "ͼƬ����CV_8UC4���ͣ���ʼ��ʧ��\n";
			return -1;
		}

		//�����С����
		left_siz = left_gpu.size();
		right_siz = right_gpu.size();
		result_siz = Size(2.5 * left_siz.width, 1.5 * left_siz.height);

		//����ͼƬ����Ч
		Mat left, right;
		left_gpu.download(left);
		right_gpu.download(right);

		//��˹ģ������Ч
		GaussianBlur(left, left, Size(3, 3), 3, 3);
		GaussianBlur(right, right, Size(3, 3), 3, 3);
		left_gpu.upload(left);
		right_gpu.upload(right);

		//SIFT��ȡ�����㣬������������
		int minHessian = 400;
		Mat descriptors1, descriptors2;
		vector<KeyPoint> keypoint1, keypoint2;
		Ptr<SIFT> sift = SIFT::create();   /*��ѡ400*/
		sift->detectAndCompute(left, Mat(), keypoint1, descriptors1);
		sift->detectAndCompute(right, Mat(), keypoint2, descriptors2);

		//������ƥ��
		vector<vector<DMatch> > matches;
		cuda::GpuMat descriptors1_gpu, descriptors2_gpu;
		descriptors1_gpu.upload(descriptors1);
		descriptors2_gpu.upload(descriptors2);
		Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L1);
		matcher->knnMatch(descriptors1_gpu, descriptors2_gpu, matches, 2);

		//Low's �㷨�޳�����ƥ��
		vector<DMatch> good_matches;
		for (int i = 0; i < matches.size(); i++) {
			if (matches[i][0].distance < 0.5 * matches[i][1].distance) {
				good_matches.push_back(matches[i][0]);
			}
		}

		//ѡȡƥ���Խ�������
		vector<Point2f> p1, p2;
		for (int i = 0; i < good_matches.size(); i++) {
			p1.push_back(keypoint1[good_matches[i].queryIdx].pt);
			p2.push_back(keypoint2[good_matches[i].trainIdx].pt);
		}
		if (p1.size() < 4) {
			cout << "ƥ����������㣬��ʼ��ʧ��\n";
			return -1;
		}

		//����findHomography��ⵥӦ�Է��̣��������ڵ�
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
			cout << "������Ӧ�Ծ���ʧ��\n";
			return -1;
		}

		//���浥Ӧ�Ծ�����
		temp_H.copyTo(H);
		H_gpu.upload(H);

		//�����ص����������x���꣬��ת�������
		{
			overlap_right = left_gpu.cols - 1;
			overlap_left = left_gpu.cols - 1;
			Point2d temp1 = Point2d(0, 0);
			Point2d res = mul(temp1, H);	/*ʹ�õ���H*/
			overlap_left = min(overlap_left, (int)res.x);
			Point2d temp2 = Point2d(0, right_gpu.rows - 1);
			res = mul(temp2, H);	/*ʹ�õ���H*/
			overlap_left = min(overlap_left, (int)res.x);
			overlap_left = max(overlap_left, 0);
			if (judge_reverse(temp1, H) || judge_reverse(temp2, H)) {
				overlap_left = 0;
			}
			cout << "��߽�:" << overlap_left << endl;
			cout << "�ұ߽�:" << overlap_right << endl;
		}
		

		//�����عⲹ������, CV_8UC4
		int inlier_siz = inlier_id.size();
		vector<double> A_red(2 * inlier_siz), A_green(2 * inlier_siz), A_blue(2 * inlier_siz);	/*����A����*/
		vector<double> b_blue(inlier_siz), b_green(inlier_siz), b_red(inlier_siz);	/*����b����*/
		double *A_red_gpu, *A_green_gpu, *A_blue_gpu;	/*����A�����gpu�ڴ�*/
		double *b_blue_gpu, *b_green_gpu, *b_red_gpu;	/*����b������gpu�ڴ�*/
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
		CHECK(cudaMemcpy(&A_blue[0], A_blue_gpu, sizeof(double) * inlier_siz * 2, cudaMemcpyDeviceToHost));	/*�������ص�cpu*/
		CHECK(cudaMemcpy(&A_green[0], A_green_gpu, sizeof(double) * inlier_siz * 2, cudaMemcpyDeviceToHost));	/*�������ص�cpu*/
		CHECK(cudaMemcpy(&A_red[0], A_red_gpu, sizeof(double) * inlier_siz * 2, cudaMemcpyDeviceToHost));	/*�������ص�cpu*/
		CHECK(cudaMemcpy(&b_blue[0], b_blue_gpu, sizeof(double) * inlier_siz, cudaMemcpyDeviceToHost));	/*�������ص�cpu*/
		CHECK(cudaMemcpy(&b_green[0], b_green_gpu, sizeof(double) * inlier_siz, cudaMemcpyDeviceToHost));	/*�������ص�cpu*/
		CHECK(cudaMemcpy(&b_red[0], b_red_gpu, sizeof(double) * inlier_siz, cudaMemcpyDeviceToHost));	/*�������ص�cpu*/
		CHECK(cudaFree(A_blue_gpu));	/*gpu�ڴ��ͷ�*/
		CHECK(cudaFree(A_green_gpu));	/*gpu�ڴ��ͷ�*/
		CHECK(cudaFree(A_red_gpu));	/*gpu�ڴ��ͷ�*/
		CHECK(cudaFree(b_blue_gpu));	/*gpu�ڴ��ͷ�*/
		CHECK(cudaFree(b_green_gpu));	/*gpu�ڴ��ͷ�*/
		CHECK(cudaFree(b_red_gpu));	/*gpu�ڴ��ͷ�*/
		CHECK(cudaFree(inliers_left_gpu));	/*gpu�ڵ������ͷ�*/
		CHECK(cudaFree(inliers_right_gpu));	/*gpu�ڵ������ͷ�*/

		//�ⳬ�������飬����عⲹ������
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

		

		cout << "��ʼ�����\n";
		return 0;
	}

	//ʹ�����е�Hƴ������ͼƬ
	int stitch(cuda::GpuMat left_gpu, cuda::GpuMat right_gpu, Mat &result) {
		//�޳���ͼ���ಿ��
		cuda::GpuMat right_temp_gpu;
		right_gpu.copyTo(right_temp_gpu);
		dim3 block_siz(32, 32);
		dim3 block_num((right_gpu.cols + block_siz.x - 1) / block_siz.x, (right_gpu.rows + block_siz.y - 1) / block_siz.y);
		if (right_gpu.type() == CV_8UC4) {
			ClearSurplus_8UC4 << <block_num, block_siz >> > (right_temp_gpu, H_gpu);
			CHECK(cudaGetLastError());
			CHECK(cudaDeviceSynchronize());
		}

		//��ͼ����͸�ӱ任
		cuda::warpPerspective(right_temp_gpu, right_temp_gpu, H, result_siz);

		//��ͼ��˹ģ��,��Ч
		//Mat temp;
		//right_temp_gpu.download(temp);
		//GaussianBlur(temp, temp, Size(3, 3), 3, 3);
		//right_temp_gpu.upload(temp);

		//��ͼ�ع�������
		//int t1 = clock();
		block_num = dim3((right_temp_gpu.cols + block_siz.x - 1) / block_siz.x, (right_temp_gpu.rows + block_siz.y - 1) / block_siz.y);
		GainCompensation_8UC4 << <block_num, block_siz >> > (right_temp_gpu, alpha_blue, beta_blue, alpha_green, beta_green, alpha_red, beta_red);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());
		//int t2 = clock();
		//cout << "gain: " << t2 - t1 << endl;

		//����ѷ����
		cuda::GpuMat seam_gpu(result_siz.height, 1, CV_32SC1, Scalar(1800));
		cuda::GpuMat left_temp_gpu(result_siz, CV_8UC4);
		left_gpu.copyTo(left_temp_gpu(Rect(0, 0, left_gpu.cols, left_gpu.rows)));
		GetOptimalSeam_CPU(left_temp_gpu, right_temp_gpu, seam_gpu, overlap_left, overlap_right);

		//��ѷ����ƴ��
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
		

		//���Ʒ����
		//DrawSeam << <1, 1 >> > (result_gpu, seam_gpu);
		//CHECK(cudaGetLastError());
		//CHECK(cudaDeviceSynchronize());

		//�������ս��
		cuda::resize(result_gpu, result_gpu, Size(0, 0), 0.3, 0.3);
		result_gpu.download(result);
		return 0;
	}
};
#pragma once
