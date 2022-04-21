#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<curand.h>


#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/calib3d/calib3d.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<cstdio>
#include<vector>
#include<cmath>
#include<ctime>
#include<iostream>

//#include "cusolver_utils.h";
#include"svd.h";

using namespace std;
using namespace cv;
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)


static long long int k = 4000;


static void CheckCUDAError(const char *msg = NULL)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

//cpu ransac
Mat findHomography_cpu(vector<Point2f> p1, vector<Point2f> p2, double projthreshold) {
	srand(time(NULL));
	double e = 0.7;
	double k = log(1.0 - 0.97) / log(1.0 - pow(e, 4));
	cout << "init k = " << k << endl;
	int siz = p1.size();

	Mat H;
	int inner = 0;
	for (int t = 0; t < k; t++) {
		//cout << endl << "t = " << t << endl;
		int x1 = rand() % siz;
		int x2 = x1, x3 = x2, x4 = x3;
		while (x2 == x1)	x2 = rand() % siz;
		while (x3 == x2 || x3 == x1)	x3 = rand() % siz;
		while (x4 == x3 || x4 == x2 || x4 == x1)	x4 = rand() % siz;
		vector<Point2f> temp_p1, temp_p2;
		temp_p1.push_back(p1[x1]);
		temp_p1.push_back(p1[x2]);
		temp_p1.push_back(p1[x3]);
		temp_p1.push_back(p1[x4]);

		temp_p2.push_back(p2[x1]);
		temp_p2.push_back(p2[x2]);
		temp_p2.push_back(p2[x3]);
		temp_p2.push_back(p2[x4]);
	//	cout << x1 << ' ' << x2 << ' ' << x3 << ' ' << x4 << endl;
		Mat temp_h;
		int temp_inner = 0;
		try
		{
			temp_h = findHomography(temp_p1, temp_p2, RANSAC, projthreshold);
		}
		catch (const Exception err)
		{
			cout << err.msg << endl;
			return H;
		}
		//cout << "temp_h : " << temp_h << endl;
		for (int i = 0; i < siz; i++) {
			Mat temp_p = (Mat_<double>(3, 1) << p1[i].x, p1[i].y, 1.0);
			Mat y = temp_h * temp_p;
			y /= y.at<double>(2, 0);
	//		cout << "i = " << i << ":  " << "x = " << temp_p << "  y = " << y << endl;

			double len = (y.at<double>(0, 0) - p2[i].x) * (y.at<double>(0, 0) - p2[i].x) + (y.at<double>(1, 0) - p2[i].y) * (y.at<double>(1, 0) - p2[i].y);
			if (len <= projthreshold * projthreshold)	temp_inner++;
		}

		if (temp_inner > inner) {
			temp_h.copyTo(H);
			inner = temp_inner;
			e = (double)inner / (double)siz;
			cout << "e = " << e << endl;
			k = log(1.0 - 0.97) / log(1.0 - pow(e, 4));
			cout << "k = " << k << endl;
		}
	//	
	}
	return H;
}

__device__ void gaussian(double *a, double *H) {
	for (int i = 0; i < 8; i++) {
		int pos = i;
		for (int j = i + 1; j < 8; j++) {
			if (fabs(a[j * 9 + i]) > fabs(a[pos * 9 + i])) {
				pos = j;
			}
		}
		for (int j = i; j < 9; j++) {
			double temp;
			temp = a[i * 9 + j];
			a[i * 9 + j] = a[pos * 9 + j];
			a[pos * 9 + j] = temp;
		}


		for (int j = 0; j < 8; j++) {
			if (j != i) {
				double temp = a[j * 9 + i];
				a[j * 9 + i] = 0;
				for (int k = i + 1; k < 9; k++) {
					a[j * 9 + k] -= temp / a[i * 9 + i] * a[i * 9 + k];
				}
			}
		}
	}

	//printf("\naa:\n");
	//for (int i = 0; i < 8; i++) {
	//	for (int j = 0; j < 9; j++) {
	//		printf("%f ", a[i * 9 + j]);
	//	}
	//	printf("\n");
	//}

	H[8] = 1;
	for (int i = 0; i < 8; i++) {
		H[i] = -a[i * 9 + 8] / a[i * 9 + i];
	}
	return;
}

__device__ int evalHomography(double *H, Point2f *p1, Point2f *p2, int siz, double projthreshold) {
	int cnt = 0;
	for (int i = 0; i < siz; i++) {
		double x = H[0] * p1[i].x + H[1] * p1[i].y + H[2];
		double y = H[3] * p1[i].x + H[4] * p1[i].y + H[5];
		double z = H[6] * p1[i].x + H[7] * p1[i].y + H[8];
		x /= z;
		y /= z;

		if ((x - p2[i].x) * (x - p2[i].x) + (y - p2[i].y) * (y - p2[i].y) <= projthreshold * projthreshold)	cnt++;
	}
	return cnt;
}

__global__ void calHomography(Point2f *p1, Point2f *p2, int siz, int k_max, int *gpu_rand_list, double *gpu_H, int *inliers_list, double projthreshold) { //const
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= k_max)	return;
	//printf("n = %d\n", n);
	int rand_list[4];
	for (int i = 0; i < 4; i++) {
		rand_list[i] = gpu_rand_list[4 * n + i];
	}
	inliers_list[n] = 0;
	if (rand_list[0] == rand_list[1] || rand_list[0] == rand_list[2] || rand_list[0] == rand_list[3]) return;
	if (rand_list[1] == rand_list[2] || rand_list[1] == rand_list[3])	return;
	if (rand_list[2] == rand_list[3])	return;

	double *H = &gpu_H[n * 9];
	double a[8][9];
	for (int i = 0; i < 4; i++) {
		a[i * 2][0] = p1[rand_list[i]].x;
		a[i * 2][1] = p1[rand_list[i]].y;
		a[i * 2][2] = 1;
		a[i * 2][3] = a[i * 2][4] = a[i * 2][5] = 0;
		a[i * 2][6] = -p2[rand_list[i]].x * p1[rand_list[i]].x;
		a[i * 2][7] = -p2[rand_list[i]].x * p1[rand_list[i]].y;
		a[i * 2][8] = -p2[rand_list[i]].x;

		a[i * 2 + 1][0] = a[i * 2 + 1][1] = a[i * 2 + 1][2] = 0;
		a[i * 2 + 1][3] = p1[rand_list[i]].x;
		a[i * 2 + 1][4] = p1[rand_list[i]].y;
		a[i * 2 + 1][5] = 1;	
		a[i * 2 + 1][6] = -p2[rand_list[i]].y * p1[rand_list[i]].x;
		a[i * 2 + 1][7] = -p2[rand_list[i]].y * p1[rand_list[i]].y;
		a[i * 2 + 1][8] = -p2[rand_list[i]].y;
	}

	//for (int i = 0; i < 4; i++) {
	//	printf("p1 %d = (%f, %f)\n", i, p1[rand_list[i]].x, p1[rand_list[i]].y);
	//	printf("p2 %d = (%f, %f)\n", i, p2[rand_list[i]].x, p2[rand_list[i]].y);
	//}

	//printf("a: \n");
	//for (int i = 0; i < 8; i++) {
	//	for (int j = 0; j < 9; j++) {
	//		printf("%f ", a[i][j]);
	//	}
	//	printf("\n");
	//}

	gaussian(&a[0][0], H);
	inliers_list[n] = evalHomography(H, p1, p2, siz, projthreshold);
	return;
}

Mat findHomography_gpu(const vector<Point2f> &p1, const vector<Point2f> &p2, double projthreshold, vector<int> &inlier_id) {
	//int k = 5000;
	//double e = 0.2;
	//k = log(1.0 - 0.97) / log(1.0 - pow(e, 4));
	//k = 5000;
	//printf("k = %d\n", k);

	int blocksize = 256;
	int siz = p1.size();
	int blocknum = (k + blocksize - 1) / blocksize;

	int start_time = clock();

	srand(time(NULL));
	vector<int> rand_list(k * 4);
	for (int i = 0; i < rand_list.size(); i++) {
		rand_list[i] = rand() % siz;
	}
	int *gpu_rand_list;
	CHECK(cudaMalloc(&gpu_rand_list, sizeof(int) * rand_list.size()));
	CHECK(cudaMemcpy(gpu_rand_list, &rand_list[0], sizeof(int) * rand_list.size(), cudaMemcpyHostToDevice));
	int rand_time = clock();
	//cout << "生成随机数耗时: " << rand_time - start_time << endl;

	Point2f *gpu_p1, *gpu_p2;
	CHECK(cudaMalloc(&gpu_p1, sizeof(Point2f) * siz));
	CHECK(cudaMemcpy(gpu_p1, &p1[0], sizeof(Point2f) * siz, cudaMemcpyHostToDevice));
	CHECK(cudaMalloc(&gpu_p2, sizeof(Point2f) * siz));
	CHECK(cudaMemcpy(gpu_p2, &p2[0], sizeof(Point2f) * siz, cudaMemcpyHostToDevice));
	double *gpu_H;
	CHECK(cudaMalloc(&gpu_H, sizeof(double) * 9 * k));
	int *inliers_list;
	CHECK(cudaMalloc(&inliers_list, sizeof(int) * k));
	

	calHomography << <blocknum, blocksize >> > (gpu_p1, gpu_p2, siz, k, gpu_rand_list, gpu_H, inliers_list, projthreshold);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	vector<int> inliers(k);
	cudaMemcpy(&inliers[0], inliers_list, sizeof(int) * k, cudaMemcpyDeviceToHost);
	int pos = 0;
	for (int i = 0; i < k; i++) {
	//	printf("inliers %d = %d\n", i, inliers[i]);
		if (inliers[i] > inliers[pos])	pos = i;
	}
	//printf("pos = %d inliers = %d, siz = %d\n", pos, inliers[pos], siz);
	double H[9];

	CHECK(cudaMemcpy(H, gpu_H + (9 * pos), sizeof(double) * 9, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(gpu_rand_list));
	CHECK(cudaFree(gpu_p1));
	CHECK(cudaFree(gpu_p2));
	CHECK(cudaFree(gpu_H));
	CHECK(cudaFree(inliers_list));

	//计算内点
	inlier_id.clear();
	int n = 9;	/*svd cols*/
	int m = 0;	/*svd lines*/
	vector<double> a_temp;
	//int t1 = clock();
	for (int i = 0; i < siz; i++) {
		double x = H[0] * p1[i].x + H[1] * p1[i].y + H[2];
		double y = H[3] * p1[i].x + H[4] * p1[i].y + H[5];
		double z = H[6] * p1[i].x + H[7] * p1[i].y + H[8];
		x /= z;
		y /= z;
		x -= p2[i].x;
		y -= p2[i].y;

		if (x * x + y * y <= projthreshold * projthreshold) {
			inlier_id.push_back(i);
			m += 2;
			a_temp.push_back(-p1[i].x);
			a_temp.push_back(-p1[i].y);
			a_temp.push_back(-1);

			a_temp.push_back(0);
			a_temp.push_back(0);
			a_temp.push_back(0);

			a_temp.push_back(p2[i].x * p1[i].x);
			a_temp.push_back(p2[i].x * p1[i].y);
			a_temp.push_back(p2[i].x);


			a_temp.push_back(0);
			a_temp.push_back(0);
			a_temp.push_back(0);

			a_temp.push_back(-p1[i].x);
			a_temp.push_back(-p1[i].y);
			a_temp.push_back(-1);

			a_temp.push_back(p2[i].y * p1[i].x);
			a_temp.push_back(p2[i].y * p1[i].y);
			a_temp.push_back(p2[i].y);
		}
	}
	//int t2 = clock();
	//cout << "计算内点耗时:" << t2 - t1 << endl;

	vector<double> A;
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < m; i++) {
			A.push_back(a_temp[i * n + j]);
		}
	}

	printf("m = %d,  n = %d\n", m, n);
	vector<double> H_svd(9, 0.0);
	if (m >= 8) {
		svd(m, n, A, H_svd);
		for (int i = 0; i < n; i++) {
			H_svd[i] /= H_svd[8];
			//	printf("%.2lf ", H_svd[i]);
		}
	}
		

	
	//printf("\n");

	for (int i = 0; i < 9; i++) {
	//	printf("%lf ", H[i]);
		H[i] = H_svd[i];
	}
	//printf("\n");
	double e = (double)m / 2.0 / (double)siz;
	//k = log(1.0 - 0.97) / log(1.0 - pow(e, 4)) + 2000;
	//if (k > 1000000)	k = 1000000;
	//if (k < 2000)		k = 2000;
	//cout << "迭代次数:" << k << endl;
	return (Mat_<double>(3, 3) << H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7], H[8]);
}

/*
int main()
{

	
	
    return 0;
}
*/


