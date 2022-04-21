/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#pragma once
#include "cusolver_utils.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

static int flag = 0;

//svd分解解方程，A矩阵按列输入,输出A矩阵svd分解后的最后一列，注意lda，VT的大小
__host__  int svd(int m, int n, std::vector<double> &A, std::vector<double> &dst) {
	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cudaStream_t stream = NULL;

	//const int m = 3;   /* 1 <= m <= 32 */
	//const int n = 2;   /* 1 <= n <= 32 */
	const int lda = m; /* lda >= m */

	/*
	 *       | 1 2 |
	 *   A = | 4 5 |
	 *       | 2 1 |
	 */

	//const std::vector<double> A = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0 };
	std::vector<double> U(lda * m, 0);  /* m-by-m unitary matrix, left singular vectors  */
	std::vector<double> VT(lda * n, 0); /* n-by-n unitary matrix, right singular vectors */
	std::vector<double> S(n, 0);        /* numerical singular value */
	//std::vector<double> S_exact = { 7.065283497082729,
	//							   1.040081297712078 }; /* exact singular values */
	int info_gpu = 0;                                  /* host copy of error info */

	double *d_A = nullptr;
	double *d_S = nullptr;  /* singular values */
	double *d_U = nullptr;  /* left singular vectors */
	double *d_VT = nullptr; /* right singular vectors */
	double *d_W = nullptr;  /* W = S*VT */

	int *devInfo = nullptr;

	int lwork = 0; /* size of workspace */
	double *d_work = nullptr;
	double *d_rwork = nullptr;

	const double h_one = 1;
	const double h_minus_one = -1;

	//std::printf("A = (matlab base-1)\n");
	//print_matrix(m, n, A.data(), lda);
	//std::printf("=====\n");

	/* step 1: create cusolver handle, bind a stream */
	CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
	CUBLAS_CHECK(cublasCreate(&cublasH));

	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

	/* step 2: copy A to device */
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(double) * S.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(double) * U.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VT), sizeof(double) * VT.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(double) * lda * n));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

	CUDA_CHECK(
		cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));

	/* step 3: query working space of SVD */
	CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

	/* step 4: compute SVD*/
	signed char jobu = 'A';  // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	CUSOLVER_CHECK(cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, d_A, lda, d_S, d_U,
		lda, // ldu
		d_VT,
		lda, // ldvt,
		d_work, lwork, d_rwork, devInfo));

	CUDA_CHECK(
		cudaMemcpyAsync(U.data(), d_U, sizeof(double) * U.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(VT.data(), d_VT, sizeof(double) * VT.size(), cudaMemcpyDeviceToHost,
		stream));
	CUDA_CHECK(
		cudaMemcpyAsync(S.data(), d_S, sizeof(double) * S.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	//std::printf("after gesvd: info_gpu = %d\n", info_gpu);
	if (0 == info_gpu) {
	//	std::printf("gesvd converges \n");
	}
	else if (0 > info_gpu) {
		std::printf("%d-th parameter is wrong \n", -info_gpu);
		exit(1);
	}
	else {
		std::printf("WARNING: info = %d : gesvd does not converge \n", info_gpu);
	}

	dst.clear();
	dst = std::vector<double>(9);
	for (int i = 0; i < n; i++) {
		dst[i] = VT[i * lda + 8];
	}

	//std::printf("S = singular values (matlab base-1)\n");
	//print_matrix(n, 1, S.data(), n);
	//std::printf("=====\n");

	//std::printf("U = left singular vectors (matlab base-1)\n");
	//print_matrix(m, m, U.data(), lda);
	//std::printf("=====\n");

	//std::printf("VT = right singular vectors (matlab base-1)\n");
	//print_matrix(n, n, VT.data(), lda);
	//std::printf("=====\n");

	//std::printf("DST = (matlab base-1)\n");
	//for (int i = 0; i < n; i++) {
	//	printf("%.2lf ", dst[i]);
	//}
	//std::printf("=====\n");

	/* free resources */
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_U));
	CUDA_CHECK(cudaFree(d_VT));
	CUDA_CHECK(cudaFree(d_S));
	CUDA_CHECK(cudaFree(d_W));
	CUDA_CHECK(cudaFree(devInfo));
	CUDA_CHECK(cudaFree(d_work));
	CUDA_CHECK(cudaFree(d_rwork));

	CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
	CUBLAS_CHECK(cublasDestroy(cublasH));

	CUDA_CHECK(cudaStreamDestroy(stream));

	//CUDA_CHECK(cudaDeviceReset());
	return 0;
}

//svd分解解方程Ax = b，A矩阵按列主序输入，dst输出方程的解
__host__  int SolveBySVD(int m, int n, std::vector<double> &A, std::vector<double> &b, std::vector<double> &dst) {
	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cudaStream_t stream = NULL;

	const int lda = m; /* lda >= m */

	/*
	 *       | 1 2 |
	 *   A = | 4 5 |
	 *       | 2 1 |
	 */

	 //const std::vector<double> A = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0 };
	std::vector<double> U(lda * m, 0);  /* m-by-m unitary matrix, left singular vectors  */
	std::vector<double> VT(n * n, 0); /* n-by-n unitary matrix, right singular vectors */
	std::vector<double> S(n, 0);        /* numerical singular value */
	//std::vector<double> S_exact = { 7.065283497082729,
	//							   1.040081297712078 }; /* exact singular values */
	int info_gpu = 0;                                  /* host copy of error info */

	double *d_A = nullptr;
	double *d_S = nullptr;  /* singular values */
	double *d_U = nullptr;  /* left singular vectors */
	double *d_VT = nullptr; /* right singular vectors */
	double *d_W = nullptr;  /* W = S*VT */

	int *devInfo = nullptr;

	int lwork = 0; /* size of workspace */
	double *d_work = nullptr;
	double *d_rwork = nullptr;

	const double h_one = 1;
	const double h_minus_one = -1;

	//std::printf("A = (matlab base-1)\n");
	//print_matrix(m, n, A.data(), lda);
	//std::printf("=====\n");

	/* step 1: create cusolver handle, bind a stream */
	CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
	CUBLAS_CHECK(cublasCreate(&cublasH));

	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

	/* step 2: copy A to device */
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(double) * S.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(double) * U.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VT), sizeof(double) * VT.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(double) * lda * n));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

	CUDA_CHECK(
		cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));

	/* step 3: query working space of SVD */
	CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork));

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

	/* step 4: compute SVD*/
	signed char jobu = 'A';  // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	CUSOLVER_CHECK(cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, d_A, lda, d_S, d_U,
		lda, // ldu
		d_VT,
		n, // ldvt,
		d_work, lwork, d_rwork, devInfo));

	CUDA_CHECK(
		cudaMemcpyAsync(U.data(), d_U, sizeof(double) * U.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(VT.data(), d_VT, sizeof(double) * VT.size(), cudaMemcpyDeviceToHost,
		stream));
	CUDA_CHECK(
		cudaMemcpyAsync(S.data(), d_S, sizeof(double) * S.size(), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));

	//std::printf("after gesvd: info_gpu = %d\n", info_gpu);
	if (0 == info_gpu) {
		//	std::printf("gesvd converges \n");
	}
	else if (0 > info_gpu) {
		std::printf("%d-th parameter is wrong \n", -info_gpu);
		exit(1);
	}
	else {
		std::printf("WARNING: info = %d : gesvd does not converge \n", info_gpu);
	}

	//打印信息
	//print_matrix(m, m, U.data(), m);
	//std::cout << "\n";
	//for (int i = 0; i < U.size(); i++)	std::cout << " " << U[i];
	//std::cout << "\n";
	//print_matrix(n, n, VT.data(), n);
	//std::cout << "\n";
	//for (int i = 0; i < VT.size(); i++)	std::cout << " " << VT[i];
	//std::cout << "\n";
	//print_matrix(n, 1, S.data(), n);
	//std::cout << "\n";
	//for (int i = 0; i < S.size(); i++)	std::cout << " " << S[i];
	//std::cout << "\n";

	//先求Sn_inv和Un
	std::vector<double> Sn_inv(n * n), UnT(n * m);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			Sn_inv[i * n + j] = 0;
			if (i == j) {
				Sn_inv[i * n + j] = 1.0 / S[j];
			}
			//printf("i = %d, j = %d , sn_inv = %.6lf\n", i, j, Sn_inv[i * n + j]);
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			UnT[j * n + i] = U[i * m + j];
		}
	}
	//std::cout << "构造完成\n";

	//矩阵乘
	double *g_Sn_inv, *g_UnT, *g_VT, *g_b;
	CUDA_CHECK(cudaMalloc(&g_Sn_inv, sizeof(double) * n * n));
	CUDA_CHECK(cudaMalloc(&g_UnT, sizeof(double) * n * m));
	CUDA_CHECK(cudaMalloc(&g_VT, sizeof(double) * n * n));
	CUDA_CHECK(cudaMalloc(&g_b, sizeof(double) * m));
	cublasSetVector(n * n, sizeof(double), Sn_inv.data(), 1, g_Sn_inv, 1);
	cublasSetVector(n * m, sizeof(double), UnT.data(), 1, g_UnT, 1);
	cublasSetVector(n * n, sizeof(double), VT.data(), 1, g_VT, 1);
	cublasSetVector(m, sizeof(double), b.data(), 1, g_b, 1);
	cublasHandle_t handle;
	cublasCreate(&handle);
	double alpha = 1.0, beta = 0.0;
	std::vector<double> temp(n * m);
	double *g_temp;
	CUDA_CHECK(cudaMalloc(&g_temp, sizeof(double) * n * m));
	cublasSetVector(n * m, sizeof(double), temp.data(), 1, g_temp, 1);
	std::vector<double> temp2(n * m);
	//std::cout << "'设置完成\n";
	cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &alpha, g_VT, n, g_Sn_inv, n, &beta, g_temp, n);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &alpha, g_temp, n, g_UnT, n, &beta, g_temp, n);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, m, &alpha, g_temp, n, g_b, m, &beta, g_temp, n);
	cublasDestroy(handle);

	cublasGetVector(n, sizeof(double), g_temp, 1, temp.data(), 1);
	dst.clear();
	for (int i = 0; i < n; i++) {
		dst.push_back(temp[i]);
	}
	CUDA_CHECK(cudaFree(g_Sn_inv));
	CUDA_CHECK(cudaFree(g_UnT));
	CUDA_CHECK(cudaFree(g_b));
	CUDA_CHECK(cudaFree(g_temp));
	CUDA_CHECK(cudaFree(g_VT));

	/* free resources */
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_U));
	CUDA_CHECK(cudaFree(d_VT));
	CUDA_CHECK(cudaFree(d_S));
	CUDA_CHECK(cudaFree(d_W));
	CUDA_CHECK(cudaFree(devInfo));
	CUDA_CHECK(cudaFree(d_work));
	CUDA_CHECK(cudaFree(d_rwork));

	CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
	CUBLAS_CHECK(cublasDestroy(cublasH));

	CUDA_CHECK(cudaStreamDestroy(stream));

	return 0;
}

/*
int main(int argc, char *argv[]) {
	int m = 50, n = 9;
	std::vector<double> a;
	for (int i = 0; i < n * m; i++) {
		a.push_back(1.0 * i);
	}
	std::vector<double> b;
	svd(m, n, a, b);
	svd(m, n, a, b);
	svd(m, n, a, b);

	for (int i = 0; i < 9; i++) {
		printf("%lf ", b[i]);
	}
}
*/

