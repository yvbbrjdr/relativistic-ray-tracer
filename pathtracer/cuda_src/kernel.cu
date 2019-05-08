/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "kernel.h"


extern __global__ void addIntensitiesCUDA(float* a, float* b, float* dst, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = a[i] + b[i];
}

extern __global__ void substractIntensitiesCUDA(float* a, float* b, float* dst, int n)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = a[i] - b[i];
}

extern __global__ void mulIntensitiesCUDA(float* a, float* b, float* dst, int n)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = a[i] * b[i];
}


extern __global__ void divIntensitiesCUDA(float* a, float* b, float* dst, int n)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = a[i] / b[i];
}


extern __global__ void mulIntensitiesCUDA(float* a, float s, float* dst, int n)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = a[i] * s;
}


extern __global__ void divIntensitiesCUDA(float* a, float s, float* dst, int n)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = a[i] * s;
}

extern __global__ void add(int a, int b, int *c)
{
	*c = a + b;
	*c = 7;
}

void addIntensities(float* a, float *b, float *dst, int n)
{
	float* dev_a, *dev_b, *dev_dst;

	cudaMalloc((void **) &dev_a, n * sizeof(float));
	cudaMalloc((void **) &dev_b, n * sizeof(float));
	cudaMalloc((void **) &dev_dst, n * sizeof(float));

	cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dst, dst, n * sizeof(float), cudaMemcpyHostToDevice);

	addIntensitiesCUDA<<<n, 1>>>(dev_a, dev_b, dev_dst, n);

	cudaMemcpy(dst, dev_dst, n * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_dst);
}

int test(){
    int a, b, c;
    int *dev_c;

    a = 3;
    b = 4;
    cudaMalloc((void **)&dev_c, sizeof(int));
    add<<<1, 1>>>(a, b, dev_c);
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d + %d is %d\n", a, b, c);
    cudaFree(dev_c);
    return 0;
}

int main() {
	return test();
}
