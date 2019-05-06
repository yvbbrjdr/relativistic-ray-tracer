#include <stdio.h>
#include <curand_kernel.h>
#include "setup.h"

#define MAX_NUM_LIGHT 20
#define MAX_NUM_BSDF 20

#define RUSSIAN_ROULETTE
#define SHADOW_RAY

#define INF_FLOAT 1e20
#define ESP_N 5e-3
#define EPS_K 1e-4

#define BLOCK_DIM 64
#define LEAF_NUMBER 4

__constant__  GPUCamera const_camera;
__constant__  GPUBSDF const_bsdfs[MAX_NUM_BSDF];
__constant__  GPULight const_lights[MAX_NUM_LIGHT];
__constant__  Parameters const_params;
__constant__  BVHParameters const_bvhparams;


__device__ void
generateRay(GPURay* ray, float x, float y)
{
    float sp[3];
    sp[0] = -(x-0.5) * const_camera.widthDivDist;
    sp[1] = -(y-0.5) * const_camera.heightDivDist;
    sp[2] = 1;
    float dir[3];
    dir[0] = -sp[0];
    dir[1] = -sp[1];
    dir[2] = -sp[2];
    float world_sp[3];
    MatrixMulVector3D(const_camera.c2w, sp, world_sp);
 }

__device__ float3
tracePixel(int x, int y)
{
   float3 s;

   int w = const_params.screenW;
   int h = const_params.screenH;

   float px = x / (float)w;
   float py = y / (float)h;

   GPURay ray;
   generateRay(&ray, px, py);
}

__global__ void
traceScene()
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= const_params.screenW * const_params.screenH) {
        return;
    }

    int x = index % const_params.screenW;
    int y = index / const_params.screenW;

    tracePixel(x, y);

    const_params.frameBuffer[3 * index] = 1.0;
    const_params.frameBuffer[3 * index + 1] = 0.5;
    const_params.frameBuffer[3 * index + 2] = 0.5;
}


__device__ float2 gridSampler(curandState *s) {
    float2 rt;
    rt.x = curand_uniform(s);
    rt.y = curand_uniform(s);
    return rt;
}

__global__ void
vectorAdd(float *A, float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        gpuAdd(A + i, B + i, C + i);
        //C[i] = A[i] + B[i];
    }
}
