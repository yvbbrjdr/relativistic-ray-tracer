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
    ray->depth = 0;
    ray->min_t = 0;
    ray->max_t = 1e10;
    float sp[3];
    float dir[3];
    initVector3D(-(x - 0.5) * const_camera.widthDivDist,
    -(y - 0.5) * const_camera.heightDivDist, 1, sp);
    negVector3D(sp, dir);
    MatrixMulVector3D(const_camera.c2w, sp, ray->o);
    addVector3D(const_camera.pos, ray->o);
    MatrixMulVector3D(const_camera.c2w, dir, ray->d);
    normalize3D(ray->d);
}

__device__ float3
tracePixel(int x, int y)
{
   float3 spec = make_float3(1.0, 0.0, 0.0);
   int w = const_params.screenW;
   int h = const_params.screenH;
   for (int i = 0; i < 5; i++)
   {
     float px = x / (float)w;
     float py = y / (float)h;
     GPURay ray;
     generateRay(&ray, px, py);
   }

   return spec;
}

__global__ void
traceScene()
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= const_params.screenW * const_params.screenH) {
        return;
    }

    curandState s;
    curand_init((unsigned int)index, 0, 0, &s);

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

__device__ bool sphereTest(int primIndex, GPURay& ray, double& t1, double& t2) {
    float* primitive = const_params.positions + 9 * primIndex;
    float* o = primitive;
    float r = primitive[3];
    float r2 = r * r;

     float m[3];
    subVector3D(o, ray.o, m);
    double b = VectorDot3D(m, ray.d);
    double c = VectorDot3D(m, m) - r2;
    double delta = b * b - c;
    if (delta < 0) {
        return false;
    }

     t1 = b - sqrt(delta);
    t2 = b + sqrt(delta);

     if (t1 >= ray.max_t || t2 <= ray.min_t) {
        return false;
    }

     return true;
}

 __device__ bool sphereIntersect(int primIndex, GPURay& r) {
    double tmp;
    return sphereTest(primIndex, r, tmp, tmp);
}

 __device__ bool sphereIntersect(int primIndex, GPURay& r, GPUIntersection *isect) {
    double t1;
    double t2;
    bool res = sphereTest(primIndex, r, t1, t2);
    if (!res) {
        return false;
    }
    isect->bsdfIndex = const_params.bsdfIndexes[primIndex];
    isect->pIndex = primIndex;

     float* primitive = const_params.positions + 9 * primIndex;
    float* o = primitive;
    double t = t1;
    if (t1 <= r.min_t) {
        t = t2;
    }
    float n[3];
    float tmp[3];
    for (int i = 0; i < 3; ++i)
    {
        tmp[i] = r.d[i] * t;
    }
    addVector3D(r.o, tmp);
    subVector3D(tmp, o, n);
    normalize3D(n);
    readVector3D(n, isect->n);
    isect->t = t;
    r.max_t = t;

     return true;
}

__device__ bool triangleIntersect(int primIndex, GPURay& r, GPUIntersection *isect) {

     float* primitive = const_params.positions + 9 * primIndex;
    float* normals = const_params.normals + 9 * primIndex;

     float* v1 = primitive;
    float* v2 = primitive + 3;
    float* v3 = primitive + 6;

     float e1[3], e2[3], s[3];
    subVector3D(v2, v1, e1);
    subVector3D(v3, v1, e2);
    subVector3D(r.o, v1, s);

     float tmp[3];
    VectorCross3D(e1, r.d, tmp);
    double f = VectorDot3D(tmp, e2);
    if (f == 0) {
        return false;
    }

     VectorCross3D(s, r.d, tmp);
    double u = VectorDot3D(tmp, e2) / f;
    VectorCross3D(e1, r.d, tmp);
    double v = VectorDot3D(tmp, s) / f;
    VectorCross3D(e1, s, tmp);
    double t = - VectorDot3D(tmp, e2) / f;

     if (!(u >= 0 && v >= 0 && u+v <= 1 && t > r.min_t && t < r.max_t && t < isect->t)) {
        return false;
    }

     r.max_t = t;

     isect->bsdfIndex = const_params.bsdfIndexes[primIndex];
    isect->t = t;
    isect->pIndex = primIndex;

     float *n1 = normals;
    float *n2 = normals + 3;
    float *n3 = normals + 6;

     float n[3];
    for (int i = 0; i < 3; ++i)
    {
        n[i] = (1 - u - v) * n1[i] + u * n2[i] + v * n3[i];
    }
    if (VectorDot3D(r.d, n) > 0)
    {
        negVector3D(n, n);
    }
    readVector3D(n, isect->n);

     return true;
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
