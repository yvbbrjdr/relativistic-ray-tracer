#include <stdio.h>
#include <curand_kernel.h>
#include "setup.h"

#define MAX_NUM_LIGHT 20
#define MAX_NUM_BSDF 20
__constant__  GPUCamera const_camera;
__constant__  GPUBSDF const_bsdfs[MAX_NUM_BSDF];
__constant__  GPULight const_lights[MAX_NUM_LIGHT];
__constant__  Parameters const_params;
__constant__  BVHParameters const_bvhparams;

#include "helper.cu"
#include "light.cu"
#include "intersect.cu"



#define RUSSIAN_ROULETTE
#define SHADOW_RAY

#define INF_FLOAT 1e20
#define ESP_N 5e-3
#define EPS_K 1e-4
#define INF_FLOAT 1e20

#define BLOCK_DIM 64
#define LEAF_NUMBER 4



__device__ bool intersect(int primIndex, GPURay& r);
__device__ bool intersect(int primIndex, GPURay& r, GPUIntersection *isect);

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
traceRay(curandState* s, GPURay* ray)
{
    GPUIntersection isect;
    isect.t = 1e10;

    bool isIntersect = false;
    for(int i = 0; i < const_params.primNum; i++)
    {
        isIntersect = intersect(i, *ray, &isect) || isIntersect;
    }

     if(!isIntersect)
        return make_float3(0.0, 0.0, 0.0);

     GPUBSDF& bsdf = const_bsdfs[isect.bsdfIndex];

     switch(bsdf.type)
    {
        case 0: case 4: return make_float3(bsdf.albedo[0], bsdf.albedo[1], bsdf.albedo[2]);
        case 1: return make_float3(0.0, 1.0, 0.0);
        case 3: return make_float3(0.0, 0.0, 1.0);
        default: break;
    }

     return make_float3(1.0, 0.0, 0.0);

 }

__device__ float3
tracePixel(curandState* s, int x, int y)
{
   float3 spec = make_float3(0.0, 0.0, 0.0);
   int w = const_params.screenW;
   int h = const_params.screenH;
   int ns_aa = const_params.ns_aa;
   for (int i = 0; i < ns_aa; i++)
   {
     float2 r = gridSampler(s);
     float px = (x + r.x) / (float)w;
     float py = (y + r.y) / (float)h;
     GPURay ray;
     generateRay(&ray, px, py);
     float3 tmpSpec = traceRay(s, &ray);
     spec.x += tmpSpec.x;
     spec.y += tmpSpec.y;
     spec.z += tmpSpec.z;
   }

   return make_float3(spec.x / ns_aa, spec.y / ns_aa, spec.z / ns_aa);
}

__global__ void
traceScene(int xStart, int yStart, int width, int height)
{
    int tIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int x = xStart + tIndex % width;
    int y = yStart + tIndex / width;
    int index = x + y * const_params.screenW;

    if (tIndex >= width * height || index > const_params.screenW * const_params.screenH) {
        return;
    }

    curandState s;
    curand_init((unsigned int)index, 0, 0, &s);

    float3 spec = tracePixel(&s, x, y);

    const_params.frameBuffer[3 * index] = spec.x;
    const_params.frameBuffer[3 * index + 1] = spec.y;
    const_params.frameBuffer[3 * index + 2] = spec.z;
}
