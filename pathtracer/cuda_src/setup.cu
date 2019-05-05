#include <cuda_runtime.h>
#include <iostream>
#include "helper.cu"
#include "setup.h"

using CGL::PathTracer;
using CGL::Camera;
using namespace std;

CUDAPathTracer::CUDAPathTracer(PathTracer* _pathtracer)
{
  pathtracer = _pathtracer;
}

CUDAPathTracer::~CUDAPathTracer()
{
  return;
}

void CUDAPathTracer::init()
{
  loadCamera();
}

void CUDAPathTracer::loadCamera()
{
  GPUCamera tmpCam;
  Camera* cam  = pathtracer->camera;
  tmpCam.widthDivDist = cam->screenW / cam->screenDist;
  tmpCam.heightDivDist = cam->screenH / cam->screenDist;
  for (int i = 0; i < 9; i++) {
    tmpCam.c2w[i] = (cam->c2w)(i / 3, i % 3);
  }

   for (int i = 0; i < 3; i++) {
    tmpCam.pos[i] = cam->pos[i];
  }

  cudaMalloc((void**)&camera,sizeof(GPUCamera));
  cudaMemcpy(camera, &tmpCam,sizeof(GPUCamera),cudaMemcpyHostToDevice);

  cudaFree(camera);
}

extern __global__ void vectorAdd(float *A, float *B, float *C, int numElements);

extern void test() {
  printf("hello\n");
}

int main() {
  test();
  return 0;
}
