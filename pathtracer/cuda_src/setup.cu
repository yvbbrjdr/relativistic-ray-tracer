#include <cuda_runtime.h>
#include <iostream>
#include "helper.cu"
#include "setup.h"
#include <map>

using CGL::PathTracer;
using CGL::Camera;
using namespace std;

struct Parameters
{
    int screenW;
    int screenH;
    int max_ray_depth; ///< maximum allowed ray depth (applies to all rays)
    int ns_aa;         ///< number of camera rays in one pixel (along one axis)
    int ns_area_light; ///< number samples per area light source
    int lightNum;
    int primNum;

    int* types;
    int* bsdfIndexes;
    float* positions;
    float* normals;
    float4* woopPositions;
    float3* camOffset;

    float* frameBuffer;

    int* BVHPrimMap;
    GPUBVHNode* BVHRoot;
};

struct BVHParameters
{
    float sceneMin[3];
    float sceneExtent[3];
    int numObjects;
    GPUBVHNode *leafNodes;
    GPUBVHNode *internalNodes;
    unsigned int*sortedMortonCodes;
    int *sortedObjectIDs;
    int *types;
    float *positions;

};

#define TILE_DIM 1

#include "kernel.cu"
#include <map>

float3* gpu_camOffset;
float4* gpu_woopPositions;

CUDAPathTracer::CUDAPathTracer(PathTracer* _pathtracer)
{
  pathtracer = _pathtracer;
}

CUDAPathTracer::~CUDAPathTracer()
{
  cudaFree(gpu_positions);
  cudaFree(gpu_bsdfIndexes);
  cudaFree(gpu_positions);
  cudaFree(gpu_normals);
  cudaFree(gpu_woopPositions);
  cudaFree(frameBuffer);
  cudaFree(BVHPrimMap);
}

void CUDAPathTracer::init()
{
  cudaDeviceReset();
  loadCamera();
  loadPrimitives();
  loadLights();
  loadParameters();
  cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 24);
}

void CUDAPathTracer::loadCamera()
{
  //printf("load camera\n");
  //printf("camera: %p\n", pathTracer->camera);
  GPUCamera tmpCam;
  Camera* cam = pathtracer->camera;
  tmpCam.widthDivDist = cam->screenW / cam->screenDist;
  tmpCam.heightDivDist = cam->screenH / cam->screenDist;
  //printf("after loading camera\n");
  for (int i = 0; i < 9; i++) {
      tmpCam.c2w[i] = cam->c2w(i / 3, i % 3);
  }

  for (int i = 0; i < 3; i++) {
      tmpCam.pos[i] = cam->pos[i];
  }

  cudaError_t err = cudaSuccess;
  //cudaMalloc((void**)&gpu_camera,sizeof(GPUCamera));
  err = cudaMemcpyToSymbol(const_camera, &tmpCam,sizeof(GPUCamera));

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

void CUDAPathTracer::loadPrimitives()
{
    vector<Primitive *>& primitives = pathtracer->primitives;
    int N = primitives.size();
    int types[N];
    int bsdfs[N];
    float *positions = new float[9 * N];
    float *normals = new float[9 * N];
    float4* woopPositions = new float4[3 * N];

    primNum = N;
    map<BSDF*, int> BSDFMap;
    for (int i = 0; i < N; i++) {

        primMap[primitives[i]] = i;
        types[i] = primitives[i]->getType();
        BSDF* bsdf  = primitives[i]->get_bsdf();

        if (BSDFMap.find(bsdf) == BSDFMap.end()) {
            int index = BSDFMap.size();
            BSDFMap[bsdf] = index;
            bsdfs[i] = index;
        }
        else{
            bsdfs[i] = BSDFMap[bsdf];
        }


        if (types[i] == 0) {
            Vector3D o = ((Sphere*)primitives[i])->o;
            positions[9 * i] = o[0];
            positions[9 * i + 1] = o[1];
            positions[9 * i + 2] = o[2];
            positions[9 * i + 3] = ((Sphere*)primitives[i])->r;
        }
        else{
            const Mesh* mesh = ((Triangle*)primitives[i])->mesh;
            int v1 = ((Triangle*)primitives[i])->v1;
            int v2 = ((Triangle*)primitives[i])->v2;
            int v3 = ((Triangle*)primitives[i])->v3;

            positions[9 * i] = mesh->positions[v1][0];
            positions[9 * i + 1] = mesh->positions[v1][1];
            positions[9 * i + 2] = mesh->positions[v1][2];
            normals[9 * i] = mesh->normals[v1][0];
            normals[9 * i + 1] = mesh->normals[v1][1];
            normals[9 * i + 2] = mesh->normals[v1][2];

            positions[9 * i + 3] = mesh->positions[v2][0] - positions[9 * i];
            positions[9 * i + 4] = mesh->positions[v2][1] - positions[9 * i + 1];
            positions[9 * i + 5] = mesh->positions[v2][2] - positions[9 * i + 2];
            normals[9 * i + 3] = mesh->normals[v2][0];
            normals[9 * i + 4] = mesh->normals[v2][1];
            normals[9 * i + 5] = mesh->normals[v2][2];

            positions[9 * i + 6] = mesh->positions[v3][0] - positions[9 * i];
            positions[9 * i + 7] = mesh->positions[v3][1] - positions[9 * i + 1];
            positions[9 * i + 8] = mesh->positions[v3][2] - positions[9 * i + 2];
            normals[9 * i + 6] = mesh->normals[v3][0];
            normals[9 * i + 7] = mesh->normals[v3][1];
            normals[9 * i + 8] = mesh->normals[v3][2];

            Matrix4x4 mtx;
            Vector3D c0(positions[9 * i + 3], positions[9 * i + 4], positions[9 * i + 5]);
            Vector3D c1(positions[9 * i + 6], positions[9 * i + 7], positions[9 * i + 8]);
            Vector3D c2 = cross(c0, c1);
            Vector3D c3(positions[9 * i], positions[9 * i + 1], positions[9 * i + 2]);

            mtx[0] = Vector4D(c0);
            mtx[1] = Vector4D(c1);
            mtx[2] = Vector4D(c2);
            mtx[3] = Vector4D(c3, 1.0);

            mtx = mtx.inv();

            woopPositions[3 * i] = make_float4(mtx(2,0), mtx(2,1), mtx(2,2), -mtx(2,3));
            woopPositions[3 * i + 1] = make_float4(mtx(0,0), mtx(0,1), mtx(0,2), mtx(0,3));
            woopPositions[3 * i + 2] = make_float4(mtx(1,0), mtx(1,1), mtx(1,2), mtx(1,3));
        }
    }
    GPUBSDF BSDFArray[BSDFMap.size()];

    for (auto itr = BSDFMap.begin(); itr != BSDFMap.end(); itr++) {
        GPUBSDF& gpu_bsdf = BSDFArray[itr->second];
        BSDF* bsdf = itr->first;
        gpu_bsdf.type = bsdf->getType();

        if (gpu_bsdf.type == 0) {
            Spectrum& albedo = ((DiffuseBSDF*)bsdf)->reflectance;
            gpu_bsdf.albedo[0] = albedo.r;
            gpu_bsdf.albedo[1] = albedo.g;
            gpu_bsdf.albedo[2] = albedo.b;
        }
        else if(gpu_bsdf.type == 1){
            Spectrum& reflectance = ((MirrorBSDF*)bsdf)->reflectance;
            gpu_bsdf.reflectance[0] = reflectance.r;
            gpu_bsdf.reflectance[1] = reflectance.g;
            gpu_bsdf.reflectance[2] = reflectance.b;
        }
        else if(gpu_bsdf.type == 2){
            Spectrum& transmittance = ((RefractionBSDF*)bsdf)->transmittance;
            gpu_bsdf.transmittance[0] = transmittance.r;
            gpu_bsdf.transmittance[1] = transmittance.g;
            gpu_bsdf.transmittance[2] = transmittance.b;
            gpu_bsdf.ior = ((RefractionBSDF*)bsdf)->ior;
        }
        else if(gpu_bsdf.type == 3){
            Spectrum& reflectance = ((GlassBSDF*)bsdf)->reflectance;
            gpu_bsdf.reflectance[0] = reflectance.r;
            gpu_bsdf.reflectance[1] = reflectance.g;
            gpu_bsdf.reflectance[2] = reflectance.b;
            Spectrum& transmittance = ((GlassBSDF*)bsdf)->transmittance;
            gpu_bsdf.transmittance[0] = transmittance.r;
            gpu_bsdf.transmittance[1] = transmittance.g;
            gpu_bsdf.transmittance[2] = transmittance.b;
            gpu_bsdf.ior = ((GlassBSDF*)bsdf)->ior;
        }
        else if(gpu_bsdf.type == 4){
            Spectrum& albedo = ((EmissionBSDF*)bsdf)->radiance;
            gpu_bsdf.albedo[0] = albedo.r;
            gpu_bsdf.albedo[1] = albedo.g;
            gpu_bsdf.albedo[2] = albedo.b;

        }
    }

    cudaMalloc((void**)&gpu_types, N * sizeof(int));
    cudaMalloc((void**)&gpu_bsdfIndexes, N * sizeof(int));
    cudaMalloc((void**)&gpu_positions, 9 * N * sizeof(float));
    cudaMalloc((void**)&gpu_normals, 9 * N * sizeof(float));
    cudaMalloc((void**)&gpu_woopPositions, 3 * N * sizeof(float4));

    cudaMemcpy(gpu_types, types, N * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_bsdfIndexes, bsdfs, N * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_positions, positions, 9 * N * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_normals, normals, 9 * N * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_woopPositions, woopPositions, 3 * N * sizeof(float4), cudaMemcpyHostToDevice);

    //cudaMalloc((void**)&gpu_bsdfs, BSDFMap.size() * sizeof(GPUBSDF));
    delete [] positions;
    delete [] normals;
    delete [] woopPositions;

    cudaError_t err = cudaSuccess;

    err = cudaMemcpyToSymbol(const_bsdfs, BSDFArray, BSDFMap.size() * sizeof(GPUBSDF));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// load Parameters
void CUDAPathTracer::loadParameters() {
    Parameters tmpParams;
    tmpParams.screenW = pathtracer->frameBuffer.w;
    tmpParams.screenH = pathtracer->frameBuffer.h;
    tmpParams.max_ray_depth = pathtracer->max_ray_depth;
    tmpParams.ns_aa = pathtracer->ns_aa;
    tmpParams.ns_area_light = pathtracer->ns_area_light;
    tmpParams.lightNum = pathtracer->scene->lights.size();
    tmpParams.types = gpu_types;
    tmpParams.bsdfIndexes = gpu_bsdfIndexes;
    tmpParams.positions = gpu_positions;
    tmpParams.normals = gpu_normals;
    tmpParams.primNum = primNum;
    tmpParams.frameBuffer = frameBuffer;
    tmpParams.BVHPrimMap = BVHPrimMap;
    tmpParams.BVHRoot = BVHRoot;
    tmpParams.woopPositions = gpu_woopPositions;

    cudaMalloc((void**)gpu_camOffset, sizeof(float3));
    tmpParams.camOffset = gpu_camOffset;

    cout << "primNum:" << primNum << endl;
    cudaError_t err = cudaSuccess;

    err = cudaMemcpyToSymbol(const_params, &tmpParams, sizeof(Parameters));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //Parameters rtParms;
    //cudaMemcpy(&rtParms, parms, sizeof(Parameters), cudaMemcpyDeviceToHost);
    //printf("screenW: %d, screenH: %d, max_ray_depth: %d, ns_aa: %d, ns_area_light: %d, lightNum: %d\n", rtParms.screenW, rtParms.screenH, rtParms.max_ray_depth, rtParms.ns_aa, rtParms.ns_area_light, rtParms.lightNum);
}

void CUDAPathTracer::toGPULight(SceneLight* l, GPULight *gpuLight) {
  gpuLight->type = l->getType();
  switch(l->getType()) {
    case 0: // DirectionalLight
    {
      DirectionalLight* light = (DirectionalLight*) l;
      for (int i = 0; i < 3; ++i) {
        gpuLight->radiance[i] = light->radiance[i];
        gpuLight->dirToLight[i] = light->dirToLight[i];
      }
    }
    break;

    case 1: // InfiniteHemisphereLight
    {
      InfiniteHemisphereLight* light = (InfiniteHemisphereLight*) l;
      for (int i = 0; i < 3; ++i) {
          gpuLight->radiance[i] = light->radiance[i];
      }
    }
    break;

    case 2: // PointLight
    {
      PointLight* light = (PointLight*) l;
      for (int i = 0; i < 3; ++i) {
        gpuLight->radiance[i] = light->radiance[i];
        gpuLight->position[i] = light->position[i];
      }
    }
    break;


    case 3: // AreaLight
    {
      AreaLight* light = (AreaLight*) l;
      for (int i = 0; i < 3; ++i) {
        gpuLight->radiance[i] = light->radiance[i];
        gpuLight->position[i] = light->position[i];
        gpuLight->direction[i] = light->direction[i];
        gpuLight->dim_x[i] = light->dim_x[i];
        gpuLight->dim_y[i] = light->dim_y[i];
        gpuLight->area = light->area;
      }
    }
    break;

    default:
    break;
  }
}

void CUDAPathTracer::loadLights() {
  int tmpLightNum = pathtracer->scene->lights.size();

  GPULight tmpLights[tmpLightNum];

  for (int i = 0; i < tmpLightNum; ++i) {
      //displayLight(pathTracer->scene->lights[i]);
      toGPULight(pathtracer->scene->lights[i], tmpLights + i);
  }
  //cudaMalloc((void**)&gpu_lights, sizeof(GPULight) * tmpLightNum);


  cudaError_t err = cudaSuccess;

  err = cudaMemcpyToSymbol(const_lights, tmpLights, sizeof(GPULight) * tmpLightNum);


  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
}

__global__ void
printInfo()
{
    for (int i = 0; i < 8; i++) {
        if (const_bsdfs[i].type == 0) {
            printf("0: %lf %lf %lf\n", const_bsdfs[i].albedo[0], const_bsdfs[i].albedo[1], const_bsdfs[i].albedo[2] );
        }
        else if (const_bsdfs[i].type == 1) {
            printf("1: %lf %lf %lf\n", const_bsdfs[i].reflectance[0], const_bsdfs[i].reflectance[1], const_bsdfs[i].reflectance[2] );
        }
        else if (const_bsdfs[i].type == 2) {
            //cout << "2" << endl;
        }
        else if (const_bsdfs[i].type == 3) {
            printf("3: %lf %lf %lf\n", const_bsdfs[i].reflectance[0], const_bsdfs[i].reflectance[1], const_bsdfs[i].reflectance[2] );
            printf("3: %lf %lf %lf\n", const_bsdfs[i].transmittance[0], const_bsdfs[i].transmittance[1], const_bsdfs[i].transmittance[2] );
        }
        else {
            printf("4: %lf %lf %lf\n", const_bsdfs[i].albedo[0], const_bsdfs[i].albedo[1], const_bsdfs[i].albedo[2] );
        }
    }


     printf("%lf %lf %lf\n", const_camera.pos[0], const_camera.pos[1], const_camera.pos[2] );

 }

extern __global__ void vectorAdd(float *A, float *B, float *C, int numElements);

extern void test() {
  printf("hello\n");
}

int main() {
  test();
  return 0;
}
