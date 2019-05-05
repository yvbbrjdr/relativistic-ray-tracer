#include "../src/pathtracer.h"

using CGL::PathTracer;

struct GPUCamera{
    float widthDivDist;
    float heightDivDist;
    float c2w[9];
    float pos[3];
};

class CUDAPathTracer {

  GPUCamera* camera;

  public:
    CUDAPathTracer(PathTracer* _pathtracer);
    ~CUDAPathTracer();
    void loadCamera();
    void init();
  private:
    PathTracer* pathtracer;
}; //class CUDAPathTracer
