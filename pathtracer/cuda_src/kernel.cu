#include <stdio.h>

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
