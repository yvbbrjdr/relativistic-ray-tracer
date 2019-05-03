#include "pathtracer.h"

__kernel void raytrace_kernel(__global Spectrum* output, int width, int height, int x, int y) {

  const int work_item_id = get_global_id(0);
  int x_coord =  work_item_id %  width + x;
  int y_coord =  work_item_id /  width + y;
  output[work_item_id] = raytrace_pixel(x_coord, y_coord);
}
