#ifndef CGL_STATICSCENE_BLACKHOLE_H
#define CGL_STATICSCENE_BLACKHOLE_H

#include "sphere.h"

namespace CGL { namespace StaticScene {

class BlackHole : public Sphere {
 public:
  double delta_theta;
  BlackHole(const SphereObject* object, const Vector3D& o, double r, double delta_theta);
  BSDF* get_bsdf() const;
  double f(double u);
  Ray next_micro_ray(const Ray &ray);
};

extern BlackHole global_black_hole;

} // namespace StaticScene
} // namespace CGL

#endif //CGL_STATICSCENE_BLACKHOLE_H
