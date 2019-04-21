#ifndef CGL_STATICSCENE_BLACKHOLE_H
#define CGL_STATICSCENE_BLACKHOLE_H

#include "sphere.h"

namespace CGL { namespace StaticScene {

class BlackHole : public Sphere {
 public:
  double delta;
  Vector3D spin_axis;
  double a;
  double m;
  BlackHole(const SphereObject* object, const Vector3D& o, double m, double delta, Vector3D spin_axis, double a);
  BSDF* get_bsdf() const;
  double f(double u);
  Ray next_micro_ray(const Ray &ray, const Ray &original);
};

extern BlackHole global_black_hole;

} // namespace StaticScene
} // namespace CGL

#endif //CGL_STATICSCENE_BLACKHOLE_H
