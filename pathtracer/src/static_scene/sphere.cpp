#include "sphere.h"

#include <cmath>

#include  "../bsdf.h"
#include "../misc/sphere_drawing.h"

namespace CGL { namespace StaticScene {

bool Sphere::test(const Ray& r, double& t1, double& t2) const {
  Vector3D tmp = r.o - o;
  double b = 2 * dot(tmp, r.d),
         c = tmp.norm2() - r2,
         d = b * b - 4 * c;
  if (d < 0)
    return false;
  t1 = (-b - sqrt(d)) / 2;
  t2 = (-b + sqrt(d)) / 2;
  return true;
}

bool Sphere::intersect(const Ray& r) const {
  return intersect(r, nullptr);
}

bool Sphere::intersect(const Ray& r, Intersection *i) const {
  double t1, t2;
  bool ret = test(r, t1, t2);
  if (ret) {
    if (r.min_t <= t1 && t1 <= r.max_t) {
      r.max_t = t1;
      if (i) {
        i->t = t1;
        i->n = normal(r.o + r.d * t1);
        i->primitive = this;
        i->bsdf = get_bsdf();
      }
    } else if (r.min_t <= t2 && t2 <= r.max_t) {
      r.max_t = t2;
      if (i) {
        i->t = t2;
        i->n = normal(r.o + r.d * t2);
        i->primitive = this;
        i->bsdf = get_bsdf();
      }
    } else {
      return false;
    }
  }
  return ret;
}

void Sphere::draw(const Color& c, float alpha) const {
  Misc::draw_sphere_opengl(o, r, c);
}

void Sphere::drawOutline(const Color& c, float alpha) const {
    //Misc::draw_sphere_opengl(o, r, c);
}


} // namespace StaticScene
} // namespace CGL
