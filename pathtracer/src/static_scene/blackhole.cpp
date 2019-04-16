#include "blackhole.h"

namespace CGL { namespace StaticScene {

BlackHole global_black_hole(nullptr, Vector3D(0, 1, 0), 0.1, 0.1);

BlackHole::BlackHole(const SphereObject* object, const Vector3D& o, double r, double delta_theta) : Sphere(object, o, r), delta_theta(delta_theta) {}

BSDF* BlackHole::get_bsdf() const {
  return nullptr;
}

double BlackHole::f(double u) {
  return -u + 3.0 * r * u * u / 2.0;
}

Ray BlackHole::next_micro_ray(const Ray &ray) {
  Ray ret(ray.o + ray.d * ray.max_t, Vector3D(), 0.0);
  Vector3D x_axis = ret.o - o;
  double d = x_axis.norm();
  x_axis.normalize();
  double u = 1 / d;
  double dx = dot(ray.d, x_axis);
  Vector3D y_axis = ray.d - dx * x_axis;
  double dy = y_axis.norm();
  y_axis.normalize();
  double up = -u * dx / dy;
  double f1 = f(u);
  double f2 = f(u + up * delta_theta / 2.0);
  double f3 = f(u + up * delta_theta / 2.0 + f1 * delta_theta * delta_theta / 4.0);
  double f4 = f(u + up * delta_theta + f2 * delta_theta / 2.0);
  u += up * delta_theta + (f1 + f2 + f3) * delta_theta * delta_theta / 6.0;
  d = 1 / u;
  double next_x = d * cos(delta_theta);
  double next_y = d * sin(delta_theta);
  ret.d = o + next_x * x_axis + next_y * y_axis - ret.o;
  ret.max_t = ret.d.norm();
  ret.d.normalize();
  return ret;
}

} // namespace StaticScene
} // namespace CGL
