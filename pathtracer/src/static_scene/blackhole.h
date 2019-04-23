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
  BlackHole(const SphereObject* object, const Vector3D& o, double m, double delta, const Vector3D spin_axis, double a);
  BSDF* get_bsdf() const;
  Ray next_micro_ray(const Ray &ray, const Ray &original);
  double dr(double r_mag, double theta, double pr);
  double dtheta(double r_mag, double theta, double ptheta);
  double dphi(double r_mag, double theta, double b, double q);
  double predpr(double r_mag, double theta, double pr, double ptheta, double b, double q);
  double dpr(double r_mag, double theta, double pr, double ptheta, double b, double q);
  double dptheta(double r_mag, double theta, double pr, double ptheta, double b, double q);
  double RDT(double r_mag, double theta, double b, double q);
  double rho(double r_mag, double theta);
  double del(double r_mag);
  double sigma(double r_mag, double theta);
  double P(double r_mag, double b);
  double R(double r_mag, double b, double q);
  double big_Theta(double theta, double b, double q);
  Vector4D evaluate(const Vector4D y, const Ray& original);
  double evalPhi(const Vector4D y, const Ray& original);
};

extern BlackHole global_black_hole;

} // namespace StaticScene
} // namespace CGL

#endif //CGL_STATICSCENE_BLACKHOLE_H
