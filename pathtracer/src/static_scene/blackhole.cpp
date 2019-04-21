#include "blackhole.h"

#define H 0.000001

namespace CGL { namespace StaticScene {

BlackHole global_black_hole(nullptr, Vector3D(0, 1, 0), 0.1, 0.1, Vector3D(0, 0, 1).unit(), 1);

BlackHole::BlackHole(const SphereObject* object, const Vector3D& o, double m, double delta, 
  Vector3D spin_axis, double a) : 
Sphere(object, o, (1.0 + sqrt(1.0 - a * a)) * m), delta_theta(delta_theta), spin_axis(spin_axis), angular_momentum(a) {}

BSDF* BlackHole::get_bsdf() const {
  return nullptr;
}

double BlackHole::f(double u) {
  return -u + 3.0 * r * u * u / 2.0;
}

double BlackHole::dr(double r_mag, double theta, double pr) {
  return del(r_mag) * pr / pow(rho(r_mag, theta), 2.0);
}

double BlackHole::dtheta(double r_mag, double theta, double ptheta) {
  return ptheta / pow(rho(r_mag, theta), 2.0);
}

double BlackHole::dphi(double r_mag, double theta, double b, double q) {
  //4th order finite difference approx of first derivative
  return (-RDT(r_mag, theta, b + 2 * H, q) + 8 * RDT(r_mag, theta, b + H, q)
     - 8 * RDT(r_mag, theta, b - H, q) + RDT(r_mag, theta, b - 2 * H, q)) / (12 * H);
}

double BlackHole::RDT(double r_mag, double theta, double b, double q) {
  return (R(r_mag, b, q) + del(r_mag) * big_Theta(theta, b, q)) /
    (2 * del(r_mag) * pow(rho(r_mag, theta), 2.0));
}

double BlackHole::rho(double r_mag, double theta) {
  return sqrt(r_mag * r_mag + pow(a * cos(theta), 2.0));
}

double BlackHole::del(double r_mag) {
  return r_mag * r_mag - 2.0 * r + a * a;
}

double BlackHole::sigma(double r_mag, theta) {
  return sqrt(pow(r_mag * r_mag + a * a), 2.0)
   - pow(a * sin(theta)) * del(r_mag);
}

double BlackHole::P(double r_mag, double b) {
  return r_mag * r_mag + a * a
    a * b;
}

double BlackHole::R(double r_mag, double b, double q) {
  return pow(P(r_mag, b), 2.0) - del(r_mag)
    * (pow((b - a), 2.0) + q);
}

double BlackHole::big_Theta(double theta, double b, double q) {
  return q - pow(cos(theta), 2.0) * 
    (b * b / pow(cos(theta), 2.0) - a * a);
}

Ray BlackHole::next_micro_ray(const Ray &ray, const Ray &original) {
  //Get beginning ray position
  Ray ret(ray.o + ray.d * ray.max_t, Vector3D(), 0.0);
  /*
  Axial symmetry implies that axes can always be chosen s.t.
  z-x plane (black-hole center coords) contains the starting point.
   Project onto z and x to get position in black-hole center coord
   using phi = 0 convention;
  */
  Vector3D r_vec = ret.o - o;
  double theta = acos(dot(r_vec.unit(), spin_axis));
  Vector3D x_axis = (r_vec - r_vec.norm() * spin_axis * cos(theta)).unit();
  Vector3D y_axis = cross(spin_axis, x_axis);
  Vector3D theta_hat = y_hat;
  Vector3D phi_hat = x_hat * cos(theta) - spin_axis * sin(theta);
  Vector3D r_hat = x_hat * sin(theta) + spin_axis * cos(theta);
  /*
  distances for the calculations assume unit mass
  scale back lengths after finishing calculation
  */
  double r_mag = r_vec.norm() / m; //in units of black hole mass
  double pr = dot(ray.d, r_hat);
  double ptheta = dot(ray.d, theta_hat);
  //Runge Kutta numerically integrate
  
  //return back to normal Vector3D
  ret.d = o + next_x * x_axis + next_y * y_axis - ret.o;
  ret.max_t = ret.d.norm();
  ret.d.normalize();
  return ret;
}

} // namespace StaticScene
} // namespace CGL
