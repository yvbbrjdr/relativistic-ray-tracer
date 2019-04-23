#include "blackhole.h"

#define H 0.01
const bool DEB = true;

namespace CGL { namespace StaticScene {

BlackHole global_black_hole(nullptr, Vector3D(0, 1, 0), 0.01, pow(10, -1), Vector3D(0, 0, 1).unit(), 0.25);

BlackHole::BlackHole(const SphereObject* object, const Vector3D& o, double m, double delta,
  Vector3D spin_axis, double a) :
Sphere(object, o, (1.0 + sqrt(1.0 - a * a)) * m), delta(delta), spin_axis(spin_axis),
a(a), m(m) {}

BSDF* BlackHole::get_bsdf() const {
  return nullptr;
}

double BlackHole::dr(double r_mag, double theta, double pr) {
  double ret = del(r_mag) * pr / pow(rho(r_mag, theta), 2.0);
  if (isnan(ret))
    throw "isnan in dr";
  return ret;
}

double BlackHole::dtheta(double r_mag, double theta, double ptheta) {
  double ret = ptheta / pow(rho(r_mag, theta), 2.0);
  if (isnan(ret))
    throw "isnan in dtheta";
  return ret;
}

double BlackHole::dphi(double r_mag, double theta, double b, double q) {
  //4th order finite difference approx of first derivative
  double ret = (-RDT(r_mag, theta, b + 2 * H, q) + 8 * RDT(r_mag, theta, b + H, q)
     - 8 * RDT(r_mag, theta, b - H, q) + RDT(r_mag, theta, b - 2 * H, q)) / (12 * H);
  if (isnan(ret))
    throw "isnan in dphi";
  return ret;
}

double BlackHole::predpr(double r_mag, double theta, double pr,
                          double ptheta, double b, double q) {
  double denom = 2 * pow(rho(r_mag, theta), 2.0);
  double ret = -del(r_mag) * pr * pr / denom
    - ptheta * ptheta / denom
    + RDT(r_mag, theta, b, q);
  if (isnan(ret))
   throw "isnan in predpr";
  return ret;
}

double BlackHole::dptheta(double r_mag, double theta, double pr,
                          double ptheta, double b, double q) {
  double ret = (-predpr(r_mag, theta + 2 * H, pr, ptheta, b, q)
   + 8 * predpr(r_mag, theta + H, pr, ptheta, b , q)
   -8 * predpr(r_mag, theta - H, pr, ptheta, b, q)
   + predpr(r_mag, theta - 2 * H, pr, ptheta, b, q))
   / (12 * H);
  if (isnan(ret))
    throw "isnan in dptheta";
  return ret;
}

double BlackHole::dpr(double r_mag, double theta, double pr,
                      double ptheta, double b, double q) {
  double ret = (-predpr(r_mag + 2 * H, theta, pr, ptheta, b, q)
          +8 * predpr(r_mag + H, theta, pr, ptheta, b, q)
          -8 * predpr(r_mag - H, theta, pr, ptheta, b, q)
           + predpr(r_mag - 2 * H, theta, pr, ptheta, b, q))
           / (12 * H);
  if (isnan(ret))
    throw "isnan in dpr";
  return ret;
}

double BlackHole::RDT(double r_mag, double theta, double b, double q) {
  double ret = (R(r_mag, b, q) + del(r_mag) * big_Theta(theta, b, q)) /
    (2 * del(r_mag) * pow(rho(r_mag, theta), 2.0));
  if (isnan(ret))
    throw "isnan in RDT";
  return ret;
}

double BlackHole::rho(double r_mag, double theta) {
  double ret = sqrt(r_mag * r_mag + pow(a * cos(theta), 2.0));
  if (isnan(ret))
    throw "isnan in rho";
  return ret;
}

double BlackHole::del(double r_mag) {
  double ret = r_mag * r_mag - 2.0 * r + a * a;
  if (isnan(ret))
    throw "isnan in del";
  return ret;
}

double BlackHole::sigma(double r_mag, double theta) {
  double ret = sqrt(pow(r_mag * r_mag + a * a, 2.0)
   - pow(a * sin(theta), 2.0) * del(r_mag));
  if (isnan(ret))
    throw "isnan in sigma";
  return ret;
}

double BlackHole::P(double r_mag, double b) {
  double ret = r_mag * r_mag + a * a - a * b;
  if (isnan(ret))
    throw "isnan in P";
  return ret;
}

double BlackHole::R(double r_mag, double b, double q) {
  double ret = pow(P(r_mag, b), 2.0) - del(r_mag)
    * (pow((b - a), 2.0) + q);
  if (isnan(ret))
    throw "isnan in R";
  return ret;
}

double BlackHole::big_Theta(double theta, double b, double q) {
  double ret = q - pow(cos(theta), 2.0) *
    (b * b / pow(cos(theta), 2.0) - a * a);
  if (isnan(ret))
    throw "isnan in big_Theta";
  return ret;
}

Vector4D BlackHole::evaluate(const Vector4D y, const Ray& original) {
  Vector4D k = Vector4D();
  k[0] = dr(y[0], y[1], y[2]);
  k[1] = dtheta(y[0], y[1], y[3]);
  k[2] = dpr(y[0], y[1], y[2], y[3], original.b, original.q);
  k[3] = dptheta(y[0], y[1], y[2], y[4], original.b, original.q);
  return k;
}

double BlackHole::evalPhi(const Vector4D y, const Ray& original) {
  return dphi(y[0], y[1], original.b, original.q);
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
  Vector3D x_hat = (r_vec - r_vec.norm() * spin_axis * cos(theta)).unit();
  Vector3D y_hat = cross(spin_axis, x_hat);
  Vector3D theta_hat = y_hat;
  Vector3D phi_hat = x_hat * cos(theta) - spin_axis * sin(theta);
  Vector3D r_hat = x_hat * sin(theta) + spin_axis * cos(theta);
  /*
  distances for the calculations assume unit mass
  scale back lengths after finishing calculation
  */
  //Initial conditions
  double r_mag = r_vec.norm() / m; //distance in terms of unit mass
  double pr = dot(ray.d, r_hat) / m;
  double ptheta = dot(ray.d, theta_hat) / m;
  Vector4D yi = Vector4D(r_mag, theta, pr, ptheta);
  //Runge Kutta numerically integrate
  Vector4D k1 = Vector4D();
  Vector4D k2 = Vector4D();
  Vector4D k3 = Vector4D();
  Vector4D k4 = Vector4D();
  k1 = delta * evaluate(yi, original);
  k2 = delta * evaluate(yi + k1 / 2.0, original);
  k3 = delta * evaluate(yi + k2 / 2.0, original);
  k4 = delta * evaluate(yi + k3, original);
  // if (DEB) {
  //   cout << k4 << endl;
  //   exit(0);
  // }
  yi += k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0;
  double delta_phi;
  double phik1 = delta * evalPhi(yi, original);
  double phik2 = delta * evalPhi(yi + k1 / 2.0, original);
  double phik3 = delta * evalPhi(yi + k2 / 2.0, original);
  double phik4 = delta * evalPhi(yi + k3 / 2.0, original);
  delta_phi = phik1 / 6.0 + phik2 / 3.0 + phik3 / 3.0 + phik4 / 6.0;
  //return back to normal Vector3D
  r_mag = yi[0] * m;
  theta = yi[1];
  Vector3D next = r_mag * Vector3D(
                      cos(delta_phi) * sin(theta),
                      sin(delta_phi) * sin(theta),
                      cos(theta)
                      );
  ret.d = o + next.x * x_hat + next.y * y_hat + next.z * spin_axis - ret.o;
  if (DEB) {
    cout << r_mag << endl;
  }
  ret.max_t = ret.d.norm();
  ret.d.normalize();
  return ret;
}

} // namespace StaticScene
} // namespace CGL
