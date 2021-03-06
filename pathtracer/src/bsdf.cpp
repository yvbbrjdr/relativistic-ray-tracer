#include "bsdf.h"

#include <iostream>
#include <algorithm>
#include <utility>

using std::min;
using std::max;
using std::swap;

namespace CGL {

void make_coord_space(Matrix3x3& o2w, const Vector3D& n) {
  Vector3D z = Vector3D(n.x, n.y, n.z);
  Vector3D h = z;
  if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z)) h.x = 1.0;
  else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z)) h.y = 1.0;
  else h.z = 1.0;

  z.normalize();
  Vector3D y = cross(h, z);
  y.normalize();
  Vector3D x = cross(z, y);
  x.normalize();

  o2w[0] = x;
  o2w[1] = y;
  o2w[2] = z;
}

// Mirror BSDF //

Spectrum MirrorBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum MirrorBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  reflect(wo, wi);
  *pdf = 1.0;
  return reflectance / abs_cos_theta(*wi);
}

// Microfacet BSDF //

double MicrofacetBSDF::G(const Vector3D& wo, const Vector3D& wi) {
  return 1.0 / (1.0 + Lambda(wi) + Lambda(wo));
}

double MicrofacetBSDF::D(const Vector3D& h) {
  double theta_h = getTheta(h),
         tan_theta_h = tan(theta_h),
         cos_theta_h = cos_theta(h),
         cos_theta_h2 = cos_theta_h * cos_theta_h,
         alpha2 = alpha * alpha;
  return exp(-tan_theta_h * tan_theta_h / alpha2) / (M_PI * alpha2 * cos_theta_h2 * cos_theta_h2);
}

Spectrum MicrofacetBSDF::F(const Vector3D& wi) {
  Spectrum eta2pk2 = eta * eta + k * k;
  double cos_theta_i = cos_theta(wi),
         cos_theta_i2 = cos_theta_i * cos_theta_i;
  Spectrum tetacos_theta_i = 2 * eta * cos_theta_i,
           R_s = (eta2pk2 - tetacos_theta_i + cos_theta_i2) / (eta2pk2 + tetacos_theta_i + cos_theta_i2),
           R_p = (eta2pk2 * cos_theta_i2 - tetacos_theta_i + 1) / (eta2pk2 * cos_theta_i2 + tetacos_theta_i + 1);
  return (R_s + R_p) / 2;
}

Spectrum MicrofacetBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  if (wo.z <= 0 || wi.z <= 0)
    return Spectrum();
  return F(wi) * G(wo, wi) * D((wo + wi).unit()) / (4 * wo.z * wi.z);
}

Spectrum MicrofacetBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
#if MICROFACET_HEMI == 0
  Vector2D uniform = sampler.get_sample();
  double alpha2 = alpha * alpha,
         theta_h = atan(sqrt(-alpha2 * log(1 - uniform.x))),
         phi_h = 2 * M_PI * uniform.y,
         sin_theta_h = sin(theta_h),
         cos_theta_h = cos(theta_h),
         tan_theta_h = tan(theta_h),
         p_theta = 2 * sin_theta_h * exp(-tan_theta_h * tan_theta_h / alpha2) / (alpha2 * cos_theta_h * cos_theta_h * cos_theta_h),
         p_phi = 0.5 / M_PI;
  Vector3D h(sin_theta_h * cos(phi_h), sin_theta_h * sin(phi_h), cos_theta_h);
  *wi = 2 * dot(wo, h) * h - wo;
  if (wi->z <= 0) {
    *pdf = 0;
    return Spectrum();
  }
  *pdf = p_theta * p_phi / (sin_theta_h * 4 * dot(*wi, h));
  return f(wo, *wi);
#else
  return f(wo, *wi = cosineHemisphereSampler.get_sample(pdf));
#endif // MICROFACET_HEMI
}

// Refraction BSDF //

Spectrum RefractionBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum RefractionBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  return Spectrum();
}

// Glass BSDF //

Spectrum GlassBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum GlassBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  if (refract(wo, wi, ior)) {
    double R0 = (1 - ior) / (1 + ior);
    R0 *= R0;
    double t = (1 - abs_cos_theta(*wi)),
           t2 = t * t,
           t4 = t2 * t2,
           R = R0 + (1 - R0) * t4 * t;
    if (coin_flip(R)) {
      reflect(wo, wi);
      *pdf = R;
      return R * reflectance / abs_cos_theta(*wi);
    } else {
      double eta;
      if (wo.z > 0)
        eta = 1 / ior;
      else
        eta = ior;
      *pdf = 1 - R;
      return (1 - R) * transmittance / (abs_cos_theta(*wi) * eta * eta);
    }
  } else {
    reflect(wo, wi);
    *pdf = 1.0;
    return reflectance / abs_cos_theta(*wi);
  }
}

void BSDF::reflect(const Vector3D& wo, Vector3D* wi) {
  *wi = Vector3D(-wo.x, -wo.y, wo.z);
}

bool BSDF::refract(const Vector3D& wo, Vector3D* wi, float ior) {
  double eta;
  if (wo.z > 0)
    eta = 1 / ior;
  else
    eta = ior;
  double wi_z2 = 1 - eta * eta * (1 - wo.z * wo.z);
  if (wi_z2 < 0)
    return false;
  *wi = Vector3D(-eta * wo.x, -eta * wo.y, sqrt(wi_z2));
  if (wo.z > 0)
    wi->z = -wi->z;
  return true;
}

// Emission BSDF //

Spectrum EmissionBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum EmissionBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  *pdf = 1.0 / PI;
  *wi  = sampler.get_sample(pdf);
  return Spectrum();
}

} // namespace CGL
