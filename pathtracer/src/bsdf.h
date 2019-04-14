#ifndef CGL_STATICSCENE_BSDF_H
#define CGL_STATICSCENE_BSDF_H

#define MICROFACET_HEMI 0

#include "CGL/CGL.h"
#include "CGL/spectrum.h"
#include "CGL/vector3D.h"
#include "CGL/matrix3x3.h"

#include "sampler.h"
#include "image.h"

#include <algorithm>

namespace CGL {

// Helper math functions. Assume all vectors are in unit hemisphere //

inline double clamp (double n, double lower, double upper) {
  return std::max(lower, std::min(n, upper));
}

inline double cos_theta(const Vector3D& w) {
  return w.z;
}

inline double abs_cos_theta(const Vector3D& w) {
  return fabs(w.z);
}

inline double sin_theta2(const Vector3D& w) {
  return fmax(0.0, 1.0 - cos_theta(w) * cos_theta(w));
}

inline double sin_theta(const Vector3D& w) {
  return sqrt(sin_theta2(w));
}

inline double cos_phi(const Vector3D& w) {
  double sinTheta = sin_theta(w);
  if (sinTheta == 0.0) return 1.0;
  return clamp(w.x / sinTheta, -1.0, 1.0);
}

inline double sin_phi(const Vector3D& w) {
  double sinTheta = sin_theta(w);
  if (sinTheta) return 0.0;
  return clamp(w.y / sinTheta, -1.0, 1.0);
}

void make_coord_space(Matrix3x3& o2w, const Vector3D& n);

/**
 * Interface for BSDFs.
 */
class BSDF {
 public:

  /**
   * Evaluate BSDF.
   * Given incident light direction wi and outgoing light direction wo. Note
   * that both wi and wo are defined in the local coordinate system at the
   * point of intersection.
   * \param wo outgoing light direction in local space of point of intersection
   * \param wi incident light direction in local space of point of intersection
   * \return reflectance in the given incident/outgoing directions
   */
  virtual Spectrum f (const Vector3D& wo, const Vector3D& wi) = 0;

  /**
   * Evaluate BSDF.
   * Given the outgoing light direction wo, compute the incident light
   * direction and store it in wi. Store the pdf of the outgoing light in pdf.
   * Again, note that wo and wi should both be defined in the local coordinate
   * system at the point of intersection.
   * \param wo outgoing light direction in local space of point of intersection
   * \param wi address to store incident light direction
   * \param pdf address to store the pdf of the output incident direction
   * \return reflectance in the output incident and given outgoing directions
   */
  virtual Spectrum sample_f (const Vector3D& wo, Vector3D* wi, float* pdf) = 0;

  /**
   * Get the emission value of the surface material. For non-emitting surfaces
   * this would be a zero energy spectrum.
   * \return emission spectrum of the surface material
   */
  virtual Spectrum get_emission () const = 0;

  /**
   * If the BSDF is a delta distribution. Materials that are perfectly specular,
   * (e.g. water, glass, mirror) only scatter light from a single incident angle
   * to a single outgoing angle. These BSDFs are best described with alpha
   * distributions that are zero except for the single direction where light is
   * scattered.
   */
  virtual bool is_delta() const = 0;

  /**
   * Reflection helper
   */
  virtual void reflect(const Vector3D& wo, Vector3D* wi);

  /**
   * Refraction helper
   */
  virtual bool refract(const Vector3D& wo, Vector3D* wi, float ior);

  const HDRImageBuffer* reflectanceMap;
  const HDRImageBuffer* normalMap;

}; // class BSDF

/**
 * Diffuse BSDF.
 */
class DiffuseBSDF : public BSDF {
 public:

  DiffuseBSDF(const Spectrum& a) : reflectance(a) { }

  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  Spectrum sample_f(const Vector3D& wo, Vector3D* wi, float* pdf);
  Spectrum get_emission() const { return Spectrum(); }
  bool is_delta() const { return false; }

private:

  Spectrum reflectance;
  CosineWeightedHemisphereSampler3D sampler;

}; // class DiffuseBSDF

/**
 * Mirror BSDF
 */
class MirrorBSDF : public BSDF {
 public:

  MirrorBSDF(const Spectrum& reflectance) : reflectance(reflectance) { }

  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  Spectrum sample_f(const Vector3D& wo, Vector3D* wi, float* pdf);
  Spectrum get_emission() const { return Spectrum(); }
  bool is_delta() const { return true; }

private:

  float roughness;
  Spectrum reflectance;

}; // class MirrorBSDF*/

/**
 * Microfacet BSDF.
 */

class MicrofacetBSDF : public BSDF {
 public:

  MicrofacetBSDF(const Spectrum& eta, const Spectrum& k, float alpha)
    : eta(eta), k(k), alpha(alpha){ }

  double getTheta(const Vector3D& w) {
      return acos(clamp(w.z, -1.0 + 1e-5, 1.0 - 1e-5));
  }

  double Lambda (const Vector3D& w) {
      double theta = getTheta(w);
      double a = 1.0 / (alpha * tan(theta));
      return 0.5 * (erf(a) - 1.0 + exp(-a * a) / (a * PI));
  }

  Spectrum F(const Vector3D& wi);

  double G(const Vector3D& wo, const Vector3D& wi);

  double D(const Vector3D& h);

  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  Spectrum sample_f(const Vector3D& wo, Vector3D* wi, float* pdf);
  Spectrum get_emission() const { return Spectrum(); }
  bool is_delta() const { return false; }

private:
  Spectrum eta, k;
  float alpha;
  UniformGridSampler2D sampler;
  CosineWeightedHemisphereSampler3D cosineHemisphereSampler;
}; // class MicrofacetBSDF

/**
 * Refraction BSDF.
 */
class RefractionBSDF : public BSDF {
 public:

  RefractionBSDF(const Spectrum& transmittance, float roughness, float ior)
    : transmittance(transmittance), roughness(roughness), ior(ior) { }

  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  Spectrum sample_f(const Vector3D& wo, Vector3D* wi, float* pdf);
  Spectrum get_emission() const { return Spectrum(); }
  bool is_delta() const { return true; }

 private:

  float ior;
  float roughness;
  Spectrum transmittance;

}; // class RefractionBSDF

/**
 * Glass BSDF.
 */
class GlassBSDF : public BSDF {
 public:

  GlassBSDF(const Spectrum& transmittance, const Spectrum& reflectance,
            float roughness, float ior) :
    transmittance(transmittance), reflectance(reflectance),
    roughness(roughness), ior(ior) { }

  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  Spectrum sample_f(const Vector3D& wo, Vector3D* wi, float* pdf);
  Spectrum get_emission() const { return Spectrum(); }
  bool is_delta() const { return true; }

 private:

  float ior;
  float roughness;
  Spectrum reflectance;
  Spectrum transmittance;

}; // class GlassBSDF

/**
 * Emission BSDF.
 */
class EmissionBSDF : public BSDF {
 public:

  EmissionBSDF(const Spectrum& radiance) : radiance(radiance) { }

  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  Spectrum sample_f(const Vector3D& wo, Vector3D* wi, float* pdf);
  Spectrum get_emission() const { return radiance; }
  bool is_delta() const { return false; }

 private:

  Spectrum radiance;
  CosineWeightedHemisphereSampler3D sampler;

}; // class EmissionBSDF

}  // namespace CGL

#endif  // CGL_STATICSCENE_BSDF_H
