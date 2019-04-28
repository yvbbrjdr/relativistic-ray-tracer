//
// TODO: Copy over 3-1 code after turning on BUILD_3-1 flag
//

#include "part1_code.h"
#include <time.h>
#include "lightspectrum.h"
#include <omp.h>
#include "static_scene/blackhole.h"

using namespace CGL::StaticScene;

using std::min;
using std::max;

namespace CGL {

  LightSpectrum PathTracer::estimate_direct_lighting_hemisphere(const Ray& r, const Intersection& isect) {
    Matrix3x3 o2w;
    make_coord_space(o2w, isect.n);
    Matrix3x3 w2o = o2w.T();
    const Vector3D& hit_p = isect.hit_p;
    const Vector3D& w_out = w2o * isect.w_out;
    int num_samples = scene->lights.size() * ns_area_light;
    LightSpectrum L_out = LightSpectrum();
    for (int i = 0; i < num_samples; ++i) {
      Vector3D w_in = hemisphereSampler->get_sample();
      Vector3D wi_world = o2w * w_in;
      Intersection isect2;
      if (bvh->intersect(Ray(hit_p + EPS_D * wi_world, wi_world), &isect2))
      {
        LightSpectrum emission = isect2.bsdf->get_emission();
        double r1 = (isect.hit_p - global_black_hole.o).norm();
        double r2 = (isect2.hit_p - global_black_hole.o).norm();
        double reduction = 0.5;
        double shift = sqrt((r1 - global_black_hole.r / reduction) * r2 / ((r2 - global_black_hole.r / reduction ) * r1));
        LightSpectrum reflected_spectra = isect.bsdf->spectrum_f(w_out, w_in);
        L_out += emission.doppler(shift) * reflected_spectra * w_in.z;
      }
    }
    L_out *= 2 * M_PI / num_samples;
    return L_out * 2 * M_PI / num_samples;
  }

  Spectrum PathTracer::estimate_direct_lighting_importance(const Ray& r, const Intersection& isect) {
    Matrix3x3 o2w;
    make_coord_space(o2w, isect.n);
    Matrix3x3 w2o = o2w.T();
    const Vector3D& hit_p = isect.hit_p;
    const Vector3D& w_out = w2o * isect.w_out;
    Spectrum L_out;
    int total_num_samples = 0;
    for (SceneLight *sl : scene->lights) {
      int num_samples = sl->is_delta_light() ? 1 : ns_area_light;
      total_num_samples += num_samples;
      for (int i = 0; i < num_samples; ++i) {
        Vector3D wi_world;
        float dist;
        float pdf;
        Spectrum sample = sl->sample_L(hit_p, &wi_world, &dist, &pdf);
        Vector3D w_in = w2o * wi_world;
        if (w_in.z < 0)
          continue;
        if (!bvh->intersect(Ray(hit_p + EPS_D * wi_world, wi_world, dist)))
          L_out += sample * isect.bsdf->f(w_out, w_in) * w_in.z / pdf;
      }
    }
    return L_out / total_num_samples;
  }

  LightSpectrum PathTracer::zero_bounce_radiance(const Ray&r, const Intersection& isect) {
    return isect.bsdf->get_emission();
  }

  LightSpectrum PathTracer::one_bounce_radiance(const Ray&r, const Intersection& isect) {
    if (direct_hemisphere_sample)
      return estimate_direct_lighting_hemisphere(r, isect);
    return estimate_direct_lighting_hemisphere(r, isect);
  }

  LightSpectrum PathTracer::at_least_one_bounce_radiance(const Ray&r, const Intersection& isect) {
    Matrix3x3 o2w;
    make_coord_space(o2w, isect.n);
    Matrix3x3 w2o = o2w.T();
    Vector3D hit_p = isect.hit_p;
    Vector3D w_out = w2o * isect.w_out;
    LightSpectrum L_out;
    if (!isect.bsdf->is_delta())
      L_out += one_bounce_radiance(r, isect);
#if ILLUM == 3
    if (max_ray_depth == r.depth)
      L_out = LightSpectrum();
#endif // ILLUM
    const double prob = 0.7;
    if (r.depth == max_ray_depth || (r.depth > 1 && coin_flip(prob))) {
      Vector3D w_in;
      float pdf;
      LightSpectrum sample = isect.bsdf->sample_spectrum_f(w_out, &w_in, &pdf);
      if (pdf == 0.0f)
        return L_out;
      Vector3D wi_world = o2w * w_in;
      Ray r2(hit_p + EPS_D * wi_world, wi_world);
      r2.depth = r.depth - 1;
      Intersection isect2;
      if (bvh->intersect(r2, &isect2)) {
        LightSpectrum L = at_least_one_bounce_radiance(r2, isect2);
        if (isect.bsdf->is_delta()) {
          L += zero_bounce_radiance(r2, isect2);
        }
        L_out += L * sample * abs(w_in.z) / pdf / prob;
      }
    }
    return L_out;
  }

  LightSpectrum PathTracer::est_radiance_global_illumination(const Ray &r) {
    Intersection isect;
    Spectrum L_out;
    if (!bvh->intersect(r, &isect))
      return LightSpectrum();
#if ILLUM == 0
    return normal_shading(isect.n);
#elif ILLUM == 1
    if (direct_hemisphere_sample)
      return estimate_direct_lighting_hemisphere(r, isect);
    return estimate_direct_lighting_importance(r, isect);
#elif ILLUM == 2
    if (max_ray_depth == 0)
      return zero_bounce_radiance(r, isect);
    if (max_ray_depth == 1)
      return zero_bounce_radiance(r, isect) + one_bounce_radiance(r, isect);
    return zero_bounce_radiance(r, isect) + at_least_one_bounce_radiance(r, isect);
#elif ILLUM == 3
    return at_least_one_bounce_radiance(r, isect);
#endif // ILLUM
  }

  Spectrum PathTracer::raytrace_pixel(size_t x, size_t y, bool useThinLens) {
    Vector2D o = Vector2D(x, y);
    Spectrum ret;
    LightSpectrum spectra = LightSpectrum();
    int i;
    double s1 = 0.0, s2 = 0.0;
    for (i = 0; i < ns_aa; ++i) {
      Vector2D sample = o;
      if (ns_aa == 1)
        sample += Vector2D(0.5, 0.5);
      else
        sample += gridSampler->get_sample();
      Ray r = [&]() -> Ray {
        if (useThinLens) {
          Vector2D samplesForLens = gridSampler->get_sample();
          return camera->generate_ray_for_thin_lens(sample.x / sampleBuffer.w, sample.y / sampleBuffer.h, samplesForLens.x, samplesForLens.y * 2 * M_PI);
        } else {
          return camera->generate_ray(sample.x / sampleBuffer.w, sample.y / sampleBuffer.h);
        }
      }();
      r.depth = max_ray_depth;
      spectra += est_radiance_global_illumination(r);
// #if ADAPTIVE == 1
//       double illum = s.illum();
//       s1 += illum;
//       s2 += illum * illum;
//       if ((i + 1) % samplesPerBatch == 0) {
//         double avg = s1 / (i + 1),
//                sd = sqrt((s2 - avg * s1) / i);
//         if (1.96 * sd / sqrt(i + 1) <= maxTolerance * avg) {
//           ++i;
//           break;
//         }
//       }
// #endif // ADAPTIVE
    }
    sampleCountBuffer[x + y * frameBuffer.w] = i;
    spectra /= i;
    // cout << ret << endl;
    // cout << spectra.toRGB() << endl;
    return spectra.toRGB();
  }

  // Diffuse BSDF //

  Spectrum DiffuseBSDF::f(const Vector3D& wo, const Vector3D& wi) {
    return reflectance / M_PI;
  }

  double softmax(double x, double k, double x0) {
    return 0.95 / (1 + exp(-k * (x - x0)));
  }

  double red_reflect(double wav) {
    return softmax(wav, 5.0, 590.0);
  }

  double blue_reflect(double wav) {
    return softmax(-wav, 5.0, 490.0);
  }

  double green_reflect(double wav) {
    if (wav < 540.0)
      return softmax(wav, 5.0, 490.0);
    else
      return softmax(-wav, 5.0, 590.0);
  }

  LightSpectrum DiffuseBSDF::spectrum_f(const Vector3D &wo, const Vector3D& wi) {
    LightSpectrum ret = LightSpectrum();
    if (reflectance.r == reflectance.g &&
        reflectance.r == reflectance.b &&
        reflectance.g == reflectance.b) //white wall effect
    {
      return ret.whiteSpectrum() / M_PI;
    }
    else if (reflectance.r == 0.6f &&
             reflectance.g == 0.2f &&
             reflectance.b == 0.2f) // this is the redwall hard code
    {
      return ret.redSpectrum() / M_PI;
    }
    else if (reflectance.r == 0.2f &&
             reflectance.g == 0.2f &&
             reflectance.b == 0.6f) // green right wall
    {
      return ret.greenSpectrum() / M_PI;
    }
    else
    {
      double step_size = (ret.max_wav - ret.min_wav) / ret.num_channels;

      #pragma omp parallel for
      for (int i = 0; i < ret.num_channels; i++) {
        double wav = ret.min_wav + i * step_size;
        ret.intensities[i] *= reflectance.r * red_reflect(wav)
                            + reflectance.g * green_reflect(wav)
                            + reflectance.b * blue_reflect(wav);
      }
      return ret;
    }
  }

  LightSpectrum DiffuseBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
    return LightSpectrum();
    // return f(wo, *wi = sampler.get_sample(pdf), _);
  }

  LightSpectrum DiffuseBSDF::sample_spectrum_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
    return spectrum_f(wo, *wi = sampler.get_sample(pdf));
  }

  // Camera //

  template<class T>
  inline T lerp(const T &a, const T &b, double t) {
    return (1 - t) * a + t * b;
  }

  Ray Camera::generate_ray(double x, double y) const {
    Vector2D bl(-tan(radians(hFov) / 2), -tan(radians(vFov) / 2));
    Ray ret(pos, (c2w * Vector3D(lerp(bl.x, -bl.x, x), lerp(bl.y, -bl.y, y), -1)).unit(), fClip);
    ret.min_t = nClip;
    return ret;
  }
}
