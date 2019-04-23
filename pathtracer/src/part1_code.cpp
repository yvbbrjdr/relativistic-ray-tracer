//
// TODO: Copy over 3-1 code after turning on BUILD_3-1 flag
//

#include "part1_code.h"
#include <time.h>
#include "static_scene/blackhole.h"

using namespace CGL::StaticScene;

using std::min;
using std::max;

namespace CGL {

  Spectrum PathTracer::estimate_direct_lighting_hemisphere(const Ray& r, const Intersection& isect) {
    Matrix3x3 o2w;
    make_coord_space(o2w, isect.n);
    Matrix3x3 w2o = o2w.T();
    const Vector3D& hit_p = isect.hit_p;
    const Vector3D& w_out = w2o * isect.w_out;
    int num_samples = scene->lights.size() * ns_area_light;
    Spectrum L_out;
    for (int i = 0; i < num_samples; ++i) {
      Vector3D w_in = hemisphereSampler->get_sample();
      Vector3D wi_world = o2w * w_in;
      Intersection isect2;
      if (bvh->intersect(Ray(hit_p + EPS_D * wi_world, wi_world), &isect2))
        L_out += isect2.bsdf->get_emission() * isect.bsdf->f(w_out, w_in) * w_in.z;
    }
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

  Spectrum PathTracer::zero_bounce_radiance(const Ray&r, const Intersection& isect) {
    return isect.bsdf->get_emission();
  }

  Spectrum PathTracer::one_bounce_radiance(const Ray&r, const Intersection& isect) {
    if (direct_hemisphere_sample)
      return estimate_direct_lighting_hemisphere(r, isect);
    return estimate_direct_lighting_importance(r, isect);
  }

  Spectrum PathTracer::at_least_one_bounce_radiance(const Ray&r, const Intersection& isect) {
    Matrix3x3 o2w;
    make_coord_space(o2w, isect.n);
    Matrix3x3 w2o = o2w.T();
    Vector3D hit_p = isect.hit_p;
    Vector3D w_out = w2o * isect.w_out;
    Spectrum L_out;
    if (!isect.bsdf->is_delta())
      L_out += one_bounce_radiance(r, isect);
#if ILLUM == 3
    if (max_ray_depth == r.depth)
      L_out = Spectrum();
#endif // ILLUM
    const double prob = 0.7;
    if (r.depth == max_ray_depth || (r.depth > 1 && coin_flip(prob))) {
      Vector3D w_in;
      float pdf;
      Spectrum sample = isect.bsdf->sample_f(w_out, &w_in, &pdf);
      if (pdf == 0.0f)
        return L_out;
      Vector3D wi_world = o2w * w_in;
      Ray r2(hit_p + EPS_D * wi_world, wi_world);
      r2.depth = r.depth - 1;
      Intersection isect2;
      if (bvh->intersect(r2, &isect2)) {
        Spectrum L = at_least_one_bounce_radiance(r2, isect2);
        if (isect.bsdf->is_delta())
          L += zero_bounce_radiance(r2, isect2);
        L_out += L * sample * abs(w_in.z) / pdf / prob;
      }
    }
    return L_out;
  }

  Spectrum PathTracer::est_radiance_global_illumination(const Ray &r) {
    Intersection isect;
    Spectrum L_out;
    if (!bvh->intersect(r, &isect))
      return envLight ? envLight->sample_dir(r) : L_out;
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
      Spectrum s = est_radiance_global_illumination(r);
      ret += s;
#if ADAPTIVE == 1
      double illum = s.illum();
      s1 += illum;
      s2 += illum * illum;
      if ((i + 1) % samplesPerBatch == 0) {
        double avg = s1 / (i + 1),
               sd = sqrt((s2 - avg * s1) / i);
        if (1.96 * sd / sqrt(i + 1) <= maxTolerance * avg) {
          ++i;
          break;
        }
      }
#endif // ADAPTIVE
    }
    sampleCountBuffer[x + y * frameBuffer.w] = i;
    return ret / i;
  }

  // Diffuse BSDF //

  Spectrum DiffuseBSDF::f(const Vector3D& wo, const Vector3D& wi) {
    return reflectance / M_PI;
  }

  Spectrum DiffuseBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
    return f(wo, *wi = sampler.get_sample(pdf));
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
    //use phi = 0 initial convention
    Vector3D r_vec = ret.o - global_black_hole.o;
    double theta = acos(dot(r_vec.unit(), global_black_hole.spin_axis));
    Vector3D x_hat = (r_vec - r_vec.norm() * global_black_hole.spin_axis * cos(theta)).unit();
    Vector3D y_hat = cross(global_black_hole.spin_axis, x_hat);
    Vector3D theta_hat = x_hat * cos(theta) - global_black_hole.spin_axis * sin(theta);
    Vector3D phi_hat = y_hat;
    //begin precalcs, remember unit mass assumption
    ret.b = dot(phi_hat, ret.d);
    ret.q = pow(dot(theta_hat, ret.d), 2.0) +
    pow(cos(theta), 2.0) * (pow(ret.b / sin(theta), 2.0) - global_black_hole.a * global_black_hole.a);
    //end precalcs
    return ret;
  }
}
