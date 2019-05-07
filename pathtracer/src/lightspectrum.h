#ifndef CGL_LIGHTSPECTRUM_H
#define CGL_LIGHTSPECTRUM_H
#include "CGL/vector3D.h"
#include "CGL/spectrum.h"
#include <valarray>
#define DEFAULT_NUM 76
#define DEFAULT_MIN 400.0
#define DEFAULT_MAX 704.0

#include <iostream>

using std::valarray;

using namespace std;

namespace CGL {

class LightSpectrum {
 public:
  const int num_channels; //number of wavelength channels
  const double min_wav; //min wavelength channel
  const double max_wav; //max wavelength channel
  valarray<double> intensities; //color channels

  /*
  Currently only supports arithmetic on same size color channels.
  Default range is 400nm to 700nm with step size 4nm
  */

  LightSpectrum(int num_channels = DEFAULT_NUM, double min_wav = DEFAULT_MIN, double max_wav = DEFAULT_MAX, valarray<double> intensities = valarray<double>(0.0, DEFAULT_NUM))
  : num_channels(num_channels), min_wav(min_wav), max_wav(max_wav), intensities(intensities) {}

  LightSpectrum(valarray<double> intensities) : num_channels(DEFAULT_NUM),
  min_wav(DEFAULT_MIN), max_wav(DEFAULT_MAX), intensities(intensities) {}

  inline LightSpectrum operator+(const LightSpectrum &rhs) const {
  	LightSpectrum l = LightSpectrum(num_channels, min_wav, max_wav, valarray<double>(0.0, num_channels));
  	l.intensities = intensities + rhs.intensities;
  	return l;
}

  inline LightSpectrum &operator+=(const LightSpectrum &rhs) {
  	intensities += rhs.intensities;
    return *this;
  }

  inline LightSpectrum operator-(const LightSpectrum &rhs) const {
    LightSpectrum l = LightSpectrum(num_channels, min_wav, max_wav, valarray<double>(0.0, num_channels));
  	l.intensities = intensities - rhs.intensities;
  	return l;
  }

  inline LightSpectrum &operator-=(const LightSpectrum &rhs) {
  	intensities -= rhs.intensities;
    return *this;
  }

  inline LightSpectrum operator*(const LightSpectrum &rhs) const {
    LightSpectrum l = LightSpectrum(num_channels, min_wav, max_wav, valarray<double>(0.0, num_channels));
  	l.intensities = intensities * rhs.intensities;
  	return l;
  }

  inline LightSpectrum &operator*=(const LightSpectrum &rhs) {
  	intensities *= rhs.intensities;
    return *this;
  }

  inline LightSpectrum operator*(double s) const {
    LightSpectrum l = LightSpectrum(num_channels, min_wav, max_wav, valarray<double>(s, num_channels));
  	l.intensities = intensities * s;
  	return l;
  }

  inline LightSpectrum &operator*=(double s) {
  	intensities *= s;
    return *this;
  }

  inline LightSpectrum operator/(const LightSpectrum &rhs) const {LightSpectrum l = LightSpectrum(num_channels, min_wav, max_wav, valarray<double>(0.0, num_channels));
  	l.intensities = intensities / rhs.intensities;
  	return l;
  }

  inline LightSpectrum &operator/=(const LightSpectrum &rhs) {
  	intensities /= rhs.intensities;
    return *this;
  }

  inline LightSpectrum operator/(double s) const {
    LightSpectrum l = LightSpectrum(num_channels, min_wav, max_wav, valarray<double>(s, num_channels));
  	l.intensities = intensities / s;
  	return l;
  }

  inline LightSpectrum &operator/=(float s) {
  	intensities /= s;
    return *this;
  }

  inline double gaussian_pdf(double x, double mean, double variance) {
    return exp(-(x - mean) * (x - mean) / (2 * variance)) / sqrt(2 * M_PI * variance);
  }

  LightSpectrum doppler(double s);
  double CIE_X(double lambda);
  double CIE_Y(double lambda);
  double CIE_Z(double lambda);
  Vector3D toCIE_XYZ(void);
  Spectrum toRGB(void);
  static LightSpectrum whiteSpectrum(void);
  static LightSpectrum redSpectrum(void);
  static LightSpectrum greenSpectrum(void);
  static LightSpectrum flourescentSpectrum(void);

	}; // class LightSpectrum
  extern const LightSpectrum white_wall;
  extern const LightSpectrum red_wall;
  extern const LightSpectrum green_wall;
  extern const LightSpectrum flourescent;
} //class CGL

#endif // CGL_LIGHTSPECTRUM_H
