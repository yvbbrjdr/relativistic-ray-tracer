#ifndef CGL_LIGHTSPECTRUM_H
#define CGL_LIGHTSPECTRUM_H
#include "CGL/vector3D.h"
#include "CGL/spectrum.h"
#include <valarray>
#define DEFAULT_NUM 400
#define DEFAULT_MIN 380.0
#define DEFAULT_MAX 780.0

using std::valarray;

namespace CGL {

class LightSpectrum {
 public:
  const int num_channels; //number of wavelength channels
  const double min_wav; //min wavelength channel
  const double max_wav; //max wavelength channel
  valarray<double> intensities; //color channels

  /*
  Currently only supports arithmetic on same size color channels.
  Default range is 380nm to 740nm with step size 1nm
  */

  LightSpectrum(int num_channels = DEFAULT_NUM, double min_wav = DEFAULT_MIN, double max_wav = DEFAULT_MAX, valarray<double> intensities = valarray<double>(DEFAULT_NUM))
  : num_channels(num_channels), min_wav(min_wav), max_wav(max_wav), intensities(intensities) {}

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

  LightSpectrum doppler(double s);
  double CIE_X(double lambda);
  double CIE_Y(double lambda);
  double CIE_Z(double lambda);
  Vector3D toCIE_XYZ(void);
  Spectrum toRGB(void);

	}; // class LightSpectrum
}

#endif // CGL_LIGHTSPECTRUM_H
