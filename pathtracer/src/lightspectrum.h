#ifndef CGL_LIGHTSPECTRUM_H
#define CGL_LIGHTSPECTRUM_H
#include <vector>
#include "CGL/vector3D.h"
#include "CGL/spectrum.h"
using std::vector;

#define DEFAULT_NUM 340
#define DEFAULT_MIN 380.0
#define DEFAULT_MAX 740.0

namespace CGL {

class LightSpectrum {
 public:
  const int num_channels; //number of wavelength channels
  const double min_wav; //min wavelength channel
  const double max_wav; //max wavelength channel
  vector<double> intensities; //color channels

  /*
  Currently only supports arithmetic on same size color channels.
  Default range is 380nm to 740nm with step size 1nm
  */

  LightSpectrum(int num_channels = DEFAULT_NUM, double min_wav = DEFAULT_MIN, double max_wav = DEFAULT_MAX, vector<double> intensities = vector<double>(DEFAULT_NUM)) 
  : num_channels(num_channels), min_wav(min_wav), max_wav(max_wav), intensities(intensities) {}

  inline LightSpectrum operator+(const LightSpectrum &rhs) const {
  	LightSpectrum l = LightSpectrum();
  	for (int i = 0; i < num_channels; i++)
  		l.intensities[i] = intensities[i] + rhs.intensities[i];
  	return l;
}

  inline LightSpectrum &operator+=(const LightSpectrum &rhs) {
    for (int i = 0; i < num_channels; i++)
  		intensities[i] += rhs.intensities[i];
    return *this;
  }

  inline LightSpectrum operator-(const LightSpectrum &rhs) const {
    LightSpectrum l = LightSpectrum();
  	for (int i = 0; i < num_channels; i++)
  		l.intensities[i] = intensities[i] - rhs.intensities[i];
  	return l;
  }

  inline LightSpectrum &operator-=(const LightSpectrum &rhs) {
    for (int i = 0; i < num_channels; i++)
  		intensities[i] -= rhs.intensities[i];
    return *this;
  }

  inline LightSpectrum operator*(const LightSpectrum &rhs) const {
  	LightSpectrum l = LightSpectrum();
  	for (int i = 0; i < num_channels; i++)
  		l.intensities[i] = intensities[i] * rhs.intensities[i];
  	return l;
  }

  inline LightSpectrum &operator*=(const LightSpectrum &rhs) {
    for (int i = 0; i < num_channels; i++)
  		intensities[i] *= rhs.intensities[i];
    return *this;
  }

  inline LightSpectrum operator*(float s) const {
    LightSpectrum l = LightSpectrum();
  	for (int i = 0; i < num_channels; i++)
  		l.intensities[i] = intensities[i] * s;
  	return l;
  }

  inline LightSpectrum &operator*=(float s) {
    for (int i = 0; i < num_channels; i++)
  		intensities[i] *= s;
    return *this;
  }

  inline LightSpectrum operator/(const LightSpectrum &rhs) const {
    LightSpectrum l = LightSpectrum();
  	for (int i = 0; i < num_channels; i++)
  		l.intensities[i] = intensities[i] / rhs.intensities[i];
  	return l;
  }

  inline LightSpectrum &operator/=(const LightSpectrum &rhs) {
    for (int i = 0; i < num_channels; i++)
  		intensities[i] /= rhs.intensities[i];
    return *this;
  }

  inline LightSpectrum operator/(float s) const {
    LightSpectrum l = LightSpectrum();
  	for (int i = 0; i < num_channels; i++)
  		l.intensities[i] = intensities[i] / s;
  	return l;
  }

  inline LightSpectrum &operator/=(float s) {
    for (int i = 0; i < num_channels; i++)
  		intensities[i] /= s;
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