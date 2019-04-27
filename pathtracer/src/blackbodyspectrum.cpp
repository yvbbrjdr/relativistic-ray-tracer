#include "blackbodyspectrum.h"
#include "CGL/vector3D.h"
#include "CGL/spectrum.h"
#include <vector>
#include <omp.h>

using std::vector;

namespace CGL {

	double BlackBodySpectrum::planck_distribution(double lambda) {
		lambda *= 1e-9;
		return (3.7417749e-16 * pow(lambda, -5.0) / (
					exp(0.014387769576158687 / (lambda * T)) - 1.0
				));
	}

	BlackBodySpectrum BlackBodySpectrum::doppler(double s) {
		BlackBodySpectrum b = BlackBodySpectrum(num_channels, min_wav, max_wav, T);
		double step_size = (max_wav - min_wav) / num_channels;
		#pragma omp parallel for
		for (int i = 0; i < num_channels; i++) {
				double lambda = min_wav + i * step_size;
				b.intensities[i] = planck_distribution(lambda / s);
			}
		return b;
	}

	Spectrum BlackBodySpectrum::toRGB(void) {
		return LightSpectrum::toRGB();
	}

	LightSpectrum BlackBodySpectrum::whiteSpectrum(void) {
		return LightSpectrum::whiteSpectrum();
	}

	LightSpectrum BlackBodySpectrum::greenSpectrum(void) {
		return LightSpectrum::greenSpectrum();
	}

	LightSpectrum BlackBodySpectrum::redSpectrum(void) {
		return LightSpectrum::redSpectrum();
	}

	LightSpectrum BlackBodySpectrum::flourescentSpectrum(void) {
		return LightSpectrum::flourescentSpectrum();
	}


} //namespace CGL
