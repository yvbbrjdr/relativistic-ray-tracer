#include "blackbodyspectrum.h"
#include "CGL/vector3D.h"
#include "CGL/spectrum.h"
#include <vector>

using std::vector;

namespace CGL {

	double BlackBodySpectrum::planck_distribution(double lambda) {
		lambda *= 1e-9;
		return (1.1910428661813628e-16 * pow(lambda, -5.0) / (
					exp(0.014387769576158687 / (lambda * T)) - 1.0
				));
	}

	BlackBodySpectrum BlackBodySpectrum::doppler(double s) {
		BlackBodySpectrum b = BlackBodySpectrum(num_channels, min_wav, max_wav, T);
		double step_size = (max_wav - min_wav) / num_channels;
		for (int i = 0; i < num_channels; i++) {
				double lambda = min_wav + i * step_size;
				b.intensities[i] = planck_distribution(lambda / s);
			}
		return b;
	}

	Spectrum BlackBodySpectrum::toRGB(void) {
		return LightSpectrum::toRGB();
	}



} //namespace CGL