#ifndef CGL_BLACKBODY_H
#define CGL_BLACKBODY_H

#define DEFAULT_T 3000.0
#include "lightspectrum.h"


namespace CGL {

	class BlackBodySpectrum : public LightSpectrum {
		public:
			double T;

		BlackBodySpectrum(int num_channels = DEFAULT_NUM, double min_wav = DEFAULT_MIN,
		double max_wav = DEFAULT_MAX, double T = DEFAULT_T) : LightSpectrum(num_channels, min_wav, max_wav), T(T) {
			double step_size = (max_wav - min_wav) / num_channels;
			for (int i = 0; i < num_channels; i++) {
				double lambda = min_wav + i * step_size;
				intensities[i] = planck_distribution(lambda);
			}
		}
		double planck_distribution(double lambda);
		BlackBodySpectrum doppler(double s);
		Spectrum toRGB(void);
	}; //class BlackBodySpectrum
}

#endif //CGL_BLACKBODY_H
