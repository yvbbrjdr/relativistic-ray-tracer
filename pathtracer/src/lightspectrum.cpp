#include "lightspectrum.h"
#include <vector>
#include "CGL/vector3D.h"
#include "CGL/matrix3x3.h"
#include "CGL/spectrum.h"
#include <iostream>

using namespace std;

namespace CGL {

	LightSpectrum LightSpectrum::doppler(double s) {
		/*
		Returns a LightSpectrum instance Doppler shifted by factor s
		Information is lost since we can only record a reduced
		range.
		*/
		LightSpectrum l = LightSpectrum();
		double step_size = (max_wav - min_wav) / num_channels;
		for (int i = 0; i < num_channels; i++) {
			double lambda = min_wav + i * step_size;
			if (lambda > s * max_wav ||
					min_wav * s < lambda)
				l.intensities[i] = 0;
			else
			{
				lambda /= s;
				int j = (int) floor((lambda - min_wav) / step_size);
				l.intensities[i] = intensities[j];
			}
		}
		return l;
	}

	double LightSpectrum::CIE_X(double lambda) {
		return 1.065 * exp(-pow(((lambda - 595.8) / 33.33), 2.0) / 2.0)
			+ 0.366 * exp(-pow(((lambda - 446.8) / 19.44), 2.0) / 2.0);
	}

	double LightSpectrum::CIE_Y(double lambda) {
		return 1.014 * exp(-pow(((log(lambda) - log(595.8)) / 0.075), 2.0) / 2.0);
	}

	double LightSpectrum::CIE_Z(double lambda) {
		return 1.839 * exp(-pow(((log(lambda) - log(449.8)) / 0.051), 2.0) / 2.0);
	}

	Vector3D LightSpectrum::toCIE_XYZ(void) {
		double step_size = (max_wav - min_wav) / num_channels;
		Vector3D XYZ = Vector3D();
		for (int i = 0; i < num_channels; i++) {
			double lambda = min_wav + i * step_size;
			XYZ.x += step_size * CIE_X(lambda) * intensities[i];
			XYZ.y += step_size * CIE_Y(lambda) * intensities[i];
			XYZ.z += step_size * CIE_Z(lambda) * intensities[i];
		}
		return XYZ;
	}

	Spectrum LightSpectrum::toRGB(void) {
		/*
		Returns native CGL, RGB Spectrum
		*/
		Matrix3x3 M_inv = Matrix3x3(
			0.41847, -0.15866, -0.082835,
			-0.091169, 0.25243, 0.015708,
			0.00092090, -0.0025498, 0.17860
														);
		Vector3D RGB = M_inv * toCIE_XYZ();
		// cout << RGB << endl;
		return Spectrum(min(10.0, max(0.0, RGB[0])),
										min(10.0, max(0.0, RGB[1])),
										min(10.0, max(0.0, RGB[2])));
	}



}
