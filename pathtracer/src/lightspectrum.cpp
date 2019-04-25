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

	double LightSpectrum::CIE_X(double wavelen) {
		double dParam1 = (wavelen-442.0)*((wavelen < 442.0)?0.0624:0.0374);
		double dParam2 = (wavelen-599.8)*((wavelen < 599.8)?0.0264:0.0323);
		double dParam3 = (wavelen-501.1)*((wavelen < 501.1)?0.0490:0.0382);
		return 0.362*exp(-0.5*dParam1*dParam1) + 1.056*exp(-0.5*dParam2*dParam2);
	}

	double LightSpectrum::CIE_Y(double wavelen) {
		double dParam1 = (wavelen-568.8)*((wavelen < 568.8)?0.0213:0.0247);
		double dParam2 = (wavelen-530.9)*((wavelen < 530.9)?0.0613:0.0322);
		return 0.821*exp(-0.5*dParam1*dParam1) + 0.286*expf(-0.5*dParam2*dParam2);
	}

	double LightSpectrum::CIE_Z(double wavelen) {
		double dParam1 = (wavelen-437.0)*((wavelen < 437.0)?0.0845:0.0278);
		double dParam2 = (wavelen-459.0)*((wavelen < 459.0)?0.0385:0.0725);
		return 1.217*expf(-0.5*dParam1*dParam1) + 0.681*exp(-0.5*dParam2*dParam2);
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
	 	Vector3D xyz = XYZ / (XYZ.x + XYZ.y + XYZ.z);
		// XYZ.x = 10.0 / xyz.y * xyz.x;
		// XYZ.y = 10.0 / xyz.y;
		// XYZ.z = 10.0 / xyz.y * (1 - xyz.x - xyz.y);
		return xyz * 10;
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
		cout << RGB << endl;
		return Spectrum(min(10.0, max(0.0, RGB[0])),
										min(10.0, max(0.0, RGB[1])),
										min(10.0, max(0.0, RGB[2])));
	}



}
