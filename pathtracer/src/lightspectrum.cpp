#include "lightspectrum.h"
#include "CGL/vector3D.h"
#include "CGL/matrix3x3.h"
#include "CGL/spectrum.h"
#include <iostream>
#include <valarray>
#include <omp.h>

using namespace std;
using std::valarray;

namespace CGL {

	const LightSpectrum white_wall = LightSpectrum::whiteSpectrum();
	const LightSpectrum red_wall = LightSpectrum::redSpectrum();
	const LightSpectrum green_wall = LightSpectrum::greenSpectrum();
	const LightSpectrum flourescent = LightSpectrum::flourescentSpectrum();

	LightSpectrum LightSpectrum::doppler(double s) {
		/*
		Returns a LightSpectrum instance Doppler shifted by factor s
		Information is lost since we can only record a reduced
		range.
		*/
		LightSpectrum l = LightSpectrum();
		double step_size = (max_wav - min_wav) / num_channels;
		#pragma omp parallel for
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
		return 0.362*exp(-0.5*dParam1*dParam1) + 1.056*exp(-0.5*dParam2*dParam2)
						-0.065*exp(-0.5*dParam3*dParam3);
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
		// #pragma omp parallel for
		for (int i = 0; i < num_channels; i++) {
			double lambda = min_wav + i * step_size;
			XYZ.x += step_size * CIE_X(lambda) * intensities[i];
			XYZ.y += step_size * CIE_Y(lambda) * intensities[i];
			XYZ.z += step_size * CIE_Z(lambda) * intensities[i];
		}
	 	Vector3D xyz = XYZ;
		// XYZ.x = 10.0 / xyz.y * xyz.x;
		// XYZ.y = 10.0 / xyz.y;
		// XYZ.z = 10.0 / xyz.y * (1 - xyz.x - xyz.y);
		// cout << xyz << endl;
		return xyz;
	}

	Spectrum LightSpectrum::toRGB(void) {
		/*
		Returns native CGL, RGB Spectrum
		*/
		Matrix3x3 M_inv = Matrix3x3(
			3.2404542, -1.5371385, -0.4985314,
			-0.9692660, 1.8760108, 0.0415560,
			0.0556434, -0.2040259, 1.0572252
			);
		Vector3D XYZ = toCIE_XYZ() * 4e-4;
		Vector3D RGB;
		// cout << XYZ.y << endl;
		RGB = M_inv * XYZ;
		// cout << RGB << endl;
		// return Spectrum(max(0.0, RGB[0]),
		// 				max(0.0, RGB[1]),
		// 				max(0.0, RGB[2]));
		return Spectrum(min(1.0, max(0.0, RGB[0])),
										min(1.0, max(0.0, RGB[1])),
										min(1.0, max(0.0, RGB[2])));
		// return Spectrum(RGB[0], RGB[1], RGB[2]);
	}

	LightSpectrum LightSpectrum::whiteSpectrum(void) {
		valarray<double> reflectances = {0.445,0.551,0.624,0.665,0.687,0.708,0.723,0.715,0.710,0.745,0.758,0.739,0.767,0.777,0.765,0.751,0.745,0.748,0.729,0.745,0.757,0.753,0.750,0.746,0.747,0.735,0.732,0.739,0.734,0.725,0.721,0.733,0.725,0.732,0.743,0.744,0.748,0.728,0.716,0.733,0.726,0.713,0.740,0.754,0.764,0.752,0.736,0.734,0.741,0.740,0.732,0.745,0.755,0.751,0.744,0.731,0.733,0.744,0.731,0.712,0.708,0.729,0.730,0.727,0.707,0.703,0.729,0.750,0.760,0.751,0.739,0.724,0.730,0.740,0.737};
		return LightSpectrum(reflectances);
	}

	LightSpectrum LightSpectrum::greenSpectrum(void) {
		valarray<double> reflectances = {0.092,0.096,0.098,0.097,0.098,0.095,0.095,0.097,0.095,0.094,0.097,0.098,0.096,0.101,0.103,0.104,0.107,0.109,0.112,0.115,0.125,0.140,0.160,0.187,0.229,0.285,0.343,0.390,0.435,0.464,0.472,0.476,0.481,0.462,0.447,0.441,0.426,0.406,0.373,0.347,0.337,0.314,0.285,0.277,0.266,0.250,0.230,0.207,0.186,0.171,0.160,0.148,0.141,0.136,0.130,0.126,0.123,0.121,0.122,0.119,0.114,0.115,0.117,0.117,0.118,0.120,0.122,0.128,0.132,0.139,0.144,0.146,0.150,0.152,0.157,0.159};
		return LightSpectrum(reflectances);
	}

	LightSpectrum LightSpectrum::redSpectrum(void) {
		valarray<double> reflectances = {0.040,0.046,0.048,0.053,0.049,0.050,0.053,0.055,0.057,0.056,0.059,0.057,0.061,0.061,0.060,0.062,0.062,0.062,0.061,0.062,0.060,0.059,0.057,0.058,0.058,0.058,0.056,0.055,0.056,0.059,0.057,0.055,0.059,0.059,0.058,0.059,0.061,0.061,0.063,0.063,0.067,0.068,0.072,0.080,0.090,0.099,0.124,0.154,0.192,0.255,0.287,0.349,0.402,0.443,0.487,0.513,0.558,0.584,0.620,0.606,0.609,0.651,0.612,0.610,0.650,0.638,0.627,0.620,0.630,0.628,0.642,0.639,0.657,0.639,0.635,0.642};
		return LightSpectrum(reflectances);
	}

	LightSpectrum LightSpectrum::flourescentSpectrum(void) {
		valarray<double> reflectances = {3.816685,3.947689,4.358293,4.691102,4.939484,5.361605,5.919799,6.124099,6.720893,7.532228,8.091853,8.725799,9.270816,9.920059,10.444841,11.063077,12.005769,12.746861,13.432943,14.021203,14.507416,15.365672,16.470662,17.303148,18.067075,18.804626,19.365283,20.086690,21.089796,21.758586,21.831821,21.971212,22.782304,23.849824,24.594455,24.665197,25.080954,26.281091,28.191974,30.035693,31.303779,32.280187,33.231373,34.182159,34.809670,35.248988,35.917653,37.150591,38.157590,39.226246,39.927377,40.740339,41.435814,42.172329,42.768661,43.222623,43.650980,44.149742,44.756448,45.330340,45.848252,46.305568,46.860484,47.190909,47.509933,47.990097,48.417218,48.724550,48.892026,49.164798,49.171862,49.164724,48.934811,48.565794,47.553848,46.021265};
		LightSpectrum l = LightSpectrum(reflectances);
		return l;
	}
} //class LightSpectrum
