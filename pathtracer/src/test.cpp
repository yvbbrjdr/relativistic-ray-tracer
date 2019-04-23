//
//  test.cpp
//  CGL
//
//  Created by Andrew Aikawa on 4/23/19.
//

#include "static_scene/blackhole.h"
#include "ray.h"

using namespace std;

CGL::StaticScene::BlackHole global_black_hole(nullptr, Vector3D(0, 1, 0), 1, pow(10, -1), Vector3D(0, 0, 1).unit(), 0.25);

const double tol = 0.01;

bool equals(double a, double b) {
  return abs(a - b) <= tol * b;
}

void testRho (void) {
  double r = 4.0;
  double theta = M_PI / 4.0;
  double ret = global_black_hole.rho(r, theta);
  if (!equals(ret, 4.0039))
  {
    cout << "rho expected: 4.0039" << endl << "got: " << ret << endl;
  }
}

void testdel (void) {
  double r = 4.0;
  double ret = global_black_hole.del(r);
  if (!equals(ret, 8.0625))
  {
    cout << "del expected: 4.0039" << endl << "got: " << ret << endl;
  }
}

void testsigma (void) {
  double r = 4.0;
  double theta = M_PI/4;
  double ret = global_black_hole.sigma(r, theta);
  if (!equals(ret, 16.0547))
  {
    cout << "sigma expected: 16.0547" << endl << "got: " << ret << endl;
  }
}

void testdr (void) {
  double r = 4.0;
  double theta = M_PI / 4;
  double pr = 0.4;
  double ret = global_black_hole.dr(r, theta, pr);
  if (!equals(ret, 0.502924))
  {
    cout << "dr expected: 0.502924" << endl << "got: " << ret << endl;
  }
}

void testdtheta (void) {
  double r = 4.0;
  double theta = M_PI / 4;
  double ptheta = 0.3;
  double ret = global_black_hole.dtheta(r, theta, ptheta);
  if (!equals(ret, 0.502924))
  {
    cout << "dtheta expected: 0.502924" << endl << "got: " << ret << endl;
  }
}

void testP (void) {
  double r = 4.0;
  double b = 0.75;
  double ret = global_black_hole.P(r, b);
  if (!equals(ret, 15.8125))
  {
    cout << "dtheta expected: 15.8125" << endl << "got: " << ret << endl;
  }
}



int main( int argc, char** argv ) {
  return 0;
}

