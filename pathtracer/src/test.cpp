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
    cout << "del expected: 8.0625" << endl << "got: " << ret << endl;
  }
}

void testsigma (void) {
  double r = 4.0;
  double theta = M_PI/4;
  double ret = global_black_hole.sigma(r, theta);
  if (!equals(ret, 16.0547))
  {
    cout << "sigma expected: 16.05465" << endl << "got: " << ret << endl;
  }
}

void testdr (void) {
  double r = 4.0;
  double theta = M_PI / 4;
  double pr = 0.4;
  double ret = global_black_hole.dr(r, theta, pr);
  if (!equals(ret, 0.20116959))
  {
    cout << "dr expected: 0.20116959" << endl << "got: " << ret << endl;
  }
}

void testdtheta (void) {
  double r = 4.0;
  double theta = M_PI / 4;
  double ptheta = 0.4;
  double ret = global_black_hole.dtheta(r, theta, ptheta);
  if (!equals(ret, 0.02495126))
  {
    cout << "dtheta expected: 0.02495126" << endl << "got: " << ret << endl;
  }
}

void testP (void) {
  double r = 4.0;
  double b = 0.75;
  double ret = global_black_hole.P(r, b);
  if (!equals(ret, 15.875))
  {
    cout << "P expected: 15.875" << endl << "got: " << ret << endl;
  }
}

void testR(void) {
  double r = 4.0;
  double b = 0.75;
  double q = 1.0;
  double ret = global_black_hole.R(r, b, q);
  if (!equals(ret, 241.9375
))
  {
    cout << "R expected: 241.9375" << endl << "got: " << ret << endl;
  }
}

void testbig_Theta(void) {
  double theta = M_PI / 4;
  double b = 0.75;
  double q = 1.0;
  double ret = global_black_hole.big_Theta(theta, b, q);
  if (!equals(ret, 0.46874999
              ))
  {
    cout << "big_Theta expected: 0.46874999" << endl << "got: " << ret << endl;
  }
}

void testRDT(void) {
  double r = 4.0;
  double theta = M_PI / 4;
  double b = 0.75;
  double q = 1.0;
  double ret = global_black_hole.RDT(r, theta, b, q);
  if (!equals(ret, 0.95053
              ))
  {
    cout << "RDT expected: 0.95053" << endl << "got: " << ret << endl;
  }
}

void testdphi(void) {
  double r = 4.0;
  double theta = M_PI / 4;
  double b = 0.75;
  double q = 1.0;
  double ret = global_black_hole.dphi(r, theta, b, q);
  if (!equals(ret, 0.108678
              ))
  {
    cout << "dphi expected: 0.108678" << endl << "got: " << ret << endl;
  }
}

void testpredpr(void) {
  double r = 4.0;
  double theta = M_PI / 4;
  double pr = 0.66;
  double ptheta = 0.66;
  double b = 0.75;
  double q = 1.0;
  double ret = global_black_hole.predpr(r, theta, pr, ptheta, b, q);
  if (!equals(ret, 0.8274113664868461
              ))
  {
    cout << "predpr expected: 0.8274113664868461" << endl << "got: " << ret << endl;
  }
}

void testdpr(void) {
  double r = 4.0;
  double theta = M_PI / 4;
  double pr = 0.66;
  double ptheta = 0.66;
  double b = 0.75;
  double q = 1.0;
  double ret = global_black_hole.dpr(r, theta, pr, ptheta, b, q);
  if (!equals(ret, 2.2929872732060885
              ))
  {
    cout << "dpr expected: 2.2929872732060885" << endl << "got: " << ret << endl;
  }
}

void testdptheta(void) {
  double r = 4.0;
  double theta = M_PI / 4;
  double pr = 0.66;
  double ptheta = 0.66;
  double b = 0.75;
  double q = 1.0;
  double ret = global_black_hole.dptheta(r, theta, pr, ptheta, b, q);
  if (!equals(ret, 1.850371707708594e-16
              ))
  {
    cout << "dptheta expected: 1.850371707708594e-16" << endl << "got: " << ret << endl;
  }
}


int main( int argc, char** argv ) {
  testRho();
  testdel();
  testsigma();
  testP();
  testR();
  testbig_Theta();
  testdr();
  testdtheta();
  testRDT();
  testpredpr();
  testdpr();
  testdptheta();
  return 0;
}

