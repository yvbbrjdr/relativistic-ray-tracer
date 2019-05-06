#include <stdio.h>
#include <math.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define Pi 3.1415926

__device__ float2 gridSampler(curandState *s) {
    float2 rt;
    rt.x = curand_uniform(s);
    rt.y = curand_uniform(s);
    return rt;
}

__device__ inline void
scaleVector3D(float* X, float a)
{
    X[0] *= a;
    X[1] *= a;
    X[2] *= a;
}

// S = X + a * Y
__device__ inline void
addScaledVector3D(float* X, float* Y, float a, float* S)
{
    S[0] = X[0] + Y[0] * a;
    S[1] = X[1] + Y[1] * a;
    S[2] = X[2] + Y[2] * a;
}

__device__ inline void
readVector3D(float x, float y, float z, float *dst) {
    dst[0] = x;
    dst[1] = y;
    dst[2] = z;
}

__device__ inline float
power(float X,float Y)
{
    return pow(X,Y);
}

__device__ inline float
dist3D(const float *X, const float *Y)
{
    return sqrt((X[0]-Y[0])*(X[0]-Y[0])+(X[1]-Y[1])*(X[1]-Y[1])+(X[2]-Y[2])*(X[2]-Y[2]));
}

__device__ inline float
norm3D(const float *X)
{
    return sqrt(X[0]*X[0]+X[1]*X[1]+X[2]*X[2]);
}

__device__ inline void
normalize3D(float *X)
{
    double norm = sqrt(X[0]*X[0]+X[1]*X[1]+X[2]*X[2]);
    X[0] /= norm;
    X[1] /= norm;
    X[2] /= norm;
}

__device__ inline void
initVector3D(const float x, const float y, const float z, float* S)
{
    S[0] = x;
    S[1] = y;
    S[2] = z;
}

__device__ inline void
VectorCross3D(const float *u, float *v, float *s) {
    s[0] = u[1] * v[2] - u[2] * v[1];
    s[1] = u[2] * v[0] - u[0] * v[2];
    s[2] = u[0] * v[1] - u[1] * v[0];
}


__device__ inline void
addVector3D(const float *X, const float *Y, float *S) {
    S[0] = X[0] + Y[0];
    S[1] = X[1] + Y[1];
    S[2] = X[2] + Y[2];
}

 __device__ inline void
subVector3D(const float *X, const float *Y, float *S) {
    S[0] = X[0] - Y[0];
    S[1] = X[1] - Y[1];
    S[2] = X[2] - Y[2];
}

__device__ inline void
readVector3D(const float* src, float* dst) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
}

__device__ inline void
readVector3D(const float3 src, float *dst) {
    dst[0] = src.x;
    dst[1] = src.y;
    dst[2] = src.z;
}

 __device__ inline void
negVector3D(const float *X,float* S)
{
    S[0] = -X[0];
    S[1] = -X[1];
    S[2] = -X[2];
}

 __device__ inline void
addVector3D(const float *X,float* S)
{
    S[0] += X[0];
    S[1] += X[1];
    S[2] += X[2];
}

inline __device__
void VectorTMulMatrix3D(float* X, float* S, float* Y)
{
  for (int i=0;i<3;i++)
   {
     Y[i]=X[0]*S[i]+X[1]*S[i+3]+X[2]*S[i+6];
   }
}

inline __device__
void MatrixMulVector3D(float* S, float* X, float* Y)
{
  for (int i = 0; i < 3; i++)
  {
    Y[i]=X[0]*S[3*i]+X[1]*S[3*i+1]+X[2]*S[3*i+2];
  }
}

__device__ inline float
localization(float r,float h)
{
    float tmp=4*r/h;
    return exp(-tmp*tmp);
}

__device__ inline float
VectorDot3D(const float *X,const float *Y)
{
    return X[0]*Y[0]+X[1]*Y[1]+X[2]*Y[2];
}

inline __device__
float det3D(const float* X)
{
    return X[0]*X[4]*X[8]+X[3]*X[7]*X[2]+X[6]*X[1]*X[5]
    -X[2]*X[4]*X[6]-X[5]*X[7]*X[0]-X[8]*X[1]*X[3];
}

inline __device__
float trace3D(const float* X)
{
    return X[0]+X[4]+X[8];
}

inline __device__
void inverse3D(const float* X,float* Y)
{
    float a=det3D(X);
    Y[0]=(X[4]*X[8]-X[5]*X[7])/a;
    Y[1]=(X[2]*X[7]-X[1]*X[8])/a;
    Y[2]=(X[1]*X[5]-X[2]*X[4])/a;
    Y[3]=(X[5]*X[6]-X[3]*X[8])/a;
    Y[4]=(X[0]*X[8]-X[2]*X[6])/a;
    Y[5]=(X[2]*X[3]-X[0]*X[5])/a;
    Y[6]=(X[3]*X[7]-X[4]*X[6])/a;
    Y[7]=(X[1]*X[6]-X[0]*X[7])/a;
    Y[8]=(X[0]*X[4]-X[1]*X[3])/a;


}

inline __device__
void VectorLMulMatrix3D(float* X,float* S,float* Y)
{
    for (int i=0;i<3;i++)
    {
        Y[i]=X[0]*S[i]+X[1]*S[i+3]+X[2]*S[i+6];
    }

}

inline __device__
void VectorRMulMatrix3D(float* X,float* S,float* Y)
{
    for (int i=0;i<3;i++)
    {
        Y[i]=X[0]*S[3*i]+X[1]*S[3*i+1]+X[2]*S[3*i+2];
    }

}

inline __device__
void MatrixMulMatrix3D(float* X,float* Y,float* Z)
{
    int i,j,k;
    for (k = 0; k < 9; k++)
    {
        i=k/3;
        j=k%3;
        Z[k]=X[3*i]*Y[i]+X[3*i+1]*Y[i+3]+X[3*i+2]*Y[i+6];
    }
}

inline __device__
void MatrixAddMatrix3D(float* X,float* Y,float* Z)
{
    int i,j,k;
    for (k = 0; k < 9; k++)
    {
        Z[k]=X[k]+Y[k];
    }
}

inline __device__
void MatrixScale3D(float* X,float a)
{
    int i,j,k;
    for (k = 0; k < 9; k++)
    {
        X[k]=X[k]*a;
    }
}

inline __device__
void MatrixTranspose3D(float* X,float* S)
{
    for(int k = 0; k < 9; k++)
    {
        S[(k % 3) * 3 + k / 3] = X[k];
    }
}

inline __device__
void make_coord_space(const float* n, float* o2w) {

    float z[3];
    float h[3];
    float x[3];
    float y[3];

    readVector3D(n, (float *) z);
    readVector3D(z, (float *) h);

    if (fabs(h[0]) <= fabs(h[1]) && fabs(h[0]) <= fabs(h[2])) h[0] = 1.0;
    else if (fabs(h[1]) <= fabs(h[0]) && fabs(h[1]) <= fabs(h[2])) h[1] = 1.0;
    else h[2] = 1.0;

    normalize3D(z);
    VectorCross3D(h, z, y);
    normalize3D(y);
    VectorCross3D(z, y, x);
    normalize3D(x);

    o2w[0] = x[0]; o2w[1] = y[0]; o2w[2] = z[0];
    o2w[3] = x[1]; o2w[4] = y[1]; o2w[5] = z[1];
    o2w[6] = x[2]; o2w[7] = y[2]; o2w[8] = z[2];
}

inline __device__ void
gpuAdd(float *A, float *B, float *C)
{
    *C = *A + *B;
}
