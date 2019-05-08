void addIntensities(float* a, float* b, float* dst, int n);
void subtractIntensities(float *a, float* b, float* dst, int n);
void mulIntensities(float *a, float* b, float* dst, int n);
void mulIntensities(float *a, float s, float* dst, int n);
void divIntensities(float *a, float* b, float* dst, int n);
void divIntensities(float *a, float s, float* dst, int n);

__global__ void addIntensitiesCUDA(float* a, float* b, float* dst, int n);
__global__ void subtractIntensitiesCUDA(float* a, float* b, float* dst, int n);
__global__ void mulIntensitiesCUDA(float *a, float *b, float *dst, int n);
__global__ void mulIntensitiesCUDA(float *a, float s, float *dst, int n);
__global__ void divIntensitiesCUDA(float *a, float *b, float *dst, int n);
__global__ void divIntensitiesCUDA(float *a, float s, float *dst, int n);
