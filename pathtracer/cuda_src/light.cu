__device__ float3 uniformHemisphereSampler(curandState *s) {
    float2 tmp = gridSampler(s);
    double r1 = tmp.x;
    double r2 = tmp.y;

     double sin_theta = sqrt(1 - r1 * r1);
    double phi = 2 * PI * r2;

     float3 rt;
    rt.x = sin_theta * cos(phi);
    rt.y = sin_theta * sin(phi);
    rt.z = r1;
    return rt;
}
