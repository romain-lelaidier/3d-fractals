#ifndef RAY_CUH
#define RAY_CUH

struct Ray {
    float3 origin;
    float3 direction;
    float3 color;
};

__device__ void bounceRay(Ray* ray, const float3 &normal);

#endif