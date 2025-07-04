#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH

#include "ray.cuh"

struct Sphere {
    float3 center;
    float radius;
    float3 color;
};

__device__ void colorizeSphere(const Sphere &sphere, Ray* ray, int i);

#endif