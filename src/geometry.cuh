#ifndef GEOMETRY_CUH
#define GEOMETRY_CUH

#include "raytracer.cuh"

struct Sphere {
    float3 center;
    float radius;
};

__device__ float colorizeSphere(Sphere* sphere, Ray* ray);

#endif