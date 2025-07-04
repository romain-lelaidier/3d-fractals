#ifndef FRACTAL_CUH
#define FRACTAL_CUH

#include "ray.cuh"
#include "math.cuh"

struct Julia {
    float4 muGenerator;
    float4 mu;
    float epsilon;  // epsilon for bounding volumes termination
    float geps;     // epsilon for gradient calculation
    uint n;
};

__device__ void colorizeJulia(const Julia &julia, Ray* ray, int i);

#endif