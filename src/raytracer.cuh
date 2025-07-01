#ifndef RAYTRACER_CUH
#define RAYTRACER_CUH

#include "math.cuh"

struct Camera {
    vec3 position;
    a2 direction;
    double wideness;
};

__global__ void rayTraceKernel(unsigned char* imageData, int width, int height, Camera* camera);

#endif