#ifndef RAYTRACER_CUH
#define RAYTRACER_CUH

#include "camera.cuh"
#include "fractal.cuh"

__global__ void rayTraceKernel(
    unsigned char* imageData,
    int width, int height,
    Camera &camera, Julia &julia
);

#endif