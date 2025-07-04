// ----------- RAYTRACER -----------
// interface for rendering the scene by raytracing
// using the GPU kernels

#ifndef RAYTRACER_CUH
#define RAYTRACER_CUH

#include "camera.cuh"
#include "scene.cuh"

__global__ void rayTraceKernel(
    unsigned char* imageData,
    int width, int height,
    Camera &camera, Scene &scene
);

#endif