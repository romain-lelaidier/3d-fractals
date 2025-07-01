#ifndef RAYTRACER_CUH
#define RAYTRACER_CUH

#include "camera.cuh"
#include "math.cuh"

struct Ray {
    vec3 origin;
    vec3 direction;
    char color[3];
};

struct Sphere {
    vec3 center;
    double radius;
};

__global__ void rayTraceKernel(unsigned char* imageData, int width, int height, Camera* camera);

#endif