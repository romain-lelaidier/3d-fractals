#include "raytracer.cuh"

__device__ unsigned char dtc(double d) {
    if (d <= 0.0) return 0;
    if (d >= 255.0) return 255;
    return (char)d;
}

__device__
void color(unsigned char* pixels, double x, double y, double wh, Camera* camera) {
    a2 d;
    d.fields.h = camera->direction.fields.h - atan(x * tan(camera->wideness/2));
    d.fields.v = camera->direction.fields.v + atan((y/wh) * tan(camera->wideness/2));

    vec3 sphereCenter = { .fields = { 0.0, 0.0, 0.0 } };
    vec3 ptc;
    vec3_sub(&ptc, &sphereCenter, &camera->position);

    vec3 u = {
        .fields = {
            cos(d.fields.v) * cos(d.fields.h),
            cos(d.fields.v) * sin(d.fields.h),
            sin(d.fields.v)
        }
    };

    double lambda = max(0.0, vec3_dot(&u, &ptc));
    double dist = lambda * lambda - 2 * lambda * vec3_dot(&u, &ptc) + vec3_dot(&ptc, &ptc);

    int c = dist < 1.0 ? 255 : 0;
    pixels[0] = c;
    pixels[1] = c;
    pixels[2] = c;

    // pixels[0] = dtc(u.raw[0] * 255);
    // pixels[1] = dtc(u.raw[1] * 255);
    // pixels[2] = dtc(u.raw[2] * 255);
}

__global__
void rayTraceKernel(unsigned char* imageData, int width, int height, Camera* camera) {
    // gathering pixel's position
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    double dw = (double)width;
    double dh = (double)height;

    // calculate pixel's color on screen
    if (px < width && py < height) {
        int index = (py * width + px) * 3;
        double x = ((double)px / dw) * 2.0 - 1.0;
        double y = ((double)py / dh) * 2.0 - 1.0;
        color(&imageData[index], x, y, dw/dh, camera);
    }
}