#include "raytracer.cuh"

#include "ray.cuh"
#include "geometry.cuh"

#define RPP 1   // rays per pixel (linear, so in reality there are RPP*RPP rays per pixel)



// ----------- dtc() -----------
//
// converts a float to an rgb value for colorization
// from [0,1] to [|0,255|]
// if the input float is not contained in [0,1], its value is clamped to [0,1]
__device__ unsigned char dtc(float d) {
    if (d <= 0.0) return 0;
    if (d >= 255.0) return 255;
    return (char)d;
}



// ----------- colorizeRay() -----------
//
// updates the ray's position depending on the scene
__device__
void colorizeRay(Ray* ray, Julia &julia) {
    ray->color.x = 0.3;
    ray->color.y = 0.3;
    ray->color.z = 0.3;

    normalize(&ray->direction);

    // Sphere s = {
    //     make_float3(0, 0, 0),
    //     1
    // };

    // colorizeSphere(&s, ray);

    colorizeJulia(julia, ray);
}



// ----------- colorizePixel() -----------
//
// colorizes the input pixel depending on the scene :
// generates rays for the pixel, more than one if anti-aliasing is wanted (RPP>1)
// and colorizes the pixel with the mean of the rays' colors
__device__
void colorizePixel(unsigned char* pixel, int px, int py, int width, int height, Camera &camera, Julia &julia) {
    float dw = (float)width;
    float dh = (float)height;
    float sx = (float)px;
    float sy = (float)py;
    float ndcx;
    float ndcy;
    Ray ray;

    int i, j;
    float3 color = make_float3(0.0, 0.0, 0.0);

    for (i = 0; i < RPP; i++) for (j = 0; j < RPP; j++) {
        ndcx = ((sx + ((float)i+0.5) / (float)RPP) / dw) * 2.0 - 1.0;
        ndcy = ((sy + ((float)j+0.5) / (float)RPP) / dh) * 2.0 - 1.0;

        ray.origin = camera.position;
        ray.direction = ndcToDirection(ndcx, ndcy, dw/dh, camera);
        colorizeRay(&ray, julia);
        color = color + ray.color;
    }

    color = color / (float)(RPP*RPP);

    pixel[0] = dtc(color.x*255);
    pixel[1] = dtc(color.y*255);
    pixel[2] = dtc(color.z*255);
}



__global__
void rayTraceKernel(
    unsigned char* imageData,
    int width, int height,
    Camera &camera, Julia &julia
) {
    // gathering pixel's position
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate pixel's color on screen
    if (px < width && py < height) {
        int index = (py * width + px) * 3;
        colorizePixel(&imageData[index], px, py, width, height, camera, julia);
    }
}