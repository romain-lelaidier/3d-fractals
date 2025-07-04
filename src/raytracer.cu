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
// calculates the ray's color depending on the scene
__device__ void colorizeRay(Scene &scene, Ray* ray, int originI = -1) {
    ray->i = -1;
    ray->distance = INFINITY;
    normalize(&ray->direction);

    float light = dot(ray->direction, make_float3(0,0,1));
    light = 1 / sqrt(1 + exp(-light));
    ray->color = light * make_float3(242, 232, 207) / 256;
    
    // this loop tries to intersect with every object in the scene
    // to determine which one is closest.
    // the result (index of the object, distance and surface normal)
    // is stored in the ray object.
    int i;
    for (i = 0; i < scene.nobjs; i++) if (i != originI) {
        switch (scene.objs[i].type) {
            case SPHERE:
                colorizeSphere(scene.objs[i].sphere, ray, i);
                break;
            case JULIA:
                colorizeJulia(scene.objs[i].julia, ray, i);
                break;
        }
    }

    if (ray->i >= 0) {
        // one object hit
        Object *obj = &scene.objs[ray->i];

        // reflection
        if (ray->bounces >= 2) return;

        if (obj->type == SPHERE) {
            // reflection
            hitAndBounceRay(ray);
            float3 color = copy_float3(ray->color);
            colorizeRay(scene, ray, ray->i);
            ray->color = 0.7 * color + 0.7 * ray->color;
        }
    }
}



// ----------- colorizePixel() -----------
//
// colorizes the input pixel depending on the scene :
// generates rays for the pixel, more than one if anti-aliasing is wanted (RPP>1)
// and colorizes the pixel with the mean of the rays' colors
__device__
void colorizePixel(unsigned char* pixel, int px, int py, int width, int height, Camera &camera, Scene &scene) {
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
        ray.bounces = 0;
        colorizeRay(scene, &ray);
        color = color + ray.color;
    }

    color = color / (float)(RPP*RPP);

    pixel[0] = dtc(color.x*255);
    pixel[1] = dtc(color.y*255);
    pixel[2] = dtc(color.z*255);
}



// ----------- rayTraceKernel() -----------
//
// Renders the scene on imageData, from one GPU kernel
__global__ void rayTraceKernel(
    unsigned char* imageData,
    int width, int height,
    Camera &camera, Scene &scene
) {
    // gathering pixel's position
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate pixel's color on screen
    if (px < width && py < height) {
        int index = (py * width + px) * 3;
        colorizePixel(&imageData[index], px, py, width, height, camera, scene);
    }
}