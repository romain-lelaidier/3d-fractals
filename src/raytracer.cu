#include "raytracer.cuh"
#include "camera.cuh"

__device__
unsigned char dtc(double d) {
    if (d <= 0.0) return 0;
    if (d >= 255.0) return 255;
    return (char)d;
}

__device__
bool intersectSphere(Sphere* sphere, Ray* ray) {
    int i;
    double upc, delta;
    double ocDist;  // distance from ray origin to sphere center (euclidean)
    vec3 otc;       // vector ray origin to sphere center
    double orcDist; // distance from ray origin to sphere surface (along the ray)
    vec3 sn;        // sphere normal at the intersection

    vec3_sub(&otc, &sphere->center, &ray->origin);
    ocDist = vec3_norm(&otc);

    if (ocDist < sphere->radius) return 0;   // ray origin inside the sphere

    upc = vec3_dot(&ray->direction, &otc);
    delta = upc * upc + sphere->radius * sphere->radius - vec3_dot(&otc, &otc);

    if (delta > 0.0) {
        // intersection
        orcDist = ocDist - sqrt(delta);

        for (i = 0; i < 3; i++) {
            sn.raw[i] = ray->origin.raw[i] + ray->direction.raw[i] * orcDist;
            ray->color[i] = dtc(abs(sn.raw[i]) * 255);
        }

        // ray->color[0] = dtc(orcDist * 10);
        // ray->color[1] = dtc(orcDist * 10);
        // ray->color[2] = dtc(orcDist * 10);

        return 1;
    }

    return 0;
}

__device__
void colorizeRay(Ray* ray) {
    ray->color[0] = 0;
    ray->color[1] = 0;
    ray->color[2] = 0;

    Sphere sphere = { { 0.0, 0.0, 0.0 }, 1.0 };
    intersectSphere(&sphere, ray);
}

__device__
void colorizePixel(unsigned char* color, double ndcx, double ndcy, double wh, Camera* camera) {
    Ray ray;
    ray.origin = camera->position;
    ndcToDirection(&ray.direction, ndcx, ndcy, wh, camera);

    colorizeRay(&ray);

    color[0] = ray.color[0];
    color[1] = ray.color[1];
    color[2] = ray.color[2];
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
        colorizePixel(&imageData[index], x, y, dw/dh, camera);
    }
}