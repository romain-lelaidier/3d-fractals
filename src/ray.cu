#include "ray.cuh"

#include "math.cuh"

__device__ void hitAndBounceRay(Ray* ray) {
    ray->origin = ray->origin + ray->distance * ray->direction;
    float3 nn = normalize(ray->normal);
    float3 nd = (-1.0) * normalize(ray->direction);
    float x = dot(nn, nd);
    ray->direction = (2.0*x) * nn - nd;
    ray->bounces++;
}