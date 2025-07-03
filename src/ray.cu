#include "ray.cuh"

#include "math.cuh"

__device__ void bounceRay(Ray* ray, const float3 &normal) {
    float3 nn = normalize(normal);
    float3 nd = (-1.0) * normalize(ray->direction);
    float x = dot(nn, nd);
    ray->direction = (2.0*x) * nn - nd;
}