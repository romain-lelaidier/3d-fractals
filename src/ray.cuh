#ifndef RAY_CUH
#define RAY_CUH

struct Ray {
    float3 origin;
    float3 direction;
    float3 color;
    // for multiple objects
    int i;          // index of the closest object's surface hit
    float distance; // distance to this object
    float3 normal;  // normal of this object's surface
    uint bounces;    // number of reflections until then
};

__device__ void hitAndBounceRay(Ray* ray);

#endif