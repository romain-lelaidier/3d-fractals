#include "math.cuh"

__device__ __host__
void vec3_add(vec3* result, vec3* a, vec3* b) {
    result->raw[0] = a->raw[0] + b->raw[0];
    result->raw[1] = a->raw[1] + b->raw[1];
    result->raw[2] = a->raw[2] + b->raw[2];
}

__device__ __host__
void vec3_sub(vec3* result, vec3* a, vec3* b) {
    result->raw[0] = a->raw[0] - b->raw[0];
    result->raw[1] = a->raw[1] - b->raw[1];
    result->raw[2] = a->raw[2] - b->raw[2];
}

__device__ __host__
double vec3_dot(vec3* a, vec3* b) {
    return a->raw[0] * b->raw[0]
        + a->raw[1] * b->raw[1]
        + a->raw[2] * b->raw[2];
}

__device__ __host__
double vec3_norm(vec3* v) {
    return sqrt(vec3_dot(v, v));
}

__device__ __host__
void vec3_normalize(vec3* v) {
    double n = vec3_norm(v);
    if (n > 0.0) {
        v->raw[0] /= n;
        v->raw[1] /= n;
        v->raw[2] /= n;
    }
}