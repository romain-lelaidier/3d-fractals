#ifndef MATH_CUH
#define MATH_CUH

#define PI 3.14159

struct a2fields {
    double h;
    double v;
};

union a2 {
    a2fields fields;
    double raw[2];
};

struct vec3fields {
    double x;
    double y;
    double z;
};

union vec3 {
    vec3fields fields;
    double raw[3];
};

__device__ __host__ void vec3_add(vec3* result, vec3* a, vec3* b);
__device__ __host__ void vec3_sub(vec3* result, vec3* a, vec3* b);
__device__ __host__ double vec3_norm(vec3* v);
__device__ __host__ double vec3_dot(vec3* a, vec3* b);
__device__ __host__ void vec3_normalize(vec3* v);
__device__ __host__ void vec3_cross(vec3* result, vec3* a, vec3* b);
__device__ __host__ void vec3_rotate(vec3* v, vec3* ax, double angle);

struct mat4 {
    double m[16];
};

__device__ __host__ void mat4_mul(mat4* result, mat4* a, mat4* b);

__device__ __host__ void clamp(double* f, double mn, double mx);

#endif