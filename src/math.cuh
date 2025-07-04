#ifndef MATH_CUH
#define MATH_CUH

#define PI 3.14159

struct a2fields {
    float h;
    float v;
};

union a2 {
    a2fields fields;
    float raw[2];
};

struct mat4 {
    float m[16];
};

__device__ __host__ void mat4_mul(mat4* result, mat4* a, mat4* b);

__device__ __host__ void clamp(float* f, float mn, float mx);

__device__ __host__ float3 operator+(const float3 &a, const float3 &b);
__device__ __host__ float3 operator-(const float3 &a, const float3 &b);
__device__ __host__ float3 operator*(float lambda, const float3 &a);
__device__ __host__ float3 operator/(const float3 &a, float lambda);
__device__ __host__ float3 cross(const float3 &a, const float3 &b);
__device__ __host__ float3 rotate(const float3 &v, const float3 &a, float angle);
__device__ __host__ float3 normalize(const float3 &a);
__device__ __host__ float dot(const float3 &a, const float3 &b);
__device__ __host__ float norm(const float3 &a);
__device__ __host__ void normalize(float3* a);
__device__ __host__ float3 copy_float3(const float3 &a);

typedef float4 Quat; // w + xi + yj + zk

__device__ __host__ Quat operator+(const Quat &a, const Quat &b);
__device__ __host__ Quat operator-(const Quat &a, const Quat &b);
__device__ __host__ Quat operator*(float lambda, const Quat &a);
__device__ __host__ Quat operator*(const Quat &a, const Quat &b);
__device__ __host__ Quat operator/(const Quat &a, float lambda);
__device__ __host__ Quat cross(const Quat &a, const Quat &b);
__device__ __host__ Quat rotate(const Quat &v, const Quat &a, float angle);
__device__ __host__ Quat normalize(const Quat &a);
__device__ __host__ float dot(const Quat &a, const Quat &b);
__device__ __host__ float norm(const Quat &a);
__device__ __host__ void normalize(Quat* a);
__device__ __host__ Quat square(const Quat &a);

#endif