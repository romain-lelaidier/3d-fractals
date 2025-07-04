#include "math.cuh"

// __device__ __host__
// void vec3_add(vec3* result, vec3* a, vec3* b) {
//     result->raw[0] = a->raw[0] + b->raw[0];
//     result->raw[1] = a->raw[1] + b->raw[1];
//     result->raw[2] = a->raw[2] + b->raw[2];
// }

// __device__ __host__
// void vec3_sub(vec3* result, vec3* a, vec3* b) {
//     result->raw[0] = a->raw[0] - b->raw[0];
//     result->raw[1] = a->raw[1] - b->raw[1];
//     result->raw[2] = a->raw[2] - b->raw[2];
// }

// __device__ __host__
// float vec3_dot(vec3* a, vec3* b) {
//     return a->raw[0] * b->raw[0]
//         + a->raw[1] * b->raw[1]
//         + a->raw[2] * b->raw[2];
// }

// __device__ __host__
// float vec3_norm(vec3* v) {
//     return sqrt(vec3_dot(v, v));
// }

// __device__ __host__
// void vec3_normalize(vec3* v) {
//     float n = vec3_norm(v);
//     if (n > 0.0) {
//         v->raw[0] /= n;
//         v->raw[1] /= n;
//         v->raw[2] /= n;
//     }
// }

// __device__ __host__
// void vec3_cross(vec3* result, vec3* a, vec3* b) {
//     result.x = a.y * b.z - a.z * b.y;
//     result.y = a.z * b.x - a.x * b.z;
//     result.z = a.x * b.y - a.y * b.x;
// }

__device__ __host__
void mat4_mul(mat4* result, mat4* a, mat4* b) {
    int i, j, k, l;
    for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) {
        k = i*4+j;
        result->m[k] = 0.0;
        for (l = 0; l < 4; l++) {
            result->m[k] += a->m[i*4+l] * b->m[l*4+j];
        }
    }
}

__device__ __host__
void clamp(float* f, float mn, float mx) {
    *f = min(max(*f, mn), mx);
}


__device__ __host__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ __host__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ __host__ float3 operator*(float lambda, const float3 &a) {
    return make_float3(lambda*a.x, lambda*a.y, lambda*a.z);
}

__device__ __host__ float3 operator/(const float3 &a, float lambda) {
    return make_float3(a.x/lambda, a.y/lambda, a.z/lambda);
}

__device__ __host__ float3 cross(const float3 &a, const float3 &b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __host__ float dot(const float3 &a, const float3 &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ __host__ float norm(const float3 &a) {
    return sqrt(dot(a, a));
}

__device__ __host__ void normalize(float3* a) {
    float n = norm(*a);
    a->x /= n;
    a->y /= n;
    a->z /= n;
}

__device__ __host__ float3 normalize(const float3 &a) {
    return a / norm(a);
}

__device__ __host__ float3 rotate(const float3 &v, const float3 &a, float angle) {
    const float identityMatrix[3][3] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };

    int i, j;
    // Normalize the axis vector
    float3 ax = a / norm(a);

    // Compute the cross product matrix of the axis vector
    float crossProductMatrix[3][3] = {
        {0, -ax.z, ax.y},
        {ax.z, 0, -ax.x},
        {-ax.y, ax.x, 0}
    };

    // Compute the rotation matrix using the Rodrigues rotation formula
    float rotationMatrix[3][3];

    float sinAngle = sin(angle);
    float cosAngle = cos(angle);

    for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) {
        rotationMatrix[i][j] = cosAngle * identityMatrix[i][j] + sinAngle * crossProductMatrix[i][j] + (1 - cosAngle) * ax.x * ax.y;
    }

    // Apply the rotation matrix to the vector
    return make_float3(
        rotationMatrix[0][0] * v.x + rotationMatrix[0][1] * v.y + rotationMatrix[0][2] * v.z,
        rotationMatrix[1][0] * v.x + rotationMatrix[1][1] * v.y + rotationMatrix[1][2] * v.z,
        rotationMatrix[2][0] * v.x + rotationMatrix[2][1] * v.y + rotationMatrix[2][2] * v.z
    );
}

__device__ __host__ float3 copy_float3(const float3 &a) {
    return make_float3(a.x, a.y, a.z);
}


__device__ __host__ Quat operator+(const Quat &a, const Quat &b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

__device__ __host__ Quat operator*(float lambda, const Quat &a) {
    return make_float4(lambda*a.x, lambda*a.y, lambda*a.z, lambda*a.w);
}

__device__ __host__ Quat square(const Quat &a) {
    return make_float4(
        a.x*a.x - a.y*a.y - a.z*a.z - a.w*a.w,
        2.0*a.x*a.y,
        2.0*a.x*a.z,
        2.0*a.x*a.w 
    );
}

__device__ __host__ Quat operator*(const Quat &a, const Quat &b) {
    return make_float4(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
        a.y * b.x + a.x * b.y + a.z * b.w - a.w * b.z, 
        a.z * b.x + a.x * b.z + a.w * b.y - a.y * b.w,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y
    );
}

__device__ __host__ float dot(const Quat &a, const Quat &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

__device__ __host__ float norm(const Quat &a) {
    return sqrt(dot(a, a));
}
