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

__device__ __host__
void vec3_cross(vec3* result, vec3* a, vec3* b) {
    result->fields.x = a->fields.y * b->fields.z - a->fields.z * b->fields.y;
    result->fields.y = a->fields.z * b->fields.x - a->fields.x * b->fields.z;
    result->fields.z = a->fields.x * b->fields.y - a->fields.y * b->fields.x;
}

__device__ __host__
void vec3_rotate(vec3* v, vec3* ax, double angle) {
    int i, j;
    // Normalize the axis vector
    vec3_normalize(ax);

    // Compute the cross product matrix of the axis vector
    double crossProductMatrix[3][3] = {
        {0, -ax->fields.z, ax->fields.y},
        {ax->fields.z, 0, -ax->fields.x},
        {-ax->fields.y, ax->fields.x, 0}
    };

    // Compute the outer product of the axis vector
    double outerProduct[3][3];
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            outerProduct[i][j] = ax->fields.x * ax->fields.x + ax->fields.y * ax->fields.y + ax->fields.z * ax->fields.z;
        }
    }

    // Compute the rotation matrix using the Rodrigues rotation formula
    double rotationMatrix[3][3];
    double identityMatrix[3][3] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };

    double sinAngle = sin(angle);
    double cosAngle = cos(angle);

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            rotationMatrix[i][j] = cosAngle * identityMatrix[i][j] +
                                    sinAngle * crossProductMatrix[i][j] +
                                    (1 - cosAngle) * ax->fields.x * ax->fields.y;
        }
    }

    // Apply the rotation matrix to the vector
    vec3 result;
    result.fields.x = rotationMatrix[0][0] * v->fields.x + rotationMatrix[0][1] * v->fields.y + rotationMatrix[0][2] * v->fields.z;
    result.fields.y = rotationMatrix[1][0] * v->fields.x + rotationMatrix[1][1] * v->fields.y + rotationMatrix[1][2] * v->fields.z;
    result.fields.z = rotationMatrix[2][0] * v->fields.x + rotationMatrix[2][1] * v->fields.y + rotationMatrix[2][2] * v->fields.z;

    for (i = 0; i < 3; i++) v->raw[i] = result.raw[i];
}

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
void clamp(double* f, double mn, double mx) {
    *f = min(max(*f, mn), mx);
}
