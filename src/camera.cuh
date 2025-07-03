#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "math.cuh"

struct Moving {
    bool B, F;
    bool L, R;
    bool U, D;
    bool RL, RR;
    bool RU, RD;
    bool P[9];
};

struct Camera {
    float3 position;
    float3 direction;
    float3 up;
    float fov;
    float moveSpeed;
    float rotaSpeed;
    Moving mv;
};

float boolsToFloat(bool l, bool r);
void cam_updateMv(Camera &camera, float dt);
__device__ __host__ void cam_moveFB(Camera &camera, bool f);
__device__ float3 ndcToDirection(float x, float y, float wh, Camera &camera);

#endif