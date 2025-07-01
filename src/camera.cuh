#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "math.cuh"

struct Moving {
    bool B, F;
    bool L, R;
    bool U, D;
    bool RL, RR;
    bool RU, RD;
};

struct Camera {
    vec3 position;
    vec3 direction;
    vec3 up;
    double fov;
    double moveSpeed;
    double rotaSpeed;
    Moving mv;
};

void cam_updateMv(Camera* camera, double dt);
__device__ __host__ void cam_moveFB(Camera* camera, bool f);
__device__ void ndcToDirection(vec3* result, double x, double y, double wh, Camera* camera);

#endif