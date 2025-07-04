#include "camera.cuh"



// ----------- boolsToFloat() -----------
//
// determines a direction for two booleans of the same movement
// for exemple : l == GOTOLEFT and r == GOTORIGHT
// if l and not r, returns -1
// if not l and r, returns 1
// if l and r, returns 0
// if not l and not r, returns 0
float boolsToFloat(bool l, bool r) {
    return l && r ? 0 : (r ? 1 : (l ? -1 : 0));
}



// ----------- ndcToDirection() -----------
//
// converts the NDC coordinates of a pixel ([-1,1]x[-1,1]) to a direction for a ray
__device__ float3 ndcToDirection(float x, float y, float wh, Camera &camera) {
    float fov_scale = tan(camera.fov / 2);
    x *= fov_scale * wh;
    y *= fov_scale;

    float3 right = cross(camera.direction, camera.up);
    normalize(&right);

    float3 true_up = cross(right, camera.direction);
    normalize(&true_up);

    return normalize(make_float3(
        camera.direction.x + x * right.x + y * true_up.x,
        camera.direction.y + x * right.y + y * true_up.y,
        camera.direction.z + x * right.z + y * true_up.z
    ));
}



// ----------- cameraUpdateMv() -----------
//
// updates the camera's position and direction according to user input
void cameraUpdateMv(KeyboardMover &kbm, Camera &camera, float dt) {
    // movement
    float3 m = make_float3(
        boolsToFloat(kbm.B, kbm.F),
        boolsToFloat(kbm.L, kbm.R),
        boolsToFloat(kbm.D, kbm.U)
    );
    m = (camera.moveSpeed * dt) * m;
    float3 right = cross(camera.direction, camera.up);
    normalize(&right);
    camera.position.x += camera.direction.x * m.x + right.x * m.y + camera.up.x * m.z;
    camera.position.y += camera.direction.y * m.x + right.y * m.y + camera.up.y * m.z;
    camera.position.z += camera.direction.z * m.x + right.z * m.y + camera.up.z * m.z;

    // rotation
    camera.direction = rotate(camera.direction, right, boolsToFloat(kbm.RD, kbm.RU) * camera.rotaSpeed);
    camera.direction = rotate(camera.direction,  camera.up, boolsToFloat(kbm.RR, kbm.RL) * camera.rotaSpeed);
    camera.up = cross(right, camera.direction);
}
