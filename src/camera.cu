#include "camera.cuh"

float boolsToFloat(bool l, bool r) {
    return l && r ? 0 : (r ? 1 : (l ? -1 : 0));
}

__device__
float3 ndcToDirection(float x, float y, float wh, Camera &camera) {
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

__device__ __host__ void cam_moveFB(Camera &camera, bool f) {
    float mul = f ? 1.0 : -1.0;
    camera.position.x += camera.direction.x * camera.moveSpeed * mul;
    camera.position.y += camera.direction.y * camera.moveSpeed * mul;
    camera.position.z += camera.direction.z * camera.moveSpeed * mul;
}

void cam_updateMv(Camera &c, float dt) {
    // movement
    float3 m = make_float3(
        boolsToFloat(c.mv.B, c.mv.F),
        boolsToFloat(c.mv.L, c.mv.R),
        boolsToFloat(c.mv.D, c.mv.U)
    );
    m = (c.moveSpeed * dt) * m;
    float3 right = cross(c.direction, c.up);
    normalize(&right);
    c.position.x += c.direction.x * m.x + right.x * m.y + c.up.x * m.z;
    c.position.y += c.direction.y * m.x + right.y * m.y + c.up.y * m.z;
    c.position.z += c.direction.z * m.x + right.z * m.y + c.up.z * m.z;

    // rotation
    c.direction = rotate(c.direction, right, boolsToFloat(c.mv.RD, c.mv.RU) * c.rotaSpeed);
    c.direction = rotate(c.direction,  c.up, boolsToFloat(c.mv.RR, c.mv.RL) * c.rotaSpeed);
    c.up = cross(right, c.direction);
}
