#include "camera.cuh"

__device__
void ndcToDirection(vec3* result, double x, double y, double wh, Camera* camera) {
    double fov_scale = tan(camera->fov / 2);
    x *= fov_scale * wh;
    y *= fov_scale;

    vec3 right;
    vec3_cross(&right, &camera->direction, &camera->up);
    vec3_normalize(&right);

    vec3 true_up;
    vec3_cross(&true_up, &right, &camera->direction);
    vec3_normalize(&true_up);

    for (int i = 0; i < 3; i++) {
        result->raw[i] = camera->direction.raw[i] + x * right.raw[i] + y * true_up.raw[i];
    }

    vec3_normalize(result);
}

__device__ __host__ void cam_moveFB(Camera* camera, bool f) {
    double mul = f ? 1.0 : -1.0;
    camera->position.raw[0] += camera->direction.raw[0] * camera->moveSpeed * mul;
    camera->position.raw[1] += camera->direction.raw[1] * camera->moveSpeed * mul;
    camera->position.raw[2] += camera->direction.raw[2] * camera->moveSpeed * mul;
}

void cam_updateMv(Camera* c, double dt) {
    int i;
    // movement
    vec3 m = {
        c->mv.F && c->mv.B ? 0.0 : (c->mv.F ? 1.0 : (c->mv.B ? -1.0 : 0.0)),
        c->mv.R && c->mv.L ? 0.0 : (c->mv.R ? 1.0 : (c->mv.L ? -1.0 : 0.0)),
        c->mv.U && c->mv.D ? 0.0 : (c->mv.U ? 1.0 : (c->mv.D ? -1.0 : 0.0))
    };
    vec3 right;
    vec3_cross(&right, &c->direction, &c->up);
    vec3_normalize(&m);
    for (i = 0; i < 3; i++) {
        c->position.raw[i] += c->direction.raw[i] * c->moveSpeed * dt * m.fields.x;
        c->position.raw[i] +=        right.raw[i] * c->moveSpeed * dt * m.fields.y;
        c->position.raw[i] +=        c->up.raw[i] * c->moveSpeed * dt * m.fields.z;
    }

    // rotation
    double rLR, rUD;
    rLR = c->mv.RR && c->mv.RL ? 0.0 : (c->mv.RR ? -1.0 : (c->mv.RL ? 1.0 : 0.0));
    rUD = c->mv.RD && c->mv.RU ? 0.0 : (c->mv.RD ? -1.0 : (c->mv.RU ? 1.0 : 0.0));
    vec3_rotate(&c->direction, &right, rUD * c->rotaSpeed);
    vec3_rotate(&c->direction, &c->up, rLR * c->rotaSpeed);
}
