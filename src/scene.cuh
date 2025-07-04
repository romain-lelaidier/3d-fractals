#ifndef SCENE_CUH
#define SCENE_CUH

#include "geometry.cuh"
#include "fractal.cuh"

enum ObjectType {
    SPHERE,
    JULIA
};

struct Object {
    ObjectType type;
    union {
        Sphere sphere;
        Julia julia;
    };
};

struct Scene {
    int nobjs;
    Object* objs;
};

#endif