#include "geometry.cuh"



// ----------- colorizeSphere() -----------
//
// if the ray intersects the sphere, this function colorizes the ray
// and makes it bounce on the sphere for reflection (updates origin and direction)
__device__
float colorizeSphere(Sphere* sphere, Ray* ray) {
    float upc, delta;
    float ocDist;   // distance from ray origin to sphere center (euclidean)
    float3 otc;     // vector ray origin to sphere center
    float orcDist;  // distance from ray origin to sphere surface (along the ray)
    float3 sn;      // sphere normal at the intersection

    otc = sphere->center - ray->origin;
    ocDist = norm(otc);

    if (ocDist < sphere->radius) return 0;   // ray origin inside the sphere

    upc = dot(ray->direction, otc);
    delta = upc * upc + sphere->radius * sphere->radius - dot(otc, otc);

    if (delta > 0.0) {
        // intersection
        orcDist = ocDist - sqrt(delta);

        sn.x = ray->origin.x + ray->direction.x * orcDist;
        sn.y = ray->origin.y + ray->direction.y * orcDist;
        sn.z = ray->origin.z + ray->direction.z * orcDist;
        // ray->color.x = abs(sn.x);
        // ray->color.y = abs(sn.y);
        // ray->color.z = abs(sn.z);

        bounceRay(ray, sn);

        float light = dot(ray->direction, make_float3(0, 0, 1));
        ray->color = make_float3(light, light, light);

        return orcDist;
    }

    return INFINITY;
}