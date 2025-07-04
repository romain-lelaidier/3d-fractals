#include "geometry.cuh"

#include "math.cuh"



// ----------- colorizeSphere() -----------
//
// if the ray intersects with the sphere, updates its color
// and stores the intersection properties (distance, surface normal)
// in the ray object. i should be the index of the object in scene.objs
__device__ void colorizeSphere(const Sphere &sphere, Ray* ray, int i) {
    float upc, delta;
    float ocDist;   // distance from ray origin to sphere center (euclidean)
    float3 otc;     // vector ray origin to sphere center
    float orcDist;  // distance from ray origin to sphere surface (along the ray)

    otc = sphere.center - ray->origin;
    ocDist = norm(otc);

    if (ocDist < sphere.radius) return;   // ray origin inside the sphere

    upc = dot(ray->direction, otc);
    if (upc < 0) return; // the sphere should not be seen from behind
    
    delta = upc * upc + sphere.radius * sphere.radius - dot(otc, otc);

    if (delta > 0.0) {
        // intersection
        orcDist = ocDist - sqrt(delta);

        if (orcDist > ray->distance) return;
        // the sphere is now the closest object
        ray->i = i;
        ray->distance = orcDist;
        ray->normal = normalize(ray->origin + orcDist * ray->direction - sphere.center);
        ray->color = sphere.color;
    }
}