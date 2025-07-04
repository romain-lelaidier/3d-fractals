#include "fractal.cuh"

#define BOUNDING_RADIUS sqrt(3.0) // radius of a bounding sphere for the set used to accelerate intersection



// ----------- juliaNormal() -----------
//
// estimates the normal vector to the surface of the set
// assumes that point is close enough to the set
__device__ float3 juliaNormal(const Julia &julia, const float3 &p) {
    float eps = julia.geps;
    Quat za = make_float4(p.x + eps, p.y + 0.0, p.z + 0.0, 0.0); // p+e.xyy
    Quat zb = make_float4(p.x - eps, p.y + 0.0, p.z + 0.0, 0.0); // p-e.xyy
    Quat zc = make_float4(p.x + 0.0, p.y + eps, p.z + 0.0, 0.0); // p+e.yxy
    Quat zd = make_float4(p.x + 0.0, p.y - eps, p.z + 0.0, 0.0); // p-e.yxy
    Quat ze = make_float4(p.x + 0.0, p.y + 0.0, p.z + eps, 0.0); // p+e.yyx
    Quat zf = make_float4(p.x + 0.0, p.y + 0.0, p.z - eps, 0.0); // p-e.yyx

    int i;
  	for (i = 0; i < julia.n; i++) {
        za = square(za) + julia.mu; 
        zb = square(zb) + julia.mu; 
        zc = square(zc) + julia.mu; 
        zd = square(zd) + julia.mu; 
        ze = square(ze) + julia.mu; 
        zf = square(zf) + julia.mu; 
    }
    return normalize( make_float3(  log2(dot(za,za))-log2(dot(zb,zb)),
                                    log2(dot(zc,zc))-log2(dot(zd,zd)),
                                    log2(dot(ze,ze))-log2(dot(zf,zf))) );
}



// ----------- rayMarch() -----------
//
// proceeds to march a ray up to the julia set
// returns the distance marched from the origin to the set
// the julia set equation is iterated on z
// if the ray intersects the set, the intersection quaternion is stored in intersection
__device__ float rayMarch(const Julia &julia, const float3 &origin, const float3 &direction) {
    Quat ro = make_float4(   origin.x,    origin.y,    origin.z, 0);
    Quat rd = make_float4(direction.x, direction.y, direction.z, 0);
    Quat z;

    float h = INFINITY; // last step length
    float t = 0.0;      // cumulated step lengths (length of the ray)
    float mdz2, mz2;
    int i;

    while (1) {
        // rayMarch step : estimate the distance from z to the julia set
        z = ro + t * rd;
        
        // for that, we can iterate the function defining the julia set on z, julia.n times
        // see [1] https://dl.acm.org/doi/pdf/10.1145/74333.74363
        mz2 = dot(z, z);    // squared module of z
        mdz2 = 1.0;         // squared module of dz = z_(i) - z_(i-1)

        for(i = 0; i < julia.n; i++) {
            z = square(z) + julia.mu;   // julia set function

            mdz2 = 4.0 * mz2 * mdz2;    // dz -> 2·z·dz, meaning |dz| -> 2·|z|·|dz|
            mz2 = dot(z,z);
            if (mz2 > 4.0) break;   // diverges
        }
        
        h = 0.25 * sqrt(mz2 / mdz2) * log(mz2);

        if (h < julia.epsilon) {
            // ray is close enough : intersection
            return t;
        }
        if (t > 2 * BOUNDING_RADIUS) {
            // diverges
            return INFINITY;
        }

        t += h;
    }
}



// ----------- clampToBoundingSphere() -----------
//
// tries to set the ray's origin at the surface of the bounding sphere.
// updates ray and returns true if the ray indeed intersects, returns false otherwise
__device__ bool clampToBoundingSphere(float *t, const Ray &ray) {
    float B, C, D, d, t0, t1;
    C = dot(ray.origin, ray.origin) - BOUNDING_RADIUS * BOUNDING_RADIUS;
    if (C < 0) return true; // in the sphere : no need to clamp
    B = 2 * dot(ray.origin, ray.direction);
    D = B*B - 4*C;
    if (D > 0.0) {
        d = sqrt(B*B - 4*C);
        t0 = (-B + d) * 0.5;
        t1 = (-B - d) * 0.5;
        *t = min(t0, t1);
        if (*t < -BOUNDING_RADIUS) *t = max(t0, t1);
        if (*t < -BOUNDING_RADIUS) return false;
        return true;
    }
    return false;
}



// ----------- colorizeJulia() -----------
//
// if the ray intersects with the Julia set, updates its color
// and stores the intersection properties (distance, surface normal)
// in the ray object. i should be the index of the object in scene.objs
__device__ void colorizeJulia(const Julia &julia, Ray* ray, int i) {
    float t0 = 0.0;
    bool clamped = clampToBoundingSphere(&t0, *ray);
    if (!clamped) return;   // the ray does not go through the bounding sphere -> no intersection

    float3 ro = ray->origin + t0 * ray->direction;
    float t = rayMarch(julia, ro, ray->direction);

    float dist = t0 + t;
    
    if (dist < INFINITY) {
        // intersection
        
        if (dist > ray->distance) return;
        // the fractal is now the closest object
        ray->i = i;
        ray->distance = dist;
        ro = ro + t * ray->direction;
        ray->normal = juliaNormal(julia, ro);

        // colorization
        float light = pow(dot(ray->normal, make_float3(0, 0, 1)), 0.85);
        light = light + dot(ray->normal, make_float3(0, 0, -1)) / 3;
        clamp(&light, 0, 1);
        ray->color = ((1-light) * make_float3(56, 102, 65) + light * make_float3(167, 201, 87)) / 256;
    }
}