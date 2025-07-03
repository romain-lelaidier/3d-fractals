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
__device__ float rayMarch(const Julia &julia, const Ray &ray) {
    Quat ro = make_float4(   ray.origin.x,    ray.origin.y,    ray.origin.z, 0);
    Quat rd = make_float4(ray.direction.x, ray.direction.y, ray.direction.z, 0);
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

            // intersection.x = min(intersection.x, abs(z.x));
            // intersection.y = min(intersection.y, abs(z.y));
            // intersection.z = min(intersection.z, abs(z.z));
            // intersection.w = min(intersection.w, dot(z,z));

            mdz2 = 4.0 * mz2 * mdz2;    // dz -> 2·z·dz, meaning |dz| -> 2·|z|·|dz|
            mz2 = dot(z,z);
            if (mz2 > 4.0) break;   // diverges
        }
        
        h = 0.25 * sqrt(mz2 / mdz2) * log(mz2);  // h = 0.5·|z|·log|z|/|z'|

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
__device__ bool clampToBoundingSphere(float3* origin, const float3 &direction) {
    float B, C, D, d, t0, t1, t;
    B = 2 * dot(*origin, direction);
    C = dot(*origin, *origin) - BOUNDING_RADIUS * BOUNDING_RADIUS;
    D = B*B - 4*C;
    if (D > 0.0) {
        d = sqrt(B*B - 4*C);
        t0 = (-B + d) * 0.5;
        t1 = (-B - d) * 0.5;
        t = min(t0, t1);
        if (t > 0.0) {
            *origin = *origin + t * direction;
        }
        return true;
    }
    return false;
}

__device__ float colorizeJulia(const Julia &julia, Ray* ray) {
    bool clamped = clampToBoundingSphere(&ray->origin, ray->direction);
    if (!clamped) return INFINITY; // the ray does not go through the bounding sphere -> no intersection

    float t = rayMarch(julia, *ray);
    if (t < INFINITY) {
        // intersection with the set : colorize the ray

        // reflection
        float3 normal = juliaNormal(julia, ray->origin + t * ray->direction);

        // ray->color = make_float3(1-t, 1-t, 1-t);
        // ray->color = make_float3(abs(normal.x), abs(normal.y), abs(normal.z));
        ray->color = normal;
    }

    return t;
}