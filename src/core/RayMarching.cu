#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <Window.h>
#include <glm/glm.hpp>

//OPERATORS
// ─── binary vector +/- ─────────────────────────────────────────
__host__ __device__ __forceinline__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// ─── scalar * / (float3 * k,  k * float3) ─────────────────────
__host__ __device__ __forceinline__ float3 operator*(const float3 &v, float k) {
    return make_float3(v.x * k, v.y * k, v.z * k);
}

__host__ __device__ __forceinline__ float3 operator*(float k, const float3 &v) { return v * k; }

__host__ __device__ __forceinline__ float3 operator*(const float3 &a, const float3 &b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ __forceinline__ float3 operator/(const float3 &v, float k) {
    float inv = 1.0f / k;
    return v * inv;
}

// MATH
__device__ __forceinline__ float3 cross3(const float3 &a, const float3 &b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float dot3(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float len3(const float3 &v) {
    return sqrtf(dot3(v, v));
}

__device__ __forceinline__ float3 normalize3(const float3 &v) {
    float inv = rsqrtf(dot3(v, v) + 1e-20f);
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}

__device__ __forceinline__ float sgn(float v) {
    return (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
}

// closest axis-aligned face normal
__device__ __forceinline__ float3 closestFaceNormal(const float3 &boxSize, const float3 &pLocal) {
    float3 half = make_float3(boxSize.x * 0.5f, boxSize.y * 0.5f, boxSize.z * 0.5f);
    float3 o = make_float3(half.x - fabsf(pLocal.x),
                           half.y - fabsf(pLocal.y),
                           half.z - fabsf(pLocal.z));

    return (o.x < o.y && o.x < o.z)
               ? make_float3(sgn(pLocal.x), 0.0f, 0.0f)
               : (o.y < o.z)
                     ? make_float3(0.0f, sgn(pLocal.y), 0.0f)
                     : make_float3(0.0f, 0.0f, sgn(pLocal.z));
}

// SAMPLING AND INTER
__device__ __forceinline__ float sampleDensity(cudaTextureObject_t densTex, const float3 &pos, const float3 &boxSize) {
    float3 uvw = make_float3(pos.x / boxSize.x,
                             pos.y / boxSize.y,
                             pos.z / boxSize.z);

    uvw.x = fminf(fmaxf(uvw.x, 0.0f), 1.0f);
    uvw.y = fminf(fmaxf(uvw.y, 0.0f), 1.0f);
    uvw.z = fminf(fmaxf(uvw.z, 0.0f), 1.0f);

    return tex3D<float>(densTex, uvw.x, uvw.y, uvw.z);
}

__device__ __forceinline__
float3 calculateNormal(cudaTextureObject_t densTex,
                       const float3 &pos, // point on/near surface (world)
                       const float3 &boxSize, // simulation AABB size
                       float s = 0.05f) // finite-difference offset
{
    /*--- central differences ------------------------------------------------*/
    const float3 ox = make_float3(s, 0, 0), oy = make_float3(0, s, 0), oz = make_float3(0, 0, s);

    float dx = sampleDensity(densTex, pos - ox, boxSize) -
               sampleDensity(densTex, pos + ox, boxSize);
    float dy = sampleDensity(densTex, pos - oy, boxSize) -
               sampleDensity(densTex, pos + oy, boxSize);
    float dz = sampleDensity(densTex, pos - oz, boxSize) -
               sampleDensity(densTex, pos + oz, boxSize);

    float3 nVol = normalize3(make_float3(dx, dy, dz));

    /*--- blend toward axis-aligned face normal near the box walls ----------*/
    const float3 half = boxSize * 0.5f; // centre the box at origin
    const float3 pLocal = pos - half; // local coords

    float3 gap = half - make_float3(fabsf(pLocal.x), fabsf(pLocal.y), fabsf(pLocal.z));
    float faceDist = fminf(fminf(gap.x, gap.y), gap.z);

    const float smoothDst = 0.3f;
    float w = fmaxf(fminf((smoothDst - faceDist) / smoothDst, 1.0f), 0.0f); // 0-1
    w = w * w * w; // ease-in³

    float3 nFace = closestFaceNormal(boxSize, pLocal);

    return normalize3(nVol * (1.0f - w) + nFace * w);
}

__device__ __forceinline__
bool fastFindNextSurface(const float3 &ro,
                         const float3 &rd, // must be unit length
                         cudaTextureObject_t densTex,
                         const float3 &boxSize,
                         float viewStep, // ui.accumulationStepSize
                         float surfMinDensity, // ui.surfaceThreshold
                         float maxDst, // early-out distance
                         /* out */ float3 &outPos,
                         /* out */ float3 &outN,
                         /* out */ bool &entering) {
    /* ------------ intersect ray with AABB ------------------------------ */
    const float3 inv = make_float3(1.f / rd.x, 1.f / rd.y, 1.f / rd.z);
    const float3 t0 = (make_float3(0.f, 0.f, 0.f) - ro) * inv;
    const float3 t1 = (boxSize - ro) * inv;

    const float3 tmin = make_float3(fminf(t0.x, t1.x),
                                    fminf(t0.y, t1.y),
                                    fminf(t0.z, t1.z));
    const float3 tmax = make_float3(fmaxf(t0.x, t1.x),
                                    fmaxf(t0.y, t1.y),
                                    fmaxf(t0.z, t1.z));

    float tNear = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
    float tFar = fminf(fminf(tmax.x, tmax.y), tmax.z);
    if (tFar < fmaxf(tNear, 0.f)) return false; // ray misses box

    /* ------------ first sample just inside the box --------------------- */
    float3 p = ro + rd * (tNear + 1e-3f);
    float dst = 0.f;
    float dHere = sampleDensity(densTex, p, boxSize);
    bool inside = (dHere > surfMinDensity);
    bool everInside = inside;

    /* ------------ adaptive march --------------------------------------- */
    while (dst < (tFar - tNear) && dst < maxDst) {
        float step = inside ? viewStep : viewStep * 2.f; // coarse in air
        step = fminf(step, tFar - tNear - dst); // stay in box
        step = fminf(step, maxDst - dst); // caller clamp

        /* ahead sample -------------------------------------------------- */
        const float3 nextP = p + rd * step;
        const float dNext = sampleDensity(densTex, nextP, boxSize);

        const float sHere = dHere - surfMinDensity;
        const float sNext = dNext - surfMinDensity;

        /* ------------ iso–surface crossed? ---------------------------- */
        if (sHere * sNext < 0.f) {
            /* two refinements (4 extra samples total) ------------------ */
            float3 a = p, b = nextP;
            float signA = sHere;

            for (int i = 0; i < 2; ++i) {
                float3 mid = (a + b) * 0.5f;
                float signM = sampleDensity(densTex, mid, boxSize) - surfMinDensity;
                if (signM * signA > 0.f) {
                    a = mid;
                    signA = signM;
                } else { b = mid; }
            }

            outPos = (a + b) * 0.5f;
            const float normalStep = viewStep * 0.5f; // tighter finite diff
            outN = calculateNormal(densTex, outPos, boxSize, normalStep);
            entering = !inside && (sNext > 0.f);
            return true;
        }

        /* advance ------------------------------------------------------- */
        everInside |= (sNext > 0.f);
        p = nextP;
        dHere = dNext;
        inside = (sNext > 0.f);
        dst += step;
    }

    /* ------------ no surface hit --------------------------------------- */
    if (!everInside || !inside) return false; // ended in air

    /* ------------ ray left box while inside fluid ---------------------- */
    outPos = ro + rd * (tFar - 1e-3f);
    outN = closestFaceNormal(boxSize, outPos - boxSize * 0.5f);
    entering = false; // we were exiting
    return true;
}


__global__ void SurfaceDebugKernel(cudaSurfaceObject_t surface,
                                   int width, int height,
                                   float3 camPos, float3 camFwd,
                                   float3 camRight, float3 camUp,
                                   float tanHalfFOV, float aspect,
                                   cudaTextureObject_t densTex,
                                   float3 boxSize,
                                   float marchStep,
                                   float surfThresh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    /*--- pixel → ray -------------------------------------------------------*/
    float u = (x + 0.5f) / width;
    float v = (y + 0.5f) / height;

    float px = (2.0f * u - 1.0f) * aspect * tanHalfFOV;
    float py = (1.0f - 2.0f * v) * tanHalfFOV;

    float3 rd = normalize3(camFwd + camRight * px + camUp * py);
    float3 ro = camPos;

    /*--- query next surface -----------------------------------------------*/
    float3 hitPos, hitN;
    bool entering;
    bool found = fastFindNextSurface(ro, rd,
                                     densTex, boxSize,
                                     marchStep, surfThresh,
                                     1e20f, // maxDst
                                     hitPos, hitN, entering);

    uchar4 c = found
                   ? make_uchar4(0, 0, 255, 255) // blue
                   : make_uchar4(0, 0, 0, 255); // black

    surf2Dwrite(c, surface, x * sizeof(uchar4), height - 1 - y);
}

extern "C"
void launchSurfaceDebug(cudaSurfaceObject_t surf,
                        const CameraCUDAParams &cam,
                        cudaTextureObject_t densTex,
                        const float3 &boxSize,
                        float marchStep,
                        float surfMinDensity,
                        float maxDst) {
    int width = cam.width;
    int height = cam.height;

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    SurfaceDebugKernel<<<grid,block>>>(
        surf,
        cam.width, cam.height,
        cam.pos, cam.forward, cam.right, cam.up,
        cam.tanHalfFOV, cam.aspect,
        densTex,
        boxSize,
        marchStep,
        surfMinDensity // ← passes the new parameter
    );
    cudaDeviceSynchronize();
}


//TESTING
// Slab‐method AABB intersection without operators
// Slab-method AABB intersection  (unchanged math; still no element-wise “*” overload)
__device__ bool intersectAABB(const float3 &ro,
                              const float3 &rd,
                              const float3 &bmin,
                              const float3 &bmax) {
    float3 inv = make_float3(1.0f / rd.x,
                             1.0f / rd.y,
                             1.0f / rd.z);

    float3 t0s = make_float3((bmin.x - ro.x) * inv.x,
                             (bmin.y - ro.y) * inv.y,
                             (bmin.z - ro.z) * inv.z);

    float3 t1s = make_float3((bmax.x - ro.x) * inv.x,
                             (bmax.y - ro.y) * inv.y,
                             (bmax.z - ro.z) * inv.z);

    float3 tmin = make_float3(fminf(t0s.x, t1s.x),
                              fminf(t0s.y, t1s.y),
                              fminf(t0s.z, t1s.z));

    float3 tmax = make_float3(fmaxf(t0s.x, t1s.x),
                              fmaxf(t0s.y, t1s.y),
                              fmaxf(t0s.z, t1s.z));

    float tNear = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
    float tFar = fminf(fminf(tmax.x, tmax.y), tmax.z);

    return tFar >= fmaxf(tNear, 0.0f);
}

/*------------------------------------------------------------*/
__global__ void AABBTestKernel(cudaSurfaceObject_t surface,
                               int width, int height,
                               float3 camPos, float3 camForward,
                               float3 camRight, float3 camUp,
                               float tanHalfFOV, float aspect,
                               float3 boxMin, float3 boxMax) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    /* 1) NDC */
    float u = (x + 0.5f) / width;
    float v = (y + 0.5f) / height;

    /* 2) Screen-plane offsets */
    float px = (2.0f * u - 1.0f) * aspect * tanHalfFOV;
    float py = (1.0f - 2.0f * v) * tanHalfFOV;

    /* 3) Ray direction (vector overloads) */
    float3 dirRight = camRight * px;
    float3 dirUp = camUp * py;
    float3 rd = normalize3(camForward + dirRight + dirUp);
    float3 ro = camPos; // origin

    /* 4) AABB hit test */
    bool hit = intersectAABB(ro, rd, boxMin, boxMax);

    uchar4 c = hit
                   ? make_uchar4(255, 255, 255, 255)
                   : make_uchar4(0, 0, 0, 255);

    surf2Dwrite(c, surface, x * sizeof(uchar4), height - 1 - y);
}


extern "C"
void launchAABBTestKernel(cudaSurfaceObject_t surface, const CameraCUDAParams &cam, const UserInput &ui) {
    // extract screen dims
    int width = cam.width;
    int height = cam.height;

    // build launch grid
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // call the same device kernel, but now boxMin is always (0,0,0)
    AABBTestKernel<<<grid, block>>>(
        surface,
        width, height,
        cam.pos,
        cam.forward,
        cam.right,
        cam.up,
        cam.tanHalfFOV,
        cam.aspect,
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(ui.boxSizeX, ui.boxSizeY, ui.boxSizeZ)
    );

    cudaDeviceSynchronize();
}
