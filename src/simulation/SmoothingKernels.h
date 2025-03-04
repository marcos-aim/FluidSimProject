#ifndef SMOOTHINGKERNELS_H
#define SMOOTHINGKERNELS_H

#include <cuda_runtime.h>
#include <glm/gtc/constants.hpp>
#include <cmath>

// Poly6 kernel for density estimation in 3D.
__device__ __forceinline__ float W_Poly6(float dst, float radius)
{
    if (dst < radius)
    {
        float scale = 315.0f / (64.0f * glm::pi<float>() * powf(radius, 9));
        float v = radius * radius - dst * dst;
        return v * v * v * scale;
    }
    return 0.0f;
}

// Spiky kernel (power 3) used for near-density calculations in 3D.
__device__ __forceinline__ float W_SpikyPow3(float dst, float radius)
{
    if (dst < radius)
    {
        float scale = 15.0f / (glm::pi<float>() * powf(radius, 6));
        float v = radius - dst;
        return v * v * v * scale;
    }
    return 0.0f;
}

// Spiky kernel (power 2) used for density estimation in 3D.
__device__ __forceinline__ float W_SpikyPow2(float dst, float radius)
{
    if (dst < radius)
    {
        float scale = 15.0f / (2.0f * glm::pi<float>() * powf(radius, 5));
        float v = radius - dst;
        return v * v * scale;
    }
    return 0.0f;
}

// Derivative of the spiky kernel (power 3) for near-density gradients.
__device__ __forceinline__ float WGrad_SpikyPow3(float dst, float radius)
{
    if (dst <= radius)
    {
        float scale = 45.0f / (glm::pi<float>() * powf(radius, 6));
        float v = radius - dst;
        return -v * v * scale;
    }
    return 0.0f;
}

// Derivative of the spiky kernel (power 2) for density gradients.
__device__ __forceinline__ float WGrad_SpikyPow2(float dst, float radius)
{
    if (dst <= radius)
    {
        float scale = 15.0f / (glm::pi<float>() * powf(radius, 5));
        float v = radius - dst;
        return -v * scale;
    }
    return 0.0f;
}

#endif // SMOOTHINGKERNELS_H
