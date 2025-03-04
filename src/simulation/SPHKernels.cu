// SPHKernels.cpp
// Adapted to work with our new SPH2 data structures, hash functions, and smoothing kernels.
// This file implements the following kernels using our standard grid–stride loop:
//   1. ExternalForcesKernel – applies gravity and predicts positions.
//   2. UpdateSpatialHashKernel – computes each particle’s grid cell, hash, and key.
//   3. BuildCellStartIndicesKernel – computes offsets into the sorted indices array.
//   4. CalculateDensitiesKernel – computes density and near–density using a 27–cell neighborhood.
//   5. CalculatePressureForceKernel – computes pressure forces and updates velocities.
//   6. CalculateViscosityKernel – computes viscosity forces and updates velocities.
//   7. UpdatePositionsKernel – updates positions and resolves collisions against the simulation box.

#include "SPH.h"
#include "HashTable.h"         // Provides: GetCell3D, HashCell3D, KeyFromHash
#include "SmoothingKernels.h"  // Provides: W_Poly6, W_SpikyPow3, WGrad_SpikyPow3, WGrad_SpikyPow2
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cmath>

//-------------------------------------------------------------------
// Kernel 1: External Forces & Prediction
//-------------------------------------------------------------------
__global__ void ExternalForcesKernel(
    float3* positions,
    float3* predictedPositions,
    float3* velocities,
    int numParticles,
    float gravity,
    float deltaTime)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < numParticles; idx += stride)
    {
        // Apply gravity (only along Y axis)
        velocities[idx].y += gravity * deltaTime;
        // Predict new position (using a fixed time factor of 1/120 as in HLSL)
        predictedPositions[idx] = make_float3(
            positions[idx].x + velocities[idx].x * (1.0f / 120.0f),
            positions[idx].y + velocities[idx].y * (1.0f / 120.0f),
            positions[idx].z + velocities[idx].z * (1.0f / 120.0f)
        );
    }
}

//-------------------------------------------------------------------
// Kernel 2: Update Spatial Hash
//-------------------------------------------------------------------
__global__ void UpdateSpatialHashKernel(
    const float3* predictedPositions,
    uint3* indices,
    unsigned int* startIndices,
    int numParticles,
    float smoothingRadius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(; idx < numParticles; idx += stride)
    {
        // Initialize startIndices to a default value (numParticles)
        startIndices[idx] = numParticles;

        int3 cell = GetCell3D(predictedPositions[idx], smoothingRadius);
        unsigned int hash = HashCell3D(cell);
        unsigned int key = KeyFromHash(hash, numParticles);

        // Store the original index, hash, and key in the indices array.
        indices[idx] = make_uint3(idx, hash, key);
    }
}

//-------------------------------------------------------------------
// Kernel 3: Build Cell Start Indices (Offsets)
//-------------------------------------------------------------------
__global__ void BuildCellStartIndicesKernel(
    const uint3* sortedIndices,
    unsigned int* offsets,
    int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles)
    {
        unsigned int currentKey = sortedIndices[idx].z;
        // If this is the first element or the key changes from the previous element...
        if (idx == 0 || sortedIndices[idx - 1].z != currentKey)
        {
            offsets[currentKey] = idx;
        }
    }
}

//-------------------------------------------------------------------
// Kernel 4: Calculate Densities
//-------------------------------------------------------------------
__global__ void CalculateDensitiesKernel(
    const float3* predictedPositions,
    const uint3* indices,
    const unsigned int* startIndices,
    float2* densities,    // (density, nearDensity)
    int numParticles,
    float smoothingRadius)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(; idx < numParticles; idx += stride)
    {
        float3 pos = predictedPositions[idx];
        int3 originCell = GetCell3D(pos, smoothingRadius);
        float sqrRadius = smoothingRadius * smoothingRadius;
        float density = 0.0f;
        float nearDensity = 0.0f;

        // Loop over 27 neighboring cells using the constant offsets3D array.
        for (int i = 0; i < 27; i++)
        {
            int3 cell = make_int3(originCell.x + offsets3D[i].x,
                                  originCell.y + offsets3D[i].y,
                                  originCell.z + offsets3D[i].z);
            unsigned int hash = HashCell3D(cell);
            unsigned int key = KeyFromHash(hash, numParticles);
            unsigned int start = startIndices[key];

            if (start == numParticles)
                continue;

            unsigned int curr = start;
            while (curr < numParticles)
            {
                uint3 indexData = indices[curr];
                if (indexData.z != key)
                    break;
                if (indexData.y != hash)
                {
                    curr++;
                    continue;
                }
                unsigned int neighborIndex = indexData.x;
                float3 neighborPos = predictedPositions[neighborIndex];
                float3 offset = make_float3(
                    neighborPos.x - pos.x,
                    neighborPos.y - pos.y,
                    neighborPos.z - pos.z);
                float distSqr = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;
                if (distSqr > sqrRadius)
                {
                    curr++;
                    continue;
                }
                float r = sqrtf(distSqr);
                density += W_SpikyPow2(r, smoothingRadius);
                nearDensity += W_SpikyPow3(r, smoothingRadius);

                curr++;
            }
        }

        densities[idx] = make_float2(density, nearDensity);
    }
}

//-------------------------------------------------------------------
// Kernel 5: Calculate Pressure Force & Update Velocities
//-------------------------------------------------------------------
__global__ void CalculatePressureForceKernel(
    const float3* predictedPositions,
    float3* velocities,
    const float2* densities,
    const uint3* indices,
    const unsigned int* startIndices,
    int numParticles,
    float smoothingRadius,
    float pMult,
    float nearPMult,
    float deltaTime,
    float restingDensity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(; idx < numParticles; idx += stride)
    {
        float density = densities[idx].x;
        float nearDensity = densities[idx].y;

        float pressure = (density - restingDensity) * pMult;
        float nearPressure = nearDensity * nearPMult;

        float3 pos = predictedPositions[idx];
        int3 originCell = GetCell3D(pos, smoothingRadius);
        float sqrRadius = smoothingRadius * smoothingRadius;

        float3 pressureForce = make_float3(0.0f, 0.0f, 0.0f);

        // Loop over 27 neighboring cells using offsets3D.
        for (int i = 0; i < 27; i++)
        {
            int3 cell = make_int3(originCell.x + offsets3D[i].x,
                                  originCell.y + offsets3D[i].y,
                                  originCell.z + offsets3D[i].z);
            unsigned int hash = HashCell3D(cell);
            unsigned int key = KeyFromHash(hash, numParticles);
            unsigned int start = startIndices[key];

            if (start == numParticles)
                continue;

            unsigned int curr = start;
            while (curr < numParticles)
            {
                uint3 indexData = indices[curr];
                if (indexData.z != key)
                    break;
                if (indexData.y != hash)
                {
                    curr++;
                    continue;
                }
                unsigned int neighborIndex = indexData.x;
                if (neighborIndex == idx)
                {
                    curr++;
                    continue;
                }
                float3 neighborPos = predictedPositions[neighborIndex];
                float3 offset = make_float3(
                    neighborPos.x - pos.x,
                    neighborPos.y - pos.y,
                    neighborPos.z - pos.z);
                float distSqr = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;
                if (distSqr > sqrRadius)
                {
                    curr++;
                    continue;
                }
                float r = sqrtf(distSqr);
                float3 dir = (r > 0.0f) ? make_float3(offset.x / r, offset.y / r, offset.z / r)
                                       : make_float3(0.0f, 1.0f, 0.0f);

                float neighborDensity = densities[neighborIndex].x;
                float neighborNearDensity = densities[neighborIndex].y;
                float neighborPressure = (neighborDensity - restingDensity) * pMult;
                float neighborNearPressure = neighborNearDensity * nearPMult;

                float sharedPressure = (pressure + neighborPressure) * 0.5f;
                float sharedNearPressure = (nearPressure + neighborNearPressure) * 0.5f;

                float gradVal = WGrad_SpikyPow3(r, smoothingRadius);
                float nearGradVal = WGrad_SpikyPow2(r, smoothingRadius);

                pressureForce.x += dir.x * (gradVal * sharedPressure / neighborDensity + nearGradVal * sharedNearPressure / neighborNearDensity);
                pressureForce.y += dir.y * (gradVal * sharedPressure / neighborDensity + nearGradVal * sharedNearPressure / neighborNearDensity);
                pressureForce.z += dir.z * (gradVal * sharedPressure / neighborDensity + nearGradVal * sharedNearPressure / neighborNearDensity);

                curr++;
            }
        }

        // Convert force to acceleration and update velocity.
        float3 acceleration = make_float3(pressureForce.x / density, pressureForce.y / density, pressureForce.z / density);
        velocities[idx].x += acceleration.x * deltaTime;
        velocities[idx].y += acceleration.y * deltaTime;
        velocities[idx].z += acceleration.z * deltaTime;
    }
}

//-------------------------------------------------------------------
// Kernel 6: Calculate Viscosity & Update Velocities
//-------------------------------------------------------------------
__global__ void CalculateViscosityKernel(
    const float3* predictedPositions,
    float3* velocities,
    const uint3* indices,
    const unsigned int* startIndices,
    int numParticles,
    float smoothingRadius,
    float viscosityStrength,
    float deltaTime)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(; idx < numParticles; idx += stride)
    {
        float3 pos = predictedPositions[idx];
        int3 originCell = GetCell3D(pos, smoothingRadius);
        float sqrRadius = smoothingRadius * smoothingRadius;

        float3 viscosityForce = make_float3(0.0f, 0.0f, 0.0f);
        float3 selfVelocity = velocities[idx];

        // Loop over 27 neighboring cells using offsets3D.
        for (int i = 0; i < 27; i++)
        {
            int3 cell = make_int3(originCell.x + offsets3D[i].x,
                                  originCell.y + offsets3D[i].y,
                                  originCell.z + offsets3D[i].z);
            unsigned int hash = HashCell3D(cell);
            unsigned int key = KeyFromHash(hash, numParticles);
            unsigned int start = startIndices[key];
            if (start == numParticles)
                continue;

            unsigned int curr = start;
            while (curr < numParticles)
            {
                uint3 indexData = indices[curr];
                if (indexData.z != key)
                    break;
                if (indexData.y != hash)
                {
                    curr++;
                    continue;
                }
                unsigned int neighborIndex = indexData.x;
                if (neighborIndex == idx)
                {
                    curr++;
                    continue;
                }
                float3 neighborPos = predictedPositions[neighborIndex];
                float3 offset = make_float3(
                    neighborPos.x - pos.x,
                    neighborPos.y - pos.y,
                    neighborPos.z - pos.z);
                float distSqr = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;
                if (distSqr > sqrRadius)
                {
                    curr++;
                    continue;
                }
                float r = sqrtf(distSqr);
                float kernelVal = W_Poly6(r, smoothingRadius);
                float3 neighborVelocity = velocities[neighborIndex];
                viscosityForce.x += (neighborVelocity.x - selfVelocity.x) * kernelVal;
                viscosityForce.y += (neighborVelocity.y - selfVelocity.y) * kernelVal;
                viscosityForce.z += (neighborVelocity.z - selfVelocity.z) * kernelVal;

                curr++;
            }
        }

        velocities[idx].x += viscosityForce.x * viscosityStrength * deltaTime;
        velocities[idx].y += viscosityForce.y * viscosityStrength * deltaTime;
        velocities[idx].z += viscosityForce.z * viscosityStrength * deltaTime;
    }
}

//-------------------------------------------------------------------
// Kernel 7: Update Positions & Resolve Collisions
//-------------------------------------------------------------------
__global__ void UpdatePositionsKernel(
    float3* positions,
    float3* velocities,
    int numParticles,
    float deltaTime,
    float3 boxSize,
    float collisionDamping)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(; idx < numParticles; idx += stride)
    {
        // Update positions based on velocities.
        positions[idx].x += velocities[idx].x * deltaTime;
        positions[idx].y += velocities[idx].y * deltaTime;
        positions[idx].z += velocities[idx].z * deltaTime;

        // Simple collision resolution against axis–aligned box boundaries.
        if (positions[idx].x < 0.0f)
        {
            positions[idx].x = 0.0f;
            velocities[idx].x = -velocities[idx].x * collisionDamping;
        }
        else if (positions[idx].x > boxSize.x)
        {
            positions[idx].x = boxSize.x;
            velocities[idx].x = -velocities[idx].x * collisionDamping;
        }

        if (positions[idx].y < 0.0f)
        {
            positions[idx].y = 0.0f;
            velocities[idx].y = -velocities[idx].y * collisionDamping;
        }
        else if (positions[idx].y > boxSize.y)
        {
            positions[idx].y = boxSize.y;
            velocities[idx].y = -velocities[idx].y * collisionDamping;
        }

        if (positions[idx].z < 0.0f)
        {
            positions[idx].z = 0.0f;
            velocities[idx].z = -velocities[idx].z * collisionDamping;
        }
        else if (positions[idx].z > boxSize.z)
        {
            positions[idx].z = boxSize.z;
            velocities[idx].z = -velocities[idx].z * collisionDamping;
        }
    }
}

struct CompareUint3 {
    __device__ bool operator()(const uint3 &a, const uint3 &b) const {
        return a.z < b.z;
    }
};

//-------------------------------------------------------------------
// Host Function: runUpdateKernels
//-------------------------------------------------------------------
void SPHSimulation::runUpdateKernels(float deltaTime)
{
    int threadsPerBlock = 256;
    int blocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    // 1. External forces & prediction.
    ExternalForcesKernel<<<blocks, threadsPerBlock>>>(d_positions, d_predicted_positions, d_velocities, numParticles, gravity, deltaTime);
    cudaDeviceSynchronize();

    // 2. Update spatial hash: compute each particle's cell, hash, and key.
    UpdateSpatialHashKernel<<<blocks, threadsPerBlock>>>(d_predicted_positions, d_indices, d_start_indices, numParticles, smoothingRadius);
    cudaDeviceSynchronize();

    // 3. Sort the indices based on the key (z component) using Thrust (radix sort).
    {
        thrust::device_ptr<uint3> dev_indices(d_indices);
        thrust::sort(dev_indices, dev_indices + numParticles, CompareUint3());
    }

    // 4. Build cell start indices (offsets) from the sorted indices.
    BuildCellStartIndicesKernel<<<blocks, threadsPerBlock>>>(d_indices, d_start_indices, numParticles);
    cudaDeviceSynchronize();

    // 5. Calculate densities (and near densities) using the neighbor search.
    CalculateDensitiesKernel<<<blocks, threadsPerBlock>>>(d_predicted_positions, d_indices, d_start_indices, d_densities, numParticles, smoothingRadius);
    cudaDeviceSynchronize();

    // 6. Calculate pressure forces and update velocities.
    CalculatePressureForceKernel<<<blocks, threadsPerBlock>>>(d_predicted_positions, d_velocities, d_densities, d_indices, d_start_indices,
                                                              numParticles, smoothingRadius, pMult, nearPMult, deltaTime, restingDensity);
    cudaDeviceSynchronize();

    // 7. Calculate viscosity forces and update velocities.
    CalculateViscosityKernel<<<blocks, threadsPerBlock>>>(d_predicted_positions, d_velocities, d_indices, d_start_indices,
                                                          numParticles, smoothingRadius, viscosityMult, deltaTime);
    cudaDeviceSynchronize();

    // 8. Update positions and resolve collisions.
    UpdatePositionsKernel<<<blocks, threadsPerBlock>>>(d_positions, d_velocities, numParticles, deltaTime, boxSize, collisionDamping);
    cudaDeviceSynchronize();
}