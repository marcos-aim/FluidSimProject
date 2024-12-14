#include "SpatialHashTable.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cmath>
#include <iostream>

// Neighbor search kernel
__global__ void neighborSearchKernel(
    const float3* positions,
    const int* cellStart,
    const int* cellEnd,
    const int* particleIds,
    int* neighborList,
    int numParticles,
    float radius,
    float cellSize,
    int3 gridSize,
    int maxNeighbors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float3 pos = positions[idx];
    int3 cell = make_int3(
        floorf(pos.x / cellSize),
        floorf(pos.y / cellSize),
        floorf(pos.z / cellSize)
    );

    int neighborCount = 0;

    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            for (int z = -1; z <= 1; ++z) {
                int3 neighborCell = cell + make_int3(x, y, z);

                // Ensure neighborCell is within bounds
                if (neighborCell.x < 0 || neighborCell.y < 0 || neighborCell.z < 0 ||
                    neighborCell.x >= gridSize.x || neighborCell.y >= gridSize.y || neighborCell.z >= gridSize.z) {
                    continue;
                }

                int neighborHash = neighborCell.x +
                                   neighborCell.y * gridSize.x +
                                   neighborCell.z * gridSize.x * gridSize.y;

                int start = cellStart[neighborHash];
                int end = cellEnd[neighborHash];
                if (start < 0 || end < 0) continue;

                for (int i = start; i < end; ++i) {
                    if (neighborCount >= maxNeighbors) break;

                    int neighborIdx = particleIds[i];
                    float3 neighborPos = positions[neighborIdx];

                    // Compute distance
                    float dx = pos.x - neighborPos.x;
                    float dy = pos.y - neighborPos.y;
                    float dz = pos.z - neighborPos.z;
                    float distSq = dx * dx + dy * dy + dz * dz;

                    if (distSq < radius * radius) {
                        neighborList[idx * maxNeighbors + neighborCount] = neighborIdx;
                        neighborCount++;
                    }
                }
            }
        }
    }

    // Fill remaining neighbor slots with -1
    for (int i = neighborCount; i < maxNeighbors; ++i) {
        neighborList[idx * maxNeighbors + i] = -1;
    }
}

// Implementation of SpatialHashTable methods
SpatialHashTable::SpatialHashTable(int gridResolution, float cellSize, int maxParticles)
    : gridResolution(gridResolution), cellSize(cellSize), maxParticles(maxParticles) {
    tableSize = gridResolution * gridResolution * gridResolution;

    cudaMalloc(&d_hashKeys, maxParticles * sizeof(int));
    cudaMalloc(&d_particleIds, maxParticles * sizeof(int));
    cudaMalloc(&d_cellStart, tableSize * sizeof(int));
    cudaMalloc(&d_cellEnd, tableSize * sizeof(int));
}

SpatialHashTable::~SpatialHashTable() {
    cudaFree(d_hashKeys);
    cudaFree(d_particleIds);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);
}

void SpatialHashTable::findNeighbors(
    const float* d_positions,
    int numParticles,
    float searchRadius,
    int* d_neighborList,
    int maxNeighbors
) {
    int3 gridSize = make_int3(gridResolution, gridResolution, gridResolution);
    int blockSize = 256;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;

    neighborSearchKernel<<<numBlocks, blockSize>>>(
        reinterpret_cast<const float3*>(d_positions),
        d_cellStart, d_cellEnd, d_particleIds,
        d_neighborList, numParticles, searchRadius,
        cellSize, gridSize, maxNeighbors
    );
    cudaDeviceSynchronize();
}
