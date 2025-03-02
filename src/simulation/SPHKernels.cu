#include "SPH.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

__global__ void hashesKernel(const glm::vec3* positions, int* hashes, int* indices, glm::ivec3 meshDims, float cellSize, int numParticles) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread idx
    unsigned int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    for (unsigned int i = idx; i < numParticles; i += stride) {
        glm::vec3 pos = positions[i];

        // Compute grid cell index for the particle
        int cellX = static_cast<int>(pos.x / cellSize);
        int cellY = static_cast<int>(pos.y / cellSize);
        int cellZ = static_cast<int>(pos.z / cellSize);

        // Clamp to grid boundaries
        cellX = max(0, min(cellX, meshDims.x - 1));
        cellY = max(0, min(cellY, meshDims.y - 1));
        cellZ = max(0, min(cellZ, meshDims.z - 1));

        int hash = cellX + cellY * meshDims.x + cellZ * meshDims.x * meshDims.y;

        hashes[i] = hash;
        indices[i] = i;
    }
}

__global__ void buildCellStartEndKernel(const int* hashes, int* cellStart, int* cellEnd, int numParticles) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread idx
    unsigned int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    for (unsigned int i = idx; i < numParticles; i += stride) {
        int currentHash = hashes[i];

        // First particle in the cell
        if (i == 0 || hashes[i - 1] != currentHash) {
            cellStart[currentHash] = i;
        }

        // Last particle in the cell
        if (i == numParticles - 1 || hashes[i + 1] != currentHash) {
            cellEnd[currentHash] = i + 1; // end is exclusive
        }
    }
}


void SPHSimulation::computeHashes(int threadsPerBlock, int blocksPerGrid) {
    cellSize = smoothingRadius; // equal to the smoothing radius
    meshDims = glm::ivec3(
            static_cast<int>(boxSize.x / cellSize),
            static_cast<int>(boxSize.y / cellSize),
            static_cast<int>(boxSize.z / cellSize)
    );

    hashesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_hashes, d_indices, meshDims, cellSize, numParticles);
    cudaDeviceSynchronize();

    // Sort particles by hash
    thrust::sort_by_key(
            thrust::device_ptr<int>(d_hashes),
            thrust::device_ptr<int>(d_hashes + numParticles),
            thrust::device_ptr<int>(d_indices)
    );
    printHashTable(d_hashes, d_indices, d_positions, numParticles);

//
//    buildCellStartEndKernel<<<blocksPerGrid, threadsPerBlock>>>(d_hashes, d_cellStart, d_cellEnd, numParticles);
//    cudaDeviceSynchronize();
}

__global__ void neighborListKernel(
        const glm::vec3* positions, // particle positions
        const int* indices,         // sorted indices
        const int* cellStart,       // start of each cell
        const int* cellEnd,         // Input: End of each cell
        glm::ivec3 meshDims,        // Grid dimensions
        float cellSize,             // Cell size
        float smoothingRadius,      // Smoothing radius
        int* neighborList,          // Output: Neighbor list (flattened)
        int* neighborCounts,        // Output: Number of neighbors per particle
        int maxNeighbors,           // Maximum neighbors per particle
        int numParticles            // Number of particles
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread idx
    unsigned int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    for (unsigned int p = idx; p < numParticles; p += stride) {
        glm::vec3 pos = positions[indices[p]];
        int neighborCount = 0;

        int cellX = static_cast<int>(pos.x / cellSize); // grid cell for the particle
        int cellY = static_cast<int>(pos.y / cellSize);
        int cellZ = static_cast<int>(pos.z / cellSize);

        // Iterate over neighboring cells (3x3x3 in 3D)
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    int nx = cellX + dx;
                    int ny = cellY + dy;
                    int nz = cellZ + dz;

                    if (nx < 0 || nx >= meshDims.x || ny < 0 || ny >= meshDims.y || nz < 0 || nz >= meshDims.z)
                        continue; // clamp to grid boundaries

                    int neighborHash = nx + ny * meshDims.x + nz * meshDims.x * meshDims.y;

                    // Iterate over particles in the neighboring cell
                    int start = cellStart[neighborHash];
                    int end = cellEnd[neighborHash];
                    for (int i = start; i < end; ++i) {
                        if (neighborCount >= maxNeighbors) break;

                        int neighborIdx = indices[i];
                        glm::vec3 neighborPos = positions[neighborIdx];

                        float dist = glm::distance(pos, neighborPos);
                        if (dist <= smoothingRadius) { // check distance
                            neighborList[p * maxNeighbors + neighborCount] = neighborIdx;
                            ++neighborCount;
                        }
                    }
                }
            }
        }
        neighborCounts[p] = neighborCount;
    }
}


void SPHSimulation::computeDensityAndPressure() {

}

void SPHSimulation::computeForces() {

}

void SPHSimulation::moveParticles(float deltaTime) {

}

void SPHSimulation::applyBoundaryConditions() {

}


void SPHSimulation::runUpdateKernels(float deltaTime) {
    // Prepare Kernel Configuration
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    int threadsPerBlock = 256;
    int blocksPerGrid = std::min(
            (numParticles + threadsPerBlock - 1) / threadsPerBlock,
            numSMs * 2
    );

    computeHashes(threadsPerBlock, blocksPerGrid);

    int* d_neighborList{};
    int* d_neighborCounts{};
    cudaMalloc(&d_neighborList, numParticles * MAX_NEIGHBORS * sizeof(int));
    cudaMalloc(&d_neighborCounts, numParticles * sizeof(int));
//    neighborListKernel<<<blocksPerGrid, threadsPerBlock>>>(
//            d_positions, d_indices, d_cellStart, d_cellEnd,
//            meshDims, cellSize, smoothingRadius, d_neighborList,
//            d_neighborCounts, MAX_NEIGHBORS, numParticles
//    );
    cudaDeviceSynchronize();
    //printNeighborList(d_neighborList, d_neighborCounts, numParticles, MAX_NEIGHBORS);
}