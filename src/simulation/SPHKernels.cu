#include <glm/ext/scalar_constants.hpp>

#include "SPH.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#define BOX_COLLISION_OFFSET 0.01f
#define BOX_COLLISION_ELASTICITY 0.8f


__global__ void hashesKernel(const glm::vec3* positions, int* hashes, int* indices, glm::ivec3 meshDims, float cellSize, int numParticles) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread idx
    const unsigned int stride = blockDim.x * gridDim.x; // total number of threads in the grid
    for (unsigned int i = idx; i < numParticles; i += stride) {
        const glm::vec3 pos = positions[i];

        // Compute grid cell index for the particle
        int cellX = static_cast<int>(pos.x / cellSize);
        int cellY = static_cast<int>(pos.y / cellSize);
        int cellZ = static_cast<int>(pos.z / cellSize);

        // Clamp to grid boundaries
        cellX = max(0, min(cellX, meshDims.x - 1));
        cellY = max(0, min(cellY, meshDims.y - 1));
        cellZ = max(0, min(cellZ, meshDims.z - 1));

        const int hash = cellX + cellY * meshDims.x + cellZ * meshDims.x * meshDims.y;

        hashes[i] = hash;
        indices[i] = static_cast<int>(i);
    }
}

__global__ void buildCellStartEndKernel(const int* hashes, unsigned int* cellStart, unsigned int* cellEnd, const int numParticles) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < numParticles; i += stride) {
        int currentHash = hashes[i];
        // First particle in the cell: either i==0 or hash changes from previous element.
        if (i == 0 || hashes[i - 1] != currentHash) {
            cellStart[currentHash] = i;
        }
        // Last particle in the cell: either at the end or next hash is different.
        if (i == numParticles - 1 || hashes[i + 1] != currentHash) {
            cellEnd[currentHash] = i + 1; // exclusive end index
        }
    }
}

__global__ void reorderParticlesKernel(
    const int* indices,                // Sorted indices (from computeHashes)
    const glm::vec3* oldPositions,       // Unsorted positions
    glm::vec3* newPositions,             // Output: Reordered positions
    int numParticles
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < numParticles; i += stride) {
        int sortedIndex = indices[i];
        newPositions[i] = oldPositions[sortedIndex];
    }
}

void SPHSimulation::computeHashes(int threadsPerBlock, int blocksPerGrid) {
    cellSize = smoothingRadius; // equal to the smoothing radius

    hashesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_positions, d_hashes, d_indices, meshDims, cellSize, numParticles);
    cudaDeviceSynchronize();

    // Sort particles by hash
    thrust::sort_by_key(
            thrust::device_ptr<int>(d_hashes),
            thrust::device_ptr<int>(d_hashes + numParticles),
            thrust::device_ptr<int>(d_indices)
    );

    // Reorder positions
    reorderParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_indices, d_positions, d_positions, numParticles);
    // Similarly, reorder d_velocities and d_prevPositions if needed.
    cudaDeviceSynchronize();


    cudaMemset(d_cellStart, 0xFF, sizeof(unsigned int) * (meshDims.x * meshDims.y * meshDims.z));
    cudaMemset(d_cellEnd, 0, sizeof(unsigned int) * (meshDims.x * meshDims.y * meshDims.z));

    buildCellStartEndKernel<<<blocksPerGrid, threadsPerBlock>>>(d_hashes, d_cellStart, d_cellEnd, numParticles);
    cudaDeviceSynchronize();
}

__global__ void neighborListKernel(
    const glm::vec3* positions,  // Particle positions
    const int* indices,          // Sorted indices from computeHashes
    const unsigned int* cellStart, // Start indices for each cell
    const unsigned int* cellEnd,   // End indices (exclusive) for each cell
    glm::ivec3 meshDims,         // Grid dimensions (cells in x,y,z)
    float cellSize,              // Cell size (usually equal to h)
    float smoothingRadius,       // Smoothing radius h
    int* neighborList,           // Output: flattened neighbor list per particle
    int* neighborCounts,         // Output: neighbor counts per particle (sorted order)
    int maxNeighbors,            // Maximum allowed neighbors per particle
    int numParticles             // Total number of particles
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Loop over each particle (sorted order)
    for (unsigned int p = idx; p < numParticles; p += stride) {
        // Get original particle index from sorted order
        const unsigned int particleIndex = indices[p];
        glm::vec3 pos = positions[particleIndex];
        int count = 0;

        // Determine grid cell for this particle
        int cellX = static_cast<int>(pos.x / cellSize);
        int cellY = static_cast<int>(pos.y / cellSize);
        int cellZ = static_cast<int>(pos.z / cellSize);

        // Loop over the 3x3x3 neighborhood of cells
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nX = cellX + dx;
                    int nY = cellY + dy;
                    int nZ = cellZ + dz;

                    // Skip out-of-bound cells
                    if (nX < 0 || nX >= meshDims.x || nY < 0 || nY >= meshDims.y || nZ < 0 || nZ >= meshDims.z)
                        continue;

                    // Compute 1D cell index (row-major order)
                    int neighborCell = nX + nY * meshDims.x + nZ * meshDims.x * meshDims.y;

                    // Retrieve the start and end indices for this cell.
                    // (Assumes cellStart and cellEnd were pre-initialized to an invalid value like -1 for empty cells)
                    unsigned int startIdx = cellStart[neighborCell];
                    unsigned int endIdx   = cellEnd[neighborCell];

                    // Skip if the cell is empty (if you initialized empty cells to 0xFFFFFFFF or -1)
                    // Here, we assume that if no particle was written, cellStart remains 0 and cellEnd remains 0.
                    if (startIdx == UINT_MAX)
                        continue;

                    // Loop over particles in the neighboring cell
                    for (unsigned int i = startIdx; i < endIdx; i++) {
                        unsigned int neighborParticleIndex = indices[i];

                        // Skip self
                        if (neighborParticleIndex == particleIndex)
                            continue;

                        glm::vec3 neighborPos = positions[neighborParticleIndex];
                        float r = glm::length(pos - neighborPos);

                        if (r <= smoothingRadius && count < maxNeighbors) {
                            neighborList[p * maxNeighbors + count] = static_cast<int>(neighborParticleIndex);
                            count++;
                        }
                    }
                }
            }
        }
        neighborCounts[p] = count;
    }
}


// Poly6 Kernel
__device__ __forceinline__ float W_poly6(float r, float h)
{
    if (r < 0.0f || r > h) {
        return 0.0f;
    }
    // Constant factor: 315 / (64πh^9)
    float alpha = 315.0f / (64.0f * glm::pi<float>() * powf(h, 9));
    float diff  = (h * h - r * r);
    return alpha * diff * diff * diff;
}

// Spiky Gradient
__device__ __forceinline__ glm::vec3 gradW_spiky(const glm::vec3 &rVec, float h)
{
    float r = glm::length(rVec);
    if (r <= 0.0f || r > h) {
        // Either zero-length vector or out of range
        return glm::vec3(0.0f);
    }
    // Constant factor: -45 / (πh^6)
    float alpha = -45.0f / (glm::pi<float>() * powf(h, 6));
    float term  = (h - r) * (h - r);

    // rVec / r is the unit direction from the neighbor to this particle
    return alpha * term * (rVec / r);
}

__device__ __forceinline__ float laplacianW_spiky(float r, float h)
{
    if (r < 0.0f || r > h) {
        return 0.0f;
    }
    // Constant factor: 45 / (πh^6)
    float alpha = 45.0f / (glm::pi<float>() * powf(h, 6));
    return alpha * (h - r);
}

__global__ void computeDensityPressureKernel(
    const glm::vec3* positions,  // Particle positions
    const int* neighborList,     // Flattened neighbor list (neighbor indices per particle)
    const int* neighborCounts,   // Number of neighbors for each particle
    float* densities,            // Output densities per particle
    float* pressures,            // Output pressures per particle
    int numParticles,            // Total number of particles
    float mass,                  // Particle mass
    float smoothingRadius,       // Smoothing radius (h)
    float gasConstant,           // Gas constant (k)
    float restingDensity         // Resting density (ρ₀)
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numParticles; i += stride) {
        float density = 0.0f;
        glm::vec3 pos = positions[i];
        int nCount = neighborCounts[i];

        // Sum contributions from all neighbors
        for (int j = 0; j < nCount; j++) {
            int neighborIdx = neighborList[i * MAX_NEIGHBORS + j];
            glm::vec3 neighborPos = positions[neighborIdx];
            float r = glm::length(pos - neighborPos);
            density += mass * W_poly6(r, smoothingRadius);
        }
        // Optionally add self-contribution:
        density += mass * W_poly6(0.0f, smoothingRadius);

        densities[i] = density;
        pressures[i] = gasConstant * (density - restingDensity);
    }
}


void SPHSimulation::computeDensityAndPressure(int threadsPerBlock, int blocksPerGrid, const int* d_neighborList, const int* d_neighborCounts) const {
    computeDensityPressureKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_positions,
        d_neighborList,
        d_neighborCounts,
        d_densities,
        d_pressures,
        numParticles,
        mass,
        smoothingRadius,
        gasConstant,
        restingDensity
    );
    cudaDeviceSynchronize();
}

__global__ void computeForcesKernel(
    const glm::vec3* positions,    // Particle positions
    const glm::vec3* velocities,   // Particle velocities
    const float* densities,        // Particle densities
    const float* pressures,        // Particle pressures
    const int* neighborList,       // Flattened neighbor list (neighbors per particle)
    const int* neighborCounts,     // Number of neighbors per particle
    glm::vec3* accelerations,      // Output: computed accelerations
    int numParticles,              // Total number of particles
    float mass,                    // Particle mass
    float smoothingRadius,         // Smoothing radius (h)
    float viscosity,               // Viscosity coefficient
    float gravity                  // Gravity (e.g., -9.8f)
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numParticles; i += stride) {
        glm::vec3 pos_i = positions[i];
        glm::vec3 vel_i = velocities[i];
        float rho_i = densities[i];
        float p_i = pressures[i];

        glm::vec3 force_pressure(0.0f);
        glm::vec3 force_viscosity(0.0f);

        int nCount = neighborCounts[i];
        // Loop over neighbors for particle i
        for (int j = 0; j < nCount; j++) {
            int neighborIdx = neighborList[i * MAX_NEIGHBORS + j];
            glm::vec3 pos_j = positions[neighborIdx];
            glm::vec3 vel_j = velocities[neighborIdx];
            float rho_j = densities[neighborIdx];
            float p_j = pressures[neighborIdx];

            glm::vec3 rVec = pos_i - pos_j;
            float r = glm::length(rVec);
            if (r > 0.0f && r <= smoothingRadius) {
                // Compute gradient using the spiky kernel gradient
                glm::vec3 gradW = gradW_spiky(rVec, smoothingRadius);
                // Symmetric pressure force formulation:
                // F_pressure += -m * ( (p_i/(rho_i^2)) + (p_j/(rho_j^2)) ) * gradW
                force_pressure += -mass * ((p_i / (rho_i * rho_i)) + (p_j / (rho_j * rho_j))) * gradW;

                // Viscosity force: using the Laplacian of the spiky kernel
                float lapW = laplacianW_spiky(r, smoothingRadius);
                force_viscosity += viscosity * mass * (vel_j - vel_i) * lapW / rho_j;
            }
        }

        // Gravity force (applied along the Y-axis)
        glm::vec3 force_gravity = mass * glm::vec3(0.0f, gravity, 0.0f);

        // Combine forces and convert to acceleration: a = F_total / m
        glm::vec3 totalForce = force_pressure + force_viscosity + force_gravity;
        accelerations[i] = totalForce / mass;
    }
}


void SPHSimulation::computeForces(int threadsPerBlock, int blocksPerGrid, const int* d_neighborList, const int* d_neighborCounts) const {
    computeForcesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_positions,
        d_velocities,
        d_densities,
        d_pressures,
        d_neighborList,
        d_neighborCounts,
        d_accelerations,
        numParticles,
        mass,
        smoothingRadius,
        viscosity,
        gravity
    );
    cudaDeviceSynchronize();
}

__global__ void moveParticlesVerletKernel(
    glm::vec3* positions,      // Current positions
    glm::vec3* prevPositions,  // Previous positions
    glm::vec3* velocities,     // Particle velocities (to be updated)
    const glm::vec3* accelerations,
    int numParticles,
    float deltaTime
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numParticles; i += stride) {
        glm::vec3 currentPos = positions[i];
        // Verlet integration: x_new = 2*x_current - x_prev + a * dt^2
        glm::vec3 newPos = 2.0f * currentPos - prevPositions[i] + accelerations[i] * deltaTime * deltaTime;
        // Compute velocity: (newPos - x_prev) / (2*dt)
        velocities[i] = (newPos - prevPositions[i]) / (2.0f * deltaTime);
        // Update previous position for the next iteration
        prevPositions[i] = currentPos;
        // Write the new position
        positions[i] = newPos;
    }
}

void SPHSimulation::moveParticles(float deltaTime, int threadsPerBlock, int blocksPerGrid) const {
    moveParticlesVerletKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_positions,
        d_prevPositions,
        d_velocities,
        d_accelerations,
        numParticles,
        deltaTime
    );
    cudaDeviceSynchronize();
}

__global__ void applyBoundaryConditionsKernel(
    glm::vec3* positions,
    glm::vec3* velocities,
    int numParticles,
    glm::vec3 boxSize,
    float boundaryOffset,
    float collisionElasticity)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numParticles; i += stride)
    {
        glm::vec3 pos = positions[i];
        glm::vec3 vel = velocities[i];

        // X-axis boundaries
        if (pos.x < boundaryOffset) {
            pos.x = boundaryOffset + (boundaryOffset - pos.x);
            vel.x = -vel.x * collisionElasticity;
        }
        if (pos.x > boxSize.x - boundaryOffset) {
            pos.x = (boxSize.x - boundaryOffset) - (pos.x - (boxSize.x - boundaryOffset));
            vel.x = -vel.x * collisionElasticity;
        }

        // Y-axis boundaries
        if (pos.y < boundaryOffset) {
            pos.y = boundaryOffset + (boundaryOffset - pos.y);
            vel.y = -vel.y * collisionElasticity;
        }
        if (pos.y > boxSize.y - boundaryOffset) {
            pos.y = (boxSize.y - boundaryOffset) - (pos.y - (boxSize.y - boundaryOffset));
            vel.y = -vel.y * collisionElasticity;
        }

        // Z-axis boundaries
        if (pos.z < boundaryOffset) {
            pos.z = boundaryOffset + (boundaryOffset - pos.z);
            vel.z = -vel.z * collisionElasticity;
        }
        if (pos.z > boxSize.z - boundaryOffset) {
            pos.z = (boxSize.z - boundaryOffset) - (pos.z - (boxSize.z - boundaryOffset));
            vel.z = -vel.z * collisionElasticity;
        }

        positions[i] = pos;
        velocities[i] = vel;
    }
}

void SPHSimulation::applyBoundaryConditions(int threadsPerBlock, int blocksPerGrid) const {
    // Launch the boundary conditions kernel.
    applyBoundaryConditionsKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_positions,
        d_velocities,
        numParticles,
        boxSize,                // Simulation box dimensions
        BOX_COLLISION_OFFSET,   // The offset used to prevent particles from getting too close to the wall
        BOX_COLLISION_ELASTICITY // How “bouncy” the collision is
    );
    cudaDeviceSynchronize();
}



void SPHSimulation::runUpdateKernels(const float deltaTime) {
    // Prepare Kernel Configuration
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int threadsPerBlock = 256;
    int blocksPerGrid = std::min(
        (numParticles + threadsPerBlock - 1) / threadsPerBlock,
        numSMs * 2
    );

    // 1. Compute spatial hashes for particles and build uniform grid structure.
    computeHashes(threadsPerBlock, blocksPerGrid);

    // 2. Allocate neighbor list arrays (flattened list and counts)
    int* d_neighborList = nullptr;
    int* d_neighborCounts = nullptr;
    cudaMalloc(&d_neighborList, numParticles * MAX_NEIGHBORS * sizeof(int));
    cudaMalloc(&d_neighborCounts, numParticles * sizeof(int));

    // 3. Build neighbor list for each particle.
    neighborListKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_positions, d_indices, d_cellStart, d_cellEnd,
        meshDims, cellSize, smoothingRadius,
        d_neighborList, d_neighborCounts, MAX_NEIGHBORS, numParticles
    );
    cudaDeviceSynchronize();

    // 4. Compute density and pressure using the neighbor list.
    computeDensityAndPressure(threadsPerBlock, blocksPerGrid, d_neighborList, d_neighborCounts);

    // 5. Compute forces acting on each particle.
    computeForces(threadsPerBlock, blocksPerGrid, d_neighborList, d_neighborCounts);

    // 6. Move particles using Verlet integration.
    moveParticles(deltaTime, threadsPerBlock, blocksPerGrid);

    // 7. Apply boundary conditions to keep particles within the simulation box.
    applyBoundaryConditions(threadsPerBlock, blocksPerGrid);

    // 8. Free the neighbor list arrays.
    cudaFree(d_neighborList);
    cudaFree(d_neighborCounts);
}
