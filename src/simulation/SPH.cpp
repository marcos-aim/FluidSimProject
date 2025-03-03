//
// Created by maiba on 12/25/2024.
//

#include "SPH.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

SPHSimulation::SPHSimulation(const UserInput& input) {
    // Initialize simulation parameters from UserInput
    numParticles = input.particleCount;
    smoothingRadius = input.h;
    mass = input.mass;
    gasConstant = input.gasConstant;
    viscosity = input.viscosityMultiplier;
    surfaceTension = input.tension;
    gravity = input.g;
    restingDensity = input.restingDensity;
    boxSize = glm::vec3(input.boxSizeX, input.boxSizeY, input.boxSizeZ);

    // Initialize timing record
    timingRecord.resize(maxTimings, 0.0f);

    // Allocate host-side particle data
    h_positions.resize(numParticles);

    // Allocate device memory for particle data
    cudaMalloc(&d_positions, numParticles * sizeof(glm::vec3));
    cudaMalloc(&d_velocities, numParticles * sizeof(glm::vec3));
    cudaMalloc(&d_accelerations, numParticles * sizeof(glm::vec3));
    cudaMalloc(&d_densities, numParticles * sizeof(float));
    cudaMalloc(&d_pressures, numParticles * sizeof(float));
    cudaMalloc(&d_prevPositions, numParticles * sizeof(glm::vec3));


    // Allocate device memory for uniform grid data
    meshDims = glm::ivec3(
    static_cast<int>(boxSize.x / smoothingRadius),
    static_cast<int>(boxSize.y / smoothingRadius),
    static_cast<int>(boxSize.z / smoothingRadius)
    );
    int numCells = meshDims.x * meshDims.y * meshDims.z;
    cudaMalloc(&d_hashes, numParticles * sizeof(int));
    cudaMalloc(&d_indices, numParticles * sizeof(int));
    cudaMalloc(&d_cellStart, numCells * sizeof(unsigned int));
    cudaMalloc(&d_cellEnd, numCells * sizeof(unsigned int));

    // Initialize device memory
    cudaMemset(d_positions, 0, numParticles * sizeof(glm::vec3));
    cudaMemset(d_velocities, 0, numParticles * sizeof(glm::vec3));
    cudaMemset(d_accelerations, 0, numParticles * sizeof(glm::vec3));
    cudaMemset(d_densities, 0, numParticles * sizeof(float));
    cudaMemset(d_pressures, 0, numParticles * sizeof(float));
    cudaMemset(d_hashes, 0, numParticles * sizeof(int));
    cudaMemset(d_indices, 0, numParticles * sizeof(int));
    cudaMemset(d_cellStart, 0xFF, numCells * sizeof(unsigned int));
    cudaMemset(d_cellEnd, 0, numCells * sizeof(unsigned int));

    std::cout << "SPHSimulation initialized with " << numParticles << " particles.\n";
}


SPHSimulation::~SPHSimulation() {
    // Free device memory for particle data
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaFree(d_accelerations);
    cudaFree(d_densities);
    cudaFree(d_pressures);

    // Free device memory for uniform grid data
    cudaFree(d_hashes);
    cudaFree(d_indices);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);

    // Print confirmation for debugging
    std::cout << "SPHSimulation resources freed. All device memory released.\n";
}


void SPHSimulation::initParticles(StartingPosition startType) {
    // Clear any existing particle data
    h_positions.clear();

    glm::vec3 spawnOrigin;                // Origin for spawning particles

    // Determine particle cube dimensions (ceiling to overestimate)
    int particlesPerSide = static_cast<int>(std::ceil(std::cbrt(numParticles)));
    float cubeSize = (particlesPerSide - 1) * SPAWN_SEPARATION + BOUNDARY_BUFFER;

    // Check if the cube fits in the box
    if (cubeSize > boxSize.x) {
        throw std::runtime_error("Top corner cube does not fit in the box. Enlarge the X dimension.");
    }
    if (cubeSize > boxSize.z) {
        throw std::runtime_error("Top corner cube does not fit in the box. Enlarge the Z dimension.");
    }
    if (cubeSize > boxSize.y) {
        throw std::runtime_error("Top corner cube does not fit in the box. Enlarge the Y dimension.");
    }

    // Initialize the particle counter
    int particleCount = 0;

    // Switch on StartingPosition
    switch (startType) {
        case StartingPosition::TOP_CORNER:
            // Spawn particles at the top corner
            spawnOrigin = glm::vec3(BOUNDARY_BUFFER, boxSize.y - BOUNDARY_BUFFER, BOUNDARY_BUFFER);
            for (int x = 0; x < particlesPerSide; ++x) {
                for (int y = 0; y < particlesPerSide; ++y) {
                    for (int z = 0; z < particlesPerSide; ++z) {
                        if (particleCount >= numParticles) break; // Stop when we reach nParticles

                        glm::vec3 pos = spawnOrigin +
                                        glm::vec3(x * SPAWN_SEPARATION, -y * SPAWN_SEPARATION, z * SPAWN_SEPARATION);
                        h_positions.push_back(pos);
                        ++particleCount;
                    }
                    if (particleCount >= numParticles) break;
                }
                if (particleCount >= numParticles) break;
            }
            break;

        default:
            throw std::invalid_argument("Starting position not implemented.");
    }

    // Copy to device memory
    cudaMemcpy(d_positions, h_positions.data(), h_positions.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prevPositions, h_positions.data(), numParticles * sizeof(glm::vec3), cudaMemcpyHostToDevice);

}


void SPHSimulation::update(float deltaTime) {
    if (isRunning) {
        runUpdateKernels(deltaTime);
    }
}

const std::vector<glm::vec3>& SPHSimulation::getParticlePositions() const {
    // Ensure the host vector has the correct size
    const_cast<std::vector<glm::vec3>&>(h_positions).resize(numParticles);

    // Copy data from device (d_positions) to host (particlePositions)
    cudaMemcpy(
            const_cast<std::vector<glm::vec3>&>(h_positions).data(), // Destination (host)
            d_positions,                                                   // Source (device)
            numParticles * sizeof(glm::vec3),                              // Size in bytes
            cudaMemcpyDeviceToHost                                         // Direction of transfer
    );

    // Return the host-side particle positions
    return h_positions;
}


// DEBUG:

void printNeighborList(const int* d_neighborList, const int* d_neighborCounts, const int numParticles, const int maxNeighbors) {
    // Allocate host memory
    std::vector<int> h_neighborList(numParticles * maxNeighbors);
    std::vector<int> h_neighborCounts(numParticles);

    // Copy data from device to host
    cudaMemcpy(h_neighborList.data(), d_neighborList, numParticles * maxNeighbors * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_neighborCounts.data(), d_neighborCounts, numParticles * sizeof(int), cudaMemcpyDeviceToHost);

    // Print neighbor list
    std::cout << "Neighbor List:\n";
    for (int i = 0; i < numParticles; ++i) {
        std::cout << "Particle " << i << " (Count: " << h_neighborCounts[i] << "): ";
        for (int j = 0; j < h_neighborCounts[i]; ++j) {
            std::cout << h_neighborList[i * maxNeighbors + j] << " ";
        }
        std::cout << "\n";
    }
}

void printHashTable(int* d_hashes, int* d_indices, glm::vec3* d_positions, int numParticles) {
    // Allocate host memory
    std::vector<int> h_hashes(numParticles);
    std::vector<int> h_indices(numParticles);
    std::vector<glm::vec3> h_positions(numParticles);

    // Copy data from device to host
    cudaMemcpy(h_hashes.data(), d_hashes, numParticles * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices.data(), d_indices, numParticles * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_positions.data(), d_positions, numParticles * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    // Print hash table
    std::cout << "Hash Table:\n";
    int currentHash = -1;
    for (int i = 0; i < numParticles; ++i) {
        if (h_hashes[i] != currentHash) {
            if (currentHash != -1) std::cout << "\n"; // Close previous group
            currentHash = h_hashes[i];
            std::cout << "Hash " << currentHash << ":";
        }
        std::cout << "\n  Particle " << h_indices[i] << " -> Position ("
                  << h_positions[h_indices[i]].x << ", "
                  << h_positions[h_indices[i]].y << ", "
                  << h_positions[h_indices[i]].z << ")";
    }
    std::cout << "\n";
}
