//
// Created by maiba on 12/25/2024.
//

#include "SPH.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#include "SPH.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

SPHSimulation::SPHSimulation(const UserInput& input) {
    // Initialize simulation parameters from UserInput:
    numParticles    = input.particleCount;
    smoothingRadius = input.h;
    mass            = input.mass;
    // Use the explicit multipliers provided by the user.
    pMult           = input.pMult;          // New: pressure multiplier
    nearPMult       = input.nearPMult;      // New: near-pressure multiplier
    viscosityMult   = input.viscosityMultiplier;
    surfaceTension  = input.tension;
    gravity         = input.g;
    restingDensity  = input.restingDensity;
    dt              = input.dt;
    collisionDamping = input.collisionDamping; // New: collision damping

    // Convert box size from UserInput to CUDA float3
    boxSize = make_float3(input.boxSizeX, input.boxSizeY, input.boxSizeZ);

    // Allocate host-side particle data (using glm::vec3 for compatibility with rendering)
    h_positions.resize(numParticles);

    // Allocate device memory for particle data using new data types:
    cudaError_t err;
    err = cudaMalloc(&d_positions, numParticles * sizeof(float3));
    if(err != cudaSuccess) {
        std::cerr << "cudaMalloc for d_positions failed: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc(&d_predicted_positions, numParticles * sizeof(float3));
    if(err != cudaSuccess) {
        std::cerr << "cudaMalloc for d_predicted_positions failed: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc(&d_velocities, numParticles * sizeof(float3));
    if(err != cudaSuccess) {
        std::cerr << "cudaMalloc for d_velocities failed: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc(&d_densities, numParticles * sizeof(float2));
    if(err != cudaSuccess) {
        std::cerr << "cudaMalloc for d_densities failed: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMalloc(&d_indices, numParticles * sizeof(uint3));
    if(err != cudaSuccess) {
        std::cerr << "cudaMalloc for d_indices failed: " << cudaGetErrorString(err) << std::endl;
    }
    // d_start_indices may be used for uniform grid or neighbor search; allocate for numParticles here.
    err = cudaMalloc(&d_start_indices, numParticles * sizeof(unsigned int));
    if(err != cudaSuccess) {
        std::cerr << "cudaMalloc for d_start_indices failed: " << cudaGetErrorString(err) << std::endl;
    }

    // Initialize device memory to zero (good practice before simulation starts)
    cudaMemset(d_positions, 0, numParticles * sizeof(float3));
    cudaMemset(d_predicted_positions, 0, numParticles * sizeof(float3));
    cudaMemset(d_velocities, 0, numParticles * sizeof(float3));
    cudaMemset(d_densities, 0, numParticles * sizeof(float2));
    cudaMemset(d_indices, 0, numParticles * sizeof(uint3));
    cudaMemset(d_start_indices, 0, numParticles * sizeof(unsigned int));

    std::cout << "SPHSimulation initialized with " << numParticles << " particles." << std::endl;
}

SPHSimulation::~SPHSimulation() {
    // Free device memory allocated for particle data
    if (d_positions) {
        cudaFree(d_positions);
        d_positions = nullptr;
    }
    if (d_predicted_positions) {
        cudaFree(d_predicted_positions);
        d_predicted_positions = nullptr;
    }
    if (d_velocities) {
        cudaFree(d_velocities);
        d_velocities = nullptr;
    }
    if (d_densities) {
        cudaFree(d_densities);
        d_densities = nullptr;
    }
    if (d_indices) {
        cudaFree(d_indices);
        d_indices = nullptr;
    }
    if (d_start_indices) {
        cudaFree(d_start_indices);
        d_start_indices = nullptr;
    }

    std::cout << "SPHSimulation resources freed. All device memory released." << std::endl;
}

void SPHSimulation::updateParameters(const UserInput& input) {
    // Update internal simulation parameters from the new input.
    smoothingRadius = input.h;
    mass            = input.mass;
    pMult           = input.pMult;            // Use new explicit parameter.
    nearPMult       = input.nearPMult;        // Use new explicit parameter.
    viscosityMult   = input.viscosityMultiplier;
    surfaceTension  = input.tension;
    gravity         = input.g;
    restingDensity  = input.restingDensity;
    dt              = input.dt;
    collisionDamping = input.collisionDamping; // Update collision damping from input.
    boxSize         = make_float3(input.boxSizeX, input.boxSizeY, input.boxSizeZ);

    std::cout << "SPHSimulation parameters updated." << std::endl;
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

    // Convert host positions (glm::vec3) to CUDA float3 array.
    std::vector<float3> devicePositions(numParticles);
    for (int i = 0; i < numParticles; i++) {
        devicePositions[i] = make_float3(h_positions[i].x, h_positions[i].y, h_positions[i].z);
    }
    cudaMemcpy(d_positions, devicePositions.data(), numParticles * sizeof(float3), cudaMemcpyHostToDevice);

    // Optionally, you can initialize d_predicted_positions similarly.
    cudaMemcpy(d_predicted_positions, devicePositions.data(), numParticles * sizeof(float3), cudaMemcpyHostToDevice);

}


void SPHSimulation::update(float deltaTime) {
    if (isRunning) {
        runUpdateKernels(deltaTime);
    }
}

std::vector<glm::vec3>& SPHSimulation::getParticlePositions() {
    // Copy device positions (float3) back to the host vector (glm::vec3)
    std::vector<float3> devicePositions(numParticles);
    cudaMemcpy(devicePositions.data(), d_positions, numParticles * sizeof(float3), cudaMemcpyDeviceToHost);

    // Convert each float3 to glm::vec3
    for (int i = 0; i < numParticles; i++) {
        h_positions[i] = glm::vec3(devicePositions[i].x, devicePositions[i].y, devicePositions[i].z);
    }
    return h_positions;
}
