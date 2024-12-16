#include "SPH.cuh"

#include <vector>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <iostream>


void SPH::initParticles(int cubeWidth) {
    // Calculate particle spacing
    float particleSpacing = h;

    // Calculate the maximum possible number of particles in the cube
    int maxParticles = cubeWidth * cubeWidth * cubeWidth;

    // Check if the cube fits within the box
    float cubeSize = cubeWidth * particleSpacing;
    if (cubeSize > boxSizeX || cubeSize > boxSizeY || cubeSize > boxSizeZ) {
        std::cerr << "Cube exceeds box dimensions! Adjusting cubeWidth to fit the box." << std::endl;
        cubeWidth = std::min({ static_cast<int>(boxSizeX / particleSpacing),
                               static_cast<int>(boxSizeY / particleSpacing),
                               static_cast<int>(boxSizeZ / particleSpacing) });
    }

    // Adjust the particle count if necessary
    if (particleCount < maxParticles) {
        std::cerr << "Particle count is less than the required number for a full cube. Adjusting particle count to "
                  << maxParticles << "." << std::endl;
        particleCount = maxParticles;
    }

    // Allocate memory for particles
    std::vector<Particle> particles(particleCount);

    // CUDA memory allocation
    Particle* deviceParticles;
    cudaMalloc(&deviceParticles, particleCount * sizeof(Particle));

    // CUDA kernel to initialize particles
    auto kernel = [] __global__(Particle* deviceParticles, int cubeWidth, float particleSpacing) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= cubeWidth * cubeWidth * cubeWidth) return;

        int z = idx / (cubeWidth * cubeWidth);
        int y = (idx / cubeWidth) % cubeWidth;
        int x = idx % cubeWidth;

        deviceParticles[idx].position = glm::vec3(x * particleSpacing, y * particleSpacing, z * particleSpacing);
        deviceParticles[idx].velocity = glm::vec3(0.0f);
        deviceParticles[idx].acceleration = glm::vec3(0.0f);
        deviceParticles[idx].force = glm::vec3(0.0f);
        deviceParticles[idx].density = 0.0f;
        deviceParticles[idx].pressure = 0.0f;
    };

    // Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (particleCount + threadsPerBlock - 1) / threadsPerBlock;
    kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceParticles, cubeWidth, particleSpacing);

    // Copy data back to host
    cudaMemcpy(particles.data(), deviceParticles, particleCount * sizeof(Particle), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(deviceParticles);

    std::cout << "Initialized " << particleCount << " particles in a cube with width " << cubeWidth << "." << std::endl;
}
