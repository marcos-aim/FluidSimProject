// RendererKernels.cu

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <glm/glm.hpp>

// CUDA kernel to update instance transformation matrices.
// For each particle, we write an identity matrix with a translation based on the particle's position.
__global__ void UpdateInstanceTransformsKernel(const float3* positions, float* instanceTransforms, int numParticles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < numParticles; idx += stride) {
        float3 pos = positions[idx];
        int base = idx * 16;
        // First row
        instanceTransforms[base + 0] = 1.0f;
        instanceTransforms[base + 1] = 0.0f;
        instanceTransforms[base + 2] = 0.0f;
        instanceTransforms[base + 3] = 0.0f;
        // Second row
        instanceTransforms[base + 4] = 0.0f;
        instanceTransforms[base + 5] = 1.0f;
        instanceTransforms[base + 6] = 0.0f;
        instanceTransforms[base + 7] = 0.0f;
        // Third row
        instanceTransforms[base + 8] = 0.0f;
        instanceTransforms[base + 9] = 0.0f;
        instanceTransforms[base + 10] = 1.0f;
        instanceTransforms[base + 11] = 0.0f;
        // Fourth row (translation)
        instanceTransforms[base + 12] = pos.x;
        instanceTransforms[base + 13] = pos.y;
        instanceTransforms[base + 14] = pos.z;
        instanceTransforms[base + 15] = 1.0f;
    }
}


// Host wrapper to launch the kernel.
// This function is declared extern "C" so it can be called from C++ code.
extern "C" void launchUpdateInstanceTransformsKernel(const float3* d_positions, float* d_instanceTransforms, int numParticles) {
    int threadsPerBlock = 256;
    int blocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    UpdateInstanceTransformsKernel<<<blocks, threadsPerBlock>>>(d_positions, d_instanceTransforms, numParticles);
    cudaDeviceSynchronize();
}
