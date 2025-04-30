// RendererKernels.h
#ifndef RENDERER_KERNELS_H
#define RENDERER_KERNELS_H

#include <cuda_runtime.h>
#include "CUDAParams.h"
#include "UserInput.h"

#ifdef __cplusplus
extern "C" {
#endif

// Host function prototype for launching the kernel.
void launchUpdateInstanceTransformsKernel(const float3 *d_positions, float *d_instanceTransforms, int numParticles);

void launchGenerateChecker(cudaSurfaceObject_t surface, int width, int height, int checkerSize);

void launchAABBTestKernel(cudaSurfaceObject_t surface, const CameraCUDAParams &cam, const UserInput &ui);


#ifdef __cplusplus
}
#endif

#endif // RENDERER_KERNELS_H
