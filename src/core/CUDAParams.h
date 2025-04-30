//
// Created by maiba on 4/30/2025.
//

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#ifndef CUDAPARAMS_H
#define CUDAPARAMS_H

struct CameraCUDAParams {
    float3 pos;
    float3 forward;
    float3 right;
    float3 up;
    float  tanHalfFOV;
    float  aspect;
    int    width;
    int    height;
};

#endif //CUDAPARAMS_H
