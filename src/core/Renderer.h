#ifndef RENDERER_H
#define RENDERER_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <filesystem>

#include "RendererKernels.h"
#include "SPH.h"

class Renderer {
public:
    Renderer();
    ~Renderer();

    void createShaderProgram();

    void prepareBoxBuffers(float width, float height, float depth);
    void prepareSphereBuffers(float radius, int slices, int stacks, const std::vector<glm::mat4> &particleTransforms);

    void drawBox(const glm::mat4& view, const glm::mat4& projection);
    void drawSpheres(const glm::mat4& view, const glm::mat4& projection, const glm::vec3& lightDirection);

    void updateInstanceBuffer(const std::vector<glm::vec3>& updatedPositions);
    void updateInstanceBufferWithCuda(float3* d_positions, int numParticles);

    std::vector<glm::mat4> sphereTransforms;

    // — CUDA↔GL interop for arbitrary framebuffers —
    void initCudaInterop(int width, int height);
    cudaSurfaceObject_t mapCudaSurface();
    void unmapCudaSurface(cudaSurfaceObject_t surf);

    // — full‐screen quad to draw that texture —
    void prepareScreenQuad();
    void drawScreenQuad();

    // DEBUG
    void initVoxelGridRenderer();
    void renderVoxelGrid(const UserInput& ui, SPHSimulation& sim, const glm::mat4& view, const glm::mat4& projection);

private:
    GLuint sceneProgram;
    GLuint textureProgram;

    GLuint boxVAO, boxVBO, boxEBO; // Box Buffers
    GLuint sphereVAO, sphereVBO, sphereEBO, instanceVBO; // Particle Sphere Buffers and Instancing Buffer

    GLuint modelLoc, viewLoc, projectionLoc, colorLoc, lightDirLoc, useInstLoc; // Uniform Location Ids

    void loadSceneUniformLocations();
    void loadTextureUniformLocations();

    std::vector<float> boxVertices;
    std::vector<unsigned int> boxEdges;
    std::vector<float> sphereVertices;
    std::vector<unsigned int> sphereIndices;

    void generateBoxData(float width, float height, float depth);
    void generateSphereData(float radius, int slices, int stacks);

    cudaGraphicsResource_t cudaInstanceResource = nullptr;

    GLuint cudaTexture = 0;
    cudaGraphicsResource_t cudaTextureResource = nullptr;
    int texWidth = 0;
    int texHeight = 0;

    // full-screen quad
    GLuint quadVAO = 0;
    GLuint quadVBO = 0;

    // uniform locations for the single shader
    GLuint cudaTexLoc;

    // DEBUG
    // unit‐cube mesh
    GLuint cubeVAO = 0;
    GLuint cubeVBO = 0;
    GLuint cubeEBO = 0;
    // per‐instance: (pos.x,pos.y,pos.z,scale)
    GLuint instVBO = 0;
    // per‐instance opacity
    GLuint opacVBO = 0;
    GLuint colorVBO = 0;
    GLuint voxelProgram = 0;
    GLint voxelModelLoc = -1;
    GLint voxelViewLoc = -1;
    GLint voxelProjLoc = -1;
    GLint voxelOpacityLoc = -1;
};

#endif // RENDERER_H
