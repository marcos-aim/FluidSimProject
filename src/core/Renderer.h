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

class Renderer {
public:
    Renderer();
    ~Renderer();

    void createShaderProgram();

    void prepareBoxBuffers(float width, float height, float depth);
    void prepareSphereBuffers(float radius, int slices, int stacks, const std::vector<glm::mat4> &particleTransforms);

    void drawBox(const glm::mat4& view, const glm::mat4& projection);
    void drawSpheres(const glm::mat4& view, const glm::mat4& projection, const glm::vec3& lightDirection);

    GLuint getShaderProgram() {return shaderProgram;}

    void updateInstanceBuffer(const std::vector<glm::vec3>& updatedPositions);
    void updateInstanceBufferWithCuda(float3* d_positions, int numParticles);

    std::vector<glm::mat4> sphereTransforms;

private:
    GLuint shaderProgram;

    GLuint boxVAO, boxVBO, boxEBO; // Box Buffers
    GLuint sphereVAO, sphereVBO, sphereEBO, instanceVBO; // Particle Sphere Buffers and Instancing Buffer

    GLuint modelLoc, viewLoc, projectionLoc, colorLoc; // Uniform Location Ids
    void loadUniformLocations();

    std::vector<float> boxVertices;
    std::vector<unsigned int> boxEdges;
    std::vector<float> sphereVertices;
    std::vector<unsigned int> sphereIndices;

    void generateBoxData(float width, float height, float depth);
    void generateSphereData(float radius, int slices, int stacks);

    cudaGraphicsResource_t cudaInstanceResource = nullptr;
};

#endif // RENDERER_H
