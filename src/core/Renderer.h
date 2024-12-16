#ifndef RENDERER_H
#define RENDERER_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>

class Renderer {
public:
    Renderer();
    ~Renderer();

    void createShaderProgram();

    void prepareBoxBuffers(float width, float height, float depth);
    void prepareSphereBuffers(float radius, int slices, int stacks, const std::vector<glm::mat4> &particleTransforms);

    void drawBox(const glm::mat4& view, const glm::mat4& projection);
    void drawSpheres(const glm::mat4& view, const glm::mat4& projection);

    GLuint getShaderProgram() {return shaderProgram;}

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
};

#endif // RENDERER_H
