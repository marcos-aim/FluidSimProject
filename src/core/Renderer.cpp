#include "Renderer.h"

Renderer::Renderer() : boxVAO(0), boxVBO(0), boxEBO(0), sphereVAO(0), sphereVBO(0), sphereEBO(0), instanceVBO(0) {}

Renderer::~Renderer() {
    glDeleteVertexArrays(1, &boxVAO);
    glDeleteBuffers(1, &boxVBO);
    glDeleteBuffers(1, &boxEBO);
    glDeleteVertexArrays(1, &sphereVAO);
    glDeleteBuffers(1, &sphereVBO);
    glDeleteBuffers(1, &sphereEBO);
    glDeleteBuffers(1, &instanceVBO);
}



void Renderer::createShaderProgram() {
    // Vertex and fragment shaders
    const char* vertexShaderSource = R"(
    #version 450 core

    layout(location = 0) in vec3 aPos;
    layout(location = 1) in mat4 instanceModel; // Per-instance model matrix

    uniform mat4 model;        // For objects like the box
    uniform mat4 view;
    uniform mat4 projection;

    uniform bool useInstance;  // Flag to differentiate between instanced and single-object rendering

    void main() {
        mat4 effectiveModel = useInstance ? instanceModel : model;
        gl_Position = projection * view * effectiveModel * vec4(aPos, 1.0);
    }

    )";

    const char* fragmentShaderSource = R"(
    #version 450 core
    uniform vec4 color;
    out vec4 FragColor;
    void main() {
        FragColor = color;
    }
    )";

    // Compile shaders
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    loadUniformLocations();

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void Renderer::loadUniformLocations() {
    modelLoc = glGetUniformLocation(shaderProgram, "model");
    viewLoc = glGetUniformLocation(shaderProgram, "view");
    projectionLoc = glGetUniformLocation(shaderProgram, "projection");
    colorLoc = glGetUniformLocation(shaderProgram, "color");
}

void Renderer::generateBoxData(float width, float height, float depth) {
    boxVertices = {
        0.0f, 0.0f, 0.0f,
        width, 0.0f, 0.0f,
        width, height, 0.0f,
        0.0f, height, 0.0f,
        0.0f, 0.0f, depth,
        width, 0.0f, depth,
        width, height, depth,
        0.0f, height, depth
    };
    boxEdges = {
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7
    };
}

void Renderer::prepareBoxBuffers(float width, float height, float depth) {
    generateBoxData(width, height, depth);

    glGenVertexArrays(1, &boxVAO);
    glGenBuffers(1, &boxVBO);
    glGenBuffers(1, &boxEBO);

    glBindVertexArray(boxVAO);

    glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glBufferData(GL_ARRAY_BUFFER, boxVertices.size() * sizeof(float), boxVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boxEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, boxEdges.size() * sizeof(unsigned int), boxEdges.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Renderer::drawBox(const glm::mat4& view, const glm::mat4& projection) {
    glm::mat4 boxModel = glm::mat4(1.0f);
    glUniform1i(glGetUniformLocation(shaderProgram, "useInstance"), false);
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(boxModel));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniform4f(colorLoc, 1.0f, 1.0f, 1.0f, 1.0f); // White for bounding box

    glBindVertexArray(boxVAO);
    glDrawElements(GL_LINES, boxEdges.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Renderer::generateSphereData(float radius, int slices, int stacks) {
    std::vector<float> vertices;
    for (int i = 0; i <= stacks; ++i) {
        float theta = i * glm::pi<float>() / stacks; // Latitude angle
        float sinTheta = glm::sin(theta);
        float cosTheta = glm::cos(theta);

        for (int j = 0; j <= slices; ++j) {
            float phi = j * 2.0f * glm::pi<float>() / slices; // Longitude angle
            float x = radius * sinTheta * glm::cos(phi);
            float y = radius * cosTheta;
            float z = radius * sinTheta * glm::sin(phi);
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }
    sphereVertices = vertices;

    std::vector<unsigned int> indices;
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int first = (i * (slices + 1)) + j;
            int second = first + slices + 1;

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }
    sphereIndices = indices;
}

void Renderer::prepareSphereBuffers(float radius, int slices, int stacks, const std::vector<glm::mat4> &particleTransforms) {
    generateSphereData(radius, slices, stacks);
    sphereTransforms = particleTransforms;

    glGenVertexArrays(1, &sphereVAO);
    glGenBuffers(1, &sphereVBO);
    glGenBuffers(1, &sphereEBO);

    glBindVertexArray(sphereVAO);

    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, sphereVertices.size() * sizeof(float), sphereVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndices.size() * sizeof(unsigned int), sphereIndices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);


    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, particleTransforms.size() * sizeof(glm::mat4), particleTransforms.data(), GL_STATIC_DRAW);

    glBindVertexArray(sphereVAO);
    for (int i = 0; i < 4; i++) {
        glEnableVertexAttribArray(1 + i);
        glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(i * sizeof(glm::vec4)));
        glVertexAttribDivisor(1 + i, 1);
    }
    glBindVertexArray(0);
}

void Renderer::drawSpheres(const glm::mat4& view, const glm::mat4& projection) {
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniform1i(glGetUniformLocation(shaderProgram, "useInstance"), true);
    glUniform4f(colorLoc, 0.0f, 0.0f, 1.0f, 1.0f);
    glBindVertexArray(sphereVAO);
    glDrawElementsInstanced(GL_TRIANGLES, sphereIndices.size(), GL_UNSIGNED_INT, nullptr, sphereTransforms.size());
    glBindVertexArray(0);
}