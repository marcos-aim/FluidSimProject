// low_poly_sphere_with_box.cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>

// Vertex and fragment shaders
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
uniform vec4 color;
out vec4 FragColor;
void main() {
    FragColor = color;
}
)";

// Function to generate a bounding box
std::vector<float> generateBoundingBoxVertices(float width, float height, float depth) {
    return {
        0.0f, 0.0f, 0.0f,
        width, 0.0f, 0.0f,
        width, height, 0.0f,
        0.0f, height, 0.0f,
        0.0f, 0.0f, depth,
        width, 0.0f, depth,
        width, height, depth,
        0.0f, height, depth
    };
}

std::vector<unsigned int> generateBoundingBoxEdges() {
    return {
        0, 1, 1, 2, 2, 3, 3, 0, // Bottom face edges
        4, 5, 5, 6, 6, 7, 7, 4, // Top face edges
        0, 4, 1, 5, 2, 6, 3, 7  // Vertical edges
    };
}

// Function to generate low-poly sphere vertices
std::vector<float> generateSphereVertices(float radius, int slices, int stacks) {
    std::vector<float> vertices;
    for (int i = 0; i <= stacks; ++i) {
        float theta = i * glm::pi<float>() / stacks; // Latitude angle
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        for (int j = 0; j <= slices; ++j) {
            float phi = j * 2.0f * glm::pi<float>() / slices; // Longitude angle
            float x = radius * sinTheta * cos(phi);
            float y = radius * cosTheta;
            float z = radius * sinTheta * sin(phi);
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }
    return vertices;
}

// Generate sphere indices
std::vector<unsigned int> generateSphereIndices(int slices, int stacks) {
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
    return indices;
}

// Camera settings
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f;

// Mouse input
bool firstMouse = true;
float yaw = -90.0f; // Initialize to face -Z
float pitch = 0.0f;
float lastX = 400.0f, lastY = 300.0f; // Center of screen
float fov = 45.0f;

void processInput(GLFWwindow* window) {
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    float cameraSpeed = 2.5f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // Reversed since y-coordinates go from bottom to top
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    fov -= (float)yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 45.0f)
        fov = 45.0f;
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a GLFW window
    GLFWwindow* window = glfwCreateWindow(800, 600, "Low-Poly Sphere with Bounding Box", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Load OpenGL functions with GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Configure callbacks
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    // Compile shaders
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Generate bounding box data
    float boxWidth = 3.0f, boxHeight = 2.0f, boxDepth = 4.0f;
    std::vector<float> boxVertices = generateBoundingBoxVertices(boxWidth, boxHeight, boxDepth);
    std::vector<unsigned int> boxEdges = generateBoundingBoxEdges();

    GLuint boxVAO, boxVBO, boxEBO;
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

    // Generate low-poly sphere data
    float sphereRadius = 0.1f;
    int sphereSlices = 8, sphereStacks = 6;
    std::vector<float> sphereVertices = generateSphereVertices(sphereRadius, sphereSlices, sphereStacks);
    std::vector<unsigned int> sphereIndices = generateSphereIndices(sphereSlices, sphereStacks);

    GLuint sphereVAO, sphereVBO, sphereEBO;
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

    // Stationary particles (example positions relative to the bounding box)
    std::vector<glm::vec3> particles = {
        {0.5f, 0.5f, 0.5f},
        {1.5f, 0.5f, 0.5f},
        {2.5f, 1.5f, 1.0f},
        {1.0f, 1.0f, 3.0f},
        {0.5f, 1.5f, 2.5f}
    };

    // Rendering loop
    while (!glfwWindowShouldClose(window)) {
        // Input handling
        processInput(window);

        // Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use shader program
        glUseProgram(shaderProgram);

        // Update view and projection matrices
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(fov), 800.0f / 600.0f, 0.1f, 100.0f);

        GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
        GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLuint projectionLoc = glGetUniformLocation(shaderProgram, "projection");
        GLuint colorLoc = glGetUniformLocation(shaderProgram, "color");

        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

        // Render bounding box
        glm::mat4 boxModel = glm::mat4(1.0f);
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(boxModel));
        glUniform4f(colorLoc, 1.0f, 1.0f, 1.0f, 1.0f); // White for bounding box
        glBindVertexArray(boxVAO);
        glDrawElements(GL_LINES, boxEdges.size(), GL_UNSIGNED_INT, 0);

        // Render particles as spheres with edges
        for (const auto& particle : particles) {
            glm::mat4 particleModel = glm::translate(glm::mat4(1.0f), particle);
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(particleModel));

            // Draw sphere faces (blue)
            glUniform4f(colorLoc, 0.0f, 0.0f, 1.0f, 1.0f); // Blue for particles
            glBindVertexArray(sphereVAO);
            glDrawElements(GL_TRIANGLES, sphereIndices.size(), GL_UNSIGNED_INT, 0);

            // Draw sphere edges (white)
            glUniform4f(colorLoc, 1.0f, 1.0f, 1.0f, 1.0f); // White for edges
            glDrawElements(GL_LINES, sphereIndices.size(), GL_UNSIGNED_INT, 0);
        }

        glBindVertexArray(0);

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &boxVAO);
    glDeleteBuffers(1, &boxVBO);
    glDeleteBuffers(1, &boxEBO);
    glDeleteVertexArrays(1, &sphereVAO);
    glDeleteBuffers(1, &sphereVBO);
    glDeleteBuffers(1, &sphereEBO);
    glDeleteProgram(shaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
