#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>

#include "Renderer.h"
#include "Window.h"

int main() {
    Window window(1280, 720, "SPH Simulation Test");

    // Initialize GLFW
    if (!window.initializeGLFW()) {
        return -1;
    }

    // Create the window
    if (!window.createWindow()) {
        return -1;
    }

    // Load OpenGL functions with GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    window.setupCallbacks();

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    Renderer renderer;
    window.rendererWindow = &renderer;
    renderer.createShaderProgram();

    // Generate bounding box data
    float boxWidth = 5.0f, boxHeight = 2.0f, boxDepth = 3.0f;
    renderer.prepareBoxBuffers(boxWidth, boxHeight, boxDepth);

    // Generate low-poly sphere data
    float sphereRadius = 0.05f;
    int sphereSlices = 6, sphereStacks = 5;

    std::vector<glm::mat4> particleTransforms;
    // Stationary particles (example positions relative to the bounding box)
    std::vector<glm::vec3> particlePositions = {
        {0.5f, 0.5f, 0.5f},
        {1.5f, 0.5f, 0.5f},
        {2.5f, 1.5f, 1.0f},
        {1.0f, 1.0f, 1.5f},
        {0.5f, 1.5f, 2.5f},
        {4, 1, 2.7},
        {3.6, 1.4, 2.4},
        {4.5, 0.7, 0.3}
    };

    for (const auto& position : particlePositions) {
        glm::mat4 transform = glm::translate(glm::mat4(1.0f), position);
        particleTransforms.push_back(transform);
    }

    renderer.prepareSphereBuffers(sphereRadius, sphereSlices, sphereStacks, particleTransforms);

    glm::vec4 clearColor = glm::vec4(0.2f, 0.2f, 0.2f, 0.5f);
    window.setupRenderHints(false, true, clearColor); // Dark gray background
    window.initializeImGui();

    // Rendering loop
    while (!glfwWindowShouldClose(window.getGLFWWindow())) {

        // Input handling
        window.assignCallbackVars();
        window.processInput();

        // Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        // Use shader program
        glUseProgram(renderer.getShaderProgram());

        window.beginFrame();

        renderer.drawBox(window.cameraView, window.cameraProjection);
        renderer.drawSpheres(window.cameraView, window.cameraProjection);

        window.renderMenu();

        window.endFrame();
    }

    // Cleanup and exit
    std::cout << "Cleaning up resources..." << std::endl;
    return 0;

    return 0;
}
