#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>

#include "Renderer.h"
#include "Window.h"
#include "SPH.h"

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

    renderer.prepareBoxBuffers(window.userInput.boxSizeX, window.userInput.boxSizeY, window.userInput.boxSizeZ);

    // Generate low-poly sphere data
    float sphereRadius = 0.05f;
    int sphereSlices = 4, sphereStacks = 4;

    std::vector<glm::mat4> particleTransforms;
    // Stationary particles (example positions relative to the bounding box)
    std::vector<glm::vec3> particlePositions;

    glm::vec4 clearColor = glm::vec4(0.2f, 0.2f, 0.2f, 0.5f);
    window.setupRenderHints(false, true, clearColor); // Dark gray background
    window.initializeImGui();

    SPHSimulation sphSim(window.userInput);
    window.setSimulation(&sphSim);
    sphSim.initParticles(StartingPosition::TOP_CORNER);
    particlePositions = sphSim.h_positions;

    for (const auto& position : particlePositions) {
        glm::mat4 transform = glm::translate(glm::mat4(1.0f), position);
        particleTransforms.push_back(transform);
    }

    renderer.prepareSphereBuffers(sphereRadius, sphereSlices, sphereStacks, particleTransforms);


    float simulationDt = window.userInput.dt; // Fixed time step from the slider
    float accumulator = 0.0f;
    auto currentTime = static_cast<float>(glfwGetTime());
    // Rendering loop
    while (!glfwWindowShouldClose(window.getGLFWWindow())) {
        auto newTime = static_cast<float>(glfwGetTime());
        float frameTime = newTime - currentTime;
        currentTime = newTime;
        accumulator += frameTime;

        // Input handling
        window.processInput();
        // Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        // Use shader program
        glUseProgram(renderer.getShaderProgram());

        simulationDt = window.userInput.dt;
        while (accumulator >= simulationDt) {
            sphSim.isRunning = window.userInput.runSimulation;
            if (sphSim.isRunning) {
                sphSim.update(simulationDt);
            }
            accumulator -= simulationDt;
        }

        if (sphSim.isRunning) {
            renderer.updateInstanceBufferWithCuda(sphSim.getDevicePositions(), sphSim.getNumParticles());
        }

        window.beginFrame();

        renderer.drawBox(window.cameraView, window.cameraProjection);

        glm::vec3 lightDirection = glm::normalize(glm::vec3(-1.0f, -1.0f, 1.0f));
        renderer.drawSpheres(window.cameraView, window.cameraProjection, lightDirection);

        window.renderMenu();

        window.endFrame();
    }

    // Cleanup and exit
    std::cout << "Cleaning up resources..." << std::endl;
    return 0;
}
