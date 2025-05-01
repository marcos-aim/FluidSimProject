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

    // Initialize GLFW + create window + load GLAD…
    if (!window.initializeGLFW() || !window.createWindow() || !gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD or start Window" << std::endl;
        return -1;
    }

    window.setupCallbacks();

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    Renderer renderer;
    window.rendererWindow = &renderer;
    renderer.createShaderProgram();
    renderer.initCudaInterop(window.width, window.height);
    renderer.prepareScreenQuad();

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

    for (const auto &position: particlePositions) {
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

        simulationDt = window.userInput.dt;
        while (accumulator >= simulationDt) {
            sphSim.isRunning = window.userInput.runSimulation;
            if (sphSim.isRunning) {
                sphSim.update(simulationDt);
            }
            accumulator -= simulationDt;
        }

        window.beginFrame();

        if (window.userInput.rayMarchRender) {
            // --- RAY-MARCH MODE ---
            // 1) map → launch the checker into the GL texture → unmap
            cudaSurfaceObject_t surf = renderer.mapCudaSurface();

            //launchGenerateChecker(surf, window.width, window.height, 32);
            auto cam = window.getCameraCUDAParams();
            auto ui = window.userInput;

            // 3) launch
            if (ui.debugSurface) {
                launchSurfaceDebug(surf, cam,
                                   sphSim.densityTex,
                                   make_float3(ui.boxSizeX, ui.boxSizeY, ui.boxSizeZ),
                                   ui.accumulationStepSize,
                                   ui.surfaceMinDensity);
            } else {
                launchAABBTestKernel(surf, cam, ui); // ← your old white-mask pass
            }

            // 4) unmap + draw
            renderer.unmapCudaSurface(surf);
            renderer.drawScreenQuad();
        } else if (window.userInput.renderVoxelGrid) {
            // --- VOXEL-GRID DEBUG MODE ---
            // Pull down the CUDA density grid and draw instanced cubes + grid lines
            renderer.renderVoxelGrid(
                window.userInput,
                sphSim,
                window.cameraView,
                window.cameraProjection
            );
        } else {
            // --- SPH BOX+SPHERE MODE ---
            if (sphSim.isRunning) {
                renderer.updateInstanceBufferWithCuda(sphSim.getDevicePositions(), sphSim.getNumParticles());
            }
            renderer.drawBox(window.cameraView, window.cameraProjection);

            glm::vec3 lightDirection = glm::normalize(glm::vec3(-1.0f, -1.0f, 1.0f));
            renderer.drawSpheres(window.cameraView, window.cameraProjection, lightDirection);
        }

        window.renderMenu();

        window.endFrame();
    }

    // Cleanup and exit
    std::cout << "Cleaning up resources..." << std::endl;
    return 0;
}
