#ifndef WINDOW_H
#define WINDOW_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <string>
#include <functional>
#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

#include "Renderer.h"

// Struct to hold user inputs
struct UserInput {
    // Box settings
    float boxSizeX = 5.0f;   // Box width
    float boxSizeY = 2.0f;   // Box height
    float boxSizeZ = 3.0f;   // Box depth

    // Particle Settings
    float particleR = 0.04f;
    int sphereSlices = 6;
    int sphereStacks = 5;

    // SPH settings
    int particleCount = 1000.0f;
    float restingDensity = 1000.0f; // Resting density of the fluid
    float viscosityMultiplier = 1.0f; // Viscosity multiplier
    float mass = 0.2f; // Pressure multiplier
    float gasConstant = 1.0f;
    float h = 0.15f;
    float g = -9.8f;
    float tension = 0.2f;

    // Simulation controls
    bool runSimulation = false;  // Toggle to run/pause the simulation
};

class Window {
public:
    // Constructor
    Window(int width, int height, const std::string& title);

    // Destructor
    ~Window();

    // Initialize GLFW
    bool initializeGLFW();

    // Initialize the window object and make it the current context
    bool createWindow();

    // Set GLFW callbacks
    void setupCallbacks();

    void setupRenderHints(bool vsync, bool antialiasing, glm::vec4& clearColor);

    // Initialize ImGui
    void initializeImGui();

    // Setup ImGui menu tabs
    void setupMenuTabs();

    // Main functions for the main loop
    void beginFrame();  // Starts ImGui frame
    void renderMenu();      // Renders the ImGui menu
    void endFrame();    // Ends ImGui frame

    // Cleanup function
    void cleanup();

    // Getters and setters
    GLFWwindow* getGLFWWindow() const { return window; }
    const UserInput& getUserInput() const { return userInput; }
    void assignCallbackVars();
    void processInput();
    void setUserInput(const UserInput& input) { userInput = input; }

    glm::mat4 cameraView;
    glm::mat4 cameraProjection;

    Renderer* rendererWindow;

private:
    // Private members
    int width, height;
    std::string title;
    GLFWwindow* window;
    UserInput userInput;

    // Callback helpers (as static functions)
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    std::vector<bool> keyStates = std::vector(1024, false);
    glm::vec3 cameraPos = glm::vec3(userInput.boxSizeX/2, userInput.boxSizeY/2, userInput.boxSizeX * 1.5f);
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
};

#endif // WINDOW_H
