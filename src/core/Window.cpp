#include "Window.h"
#include <iostream>
#include <utility>

// Constructor
Window::Window(int width, int height, std::string  title)
    : width(width), height(height), title(std::move(title)), window(nullptr) {}

// Destructor
Window::~Window() {
    cleanup();
}

// Initialize GLFW
bool Window::initializeGLFW() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW." << std::endl;
        return false;
    }
    // Set GLFW window hints for OpenGL version and profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    return true;
}

// Create the GLFW window and make it the current context
bool Window::createWindow() {
    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    // In createWindow() after window creation, add:
    glfwSetWindowUserPointer(window, this);

    return true;
}

// Set GLFW callbacks
void Window::setupCallbacks() {
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
}

void Window::setupRenderHints(bool vsync, bool antialiasing, glm::vec4& clearColor) {
    if (antialiasing) {
        glfwWindowHint(GLFW_SAMPLES, 4); // Request 4x MSAA during context creation
        glEnable(GL_MULTISAMPLE);       // Enable MSAA in OpenGL
        glEnable(GL_LINE_SMOOTH);         // Enable line smoothing
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST); // Request the best quality
    }
    if (!vsync) {
        glfwSwapInterval(0);
    }

    glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
}

// Initialize ImGui
void Window::initializeImGui() const {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");
}

// Setup ImGui menu tabs
void Window::setupMenuTabs() {
    ImGui::Begin("Simulation Menu");

    // Box Settings Dropdown
    if (ImGui::CollapsingHeader("Box Settings")) {
        bool sizeXChanged = ImGui::SliderFloat("Box Size X", &userInput.boxSizeX, 0.1f, 10.0f);
        bool sizeYChanged = ImGui::SliderFloat("Box Size Y", &userInput.boxSizeY, 0.1f, 10.0f);
        bool sizeZChanged = ImGui::SliderFloat("Box Size Z", &userInput.boxSizeZ, 0.1f, 10.0f);

        // Check and print messages when a value changes
        if (sizeXChanged || sizeYChanged || sizeZChanged) {
            rendererWindow->prepareBoxBuffers(userInput.boxSizeX, userInput.boxSizeY, userInput.boxSizeZ);
        }
    }

    if (ImGui::CollapsingHeader("Particle Settings")) {
        bool radiusChanged = ImGui::SliderFloat("Particle Radius", &userInput.particleR, 0.01f, 0.3f);
        bool slicesChanged = ImGui::SliderInt("Sphere Slices", &userInput.sphereSlices, 4, 20);
        bool stacksChanged = ImGui::SliderInt("Sphere Stacks", &userInput.sphereStacks, 4, 20);

        // Check and update sphere buffers if any sphere-related parameter changes
        if (radiusChanged || slicesChanged || stacksChanged) {
            rendererWindow->prepareSphereBuffers(userInput.particleR, userInput.sphereSlices, userInput.sphereStacks,
                rendererWindow->sphereTransforms);
        }
    }

    // SPH Settings Dropdown
    if (ImGui::CollapsingHeader("SPH Settings")) {
        ImGui::SliderInt("Particle Count", &userInput.particleCount, 0, 50000, "%.1f");
        ImGui::SliderFloat("Resting Density", &userInput.restingDensity, 500.0f, 2000.0f, "%.1f");
        ImGui::SliderFloat("Viscosity Multiplier", &userInput.viscosityMultiplier, 0.1f, 10.0f, "%.2f");
        ImGui::SliderFloat("Mass", &userInput.mass, 0.1f, 5.0f, "%.2f");
        ImGui::SliderFloat("Gas Constant", &userInput.gasConstant, 0.1f, 10.0f, "%.2f");
        ImGui::SliderFloat("Smoothing Radius (h)", &userInput.h, 0.05f, 1.0f, "%.3f");
        ImGui::SliderFloat("Gravity (g)", &userInput.g, -20.0f, 0.0f, "%.1f");
        ImGui::SliderFloat("Surface Tension", &userInput.tension, 0.0f, 1.0f, "%.2f");

        if (ImGui::Button("Reset to Defaults")) {
            userInput.restingDensity = 1000.0f;
            userInput.viscosityMultiplier = 1.0f;
            userInput.mass = 0.2f;
            userInput.gasConstant = 1.0f;
            userInput.h = 0.15f;
            userInput.g = -9.8f;
            userInput.tension = 0.2f;
        }
    }


    // Simulation Controls Dropdown
    if (ImGui::CollapsingHeader("Simulation Controls")) {
        if (ImGui::Checkbox("Run Simulation", &userInput.runSimulation)) {
            if (userInput.runSimulation) {
                std::cout << "Simulation started." << std::endl;
            } else {
                std::cout << "Simulation paused." << std::endl;
            }
        }
    }
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();
}


// Start ImGui frame
void Window::beginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

// Render ImGui menu
void Window::renderMenu() {
    setupMenuTabs(); // Setup menu tabs in the render loop
}

// End ImGui frame
void Window::endFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
    glfwPollEvents();
}

// Cleanup function
void Window::cleanup() {
    if (window) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

// Framebuffer size callback (static)
void Window::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void Window::keyCallback(GLFWwindow* glfwWindow, int key, int scancode, int action, int mods) {
    // Retrieve the Window instance:
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(glfwWindow));
    if (key >= 0 && key < win->keyStates.size()) {
        if (action == GLFW_PRESS) {
            win->keyStates[key] = true;
            if (win->keyStates[GLFW_KEY_ESCAPE])
                glfwSetWindowShouldClose(glfwWindow, true);
        } else if (action == GLFW_RELEASE) {
            win->keyStates[key] = false;
        }
    }
}


float localLastXpos;
float localLastYpos;
void Window::processInput() {
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    float cameraSpeed = 2.5f * deltaTime; // Frame-independent movement speed

    if (keyStates[GLFW_KEY_W]) {
        cameraPos += cameraSpeed * cameraFront;
    }
    if (keyStates[GLFW_KEY_S]) {
        cameraPos -= cameraSpeed * cameraFront;
    }
    if (keyStates[GLFW_KEY_A]) {
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    }
    if (keyStates[GLFW_KEY_D]) {
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    }

    // Up (Space) and Down (Shift)
    if (keyStates[GLFW_KEY_SPACE]) {
        cameraPos += cameraSpeed * cameraUp; // Move up
    }
    if (keyStates[GLFW_KEY_LEFT_SHIFT] || keyStates[GLFW_KEY_RIGHT_SHIFT]) {
        cameraPos -= cameraSpeed * cameraUp; // Move down
    }

    if (keyStates[GLFW_KEY_M]) { // menu mouse free
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        cameraMode = false;
    }
    if (keyStates[GLFW_KEY_C]) { // camera locked
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetCursorPos(window, localLastXpos, localLastYpos);
        cameraMode = true;
    }

    cameraView = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    cameraProjection = glm::perspective(glm::radians(fov), static_cast<float>(width) / static_cast<float>(height), 0.1f, 100.0f);
}

void Window::mouseCallback(GLFWwindow* glfwWindow, double xpos, double ypos) {
    // Retrieve the Window instance
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(glfwWindow));

    // If the GUI wants to capture the mouse (or camera mode is disabled), do nothing.
    if (ImGui::GetIO().WantCaptureMouse || !win->cameraMode)
        return;

    if (win->firstMouse) {
        win->lastX = xpos;
        win->lastY = ypos;
        win->firstMouse = false;
    }

    float xoffset = xpos - win->lastX;
    float yoffset = win->lastY - ypos; // reversed since y goes from bottom to top
    win->lastX = xpos;
    win->lastY = ypos;

    localLastYpos = ypos;
    localLastXpos = xpos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    win->yaw += xoffset;
    win->pitch += yoffset;

    if (win->pitch > 89.0f)
        win->pitch = 89.0f;
    if (win->pitch < -89.0f)
        win->pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(win->yaw)) * cos(glm::radians(win->pitch));
    front.y = sin(glm::radians(win->pitch));
    front.z = sin(glm::radians(win->yaw)) * cos(glm::radians(win->pitch));
    win->cameraFront = glm::normalize(front);
}

void Window::scrollCallback(GLFWwindow* glfwWindow, double xoffset, double yoffset) {
    auto* win = static_cast<Window*>(glfwGetWindowUserPointer(glfwWindow));
    win->fov -= static_cast<float>(yoffset);
    if (win->fov < 1.0f)
        win->fov = 1.0f;
    if (win->fov > 45.0f)
        win->fov = 45.0f;
}