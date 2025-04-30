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

    glm::vec3 boxCenter(
    userInput.boxSizeX * 0.5f,
    userInput.boxSizeY * 0.5f,
    userInput.boxSizeZ * 0.5f
    );

    float aspect = float(width) / float(height);
    float fovRad = glm::radians(fov);

    float halfH = userInput.boxSizeY * 0.5f;
    float halfW = userInput.boxSizeX * 0.5f;

    float dY = halfH / tan(fovRad * 0.5f);
    float dX = halfW / (tan(fovRad * 0.5f) * aspect);
    float d  = glm::max(dX, dY) * 1.5f;

    cameraPos   = boxCenter + glm::vec3(0.0f, 0.0f, d);
    cameraFront = glm::normalize(boxCenter - cameraPos);

    cameraView       = glm::lookAt(cameraPos, boxCenter, cameraUp);
    cameraProjection = glm::perspective(fovRad, aspect, 0.1f, 100.0f);

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

// Start ImGui frame
void Window::beginFrame() {
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Disable mouse input when in camera mode
    io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
    io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse; // clear it first

    if (cameraMode) {
        io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
    }

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
    if (!win->cameraMode)
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
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(glfwWindow));
    if (!win->cameraMode)
        return;

    win->fov -= static_cast<float>(yoffset);
    if (win->fov < 1.0f)
        win->fov = 1.0f;
    if (win->fov > 45.0f)
        win->fov = 45.0f;
}