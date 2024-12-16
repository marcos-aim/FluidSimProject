cmake_minimum_required(VERSION 3.29)
project(FluidSim LANGUAGES CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

# add executable
add_executable(FluidSim ../FluidSimProject/src/main.cpp
        src/core/Renderer.cpp
        src/core/Renderer.h
        src/core/Window.cpp
        src/core/Window.h
        src/simulation/SPH.cu
        src/simulation/SPH.cuh
        src/core/Timer.cpp
        src/core/Timer.h)

# include directories for headers
target_include_directories(FluidSim PRIVATE ../FluidSimProject/include)
target_include_directories(FluidSim PRIVATE ../FluidSimProject/src/core)
target_include_directories(FluidSim PRIVATE ../FluidSimProject/src/simulation)
target_include_directories(FluidSim PRIVATE ../FluidSimProject/include/ImGui)

# Include CUDA headers and link CUDA libraries
find_package(CUDAToolkit REQUIRED)
target_include_directories(FluidSim PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Link GLAD, GLFW, OpenGL, and ImGui libraries
target_link_libraries(FluidSim
        ${CMAKE_SOURCE_DIR}/lib/libglad.a
        ${CMAKE_SOURCE_DIR}/lib/libglfw3.a
        ${CMAKE_SOURCE_DIR}/lib/libimgui.a  # Link ImGui static library
        opengl32
)