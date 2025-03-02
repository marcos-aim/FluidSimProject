cmake_minimum_required(VERSION 3.29)
project(FluidSim LANGUAGES CXX CUDA)

# set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# add executable
add_executable(FluidSim ../FluidSimProject/src/main.cpp
        src/core/Renderer.cpp
        src/core/Window.cpp
        src/core/Timer.cpp
        src/simulation/SPH.cpp
        src/simulation/SPH.h
        src/simulation/SPHKernels.cu)

# include directories for headers
target_include_directories(FluidSim PRIVATE ../FluidSimProject/include)
target_include_directories(FluidSim PRIVATE ../FluidSimProject/src/core)
target_include_directories(FluidSim PRIVATE ../FluidSimProject/src/simulation)
target_include_directories(FluidSim PRIVATE ../FluidSimProject/include/ImGui)

# Link GLAD, GLFW, OpenGL, and ImGui libraries
target_link_libraries(FluidSim
        ${CMAKE_SOURCE_DIR}/lib/glad.lib
        ${CMAKE_SOURCE_DIR}/lib/glfw3_mt.lib
        ${CMAKE_SOURCE_DIR}/lib/imgui.lib  # Link ImGui static library
        opengl32
)

find_package(CUDAToolkit REQUIRED)
target_include_directories(FluidSim PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(FluidSim PRIVATE ${CUDAToolkit_LIBRARIES})

set_target_properties(FluidSim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)