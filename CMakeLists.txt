cmake_minimum_required(VERSION 3.29)
project(FluidSimProject)

# set C++ standard
set(CMAKE_CXX_STANDARD 20)

# add executable
add_executable(FluidSim ../FluidSimProject/src/main.cpp)

# include directories for headers
target_include_directories(FluidSim PRIVATE ../FluidSimProject/include)

# link GLAD, GLFW, and OpenGL libraries
target_link_libraries(FluidSim ${CMAKE_SOURCE_DIR}/lib/libglad.a ${CMAKE_SOURCE_DIR}/lib/libglfw3.a opengl32)
