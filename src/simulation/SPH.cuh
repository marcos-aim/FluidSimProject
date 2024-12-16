//
// Created by maiba on 12/14/2024.
//

#ifndef SPH_CUH
#define SPH_CUH
#include "Window.h"


// SPH Class
class SPH {
public:
    // Box settings
    float boxSizeX;
    float boxSizeY;
    float boxSizeZ;

    // SPH settings
    int particleCount;
    float restingDensity;
    float viscosityMultiplier;
    float mass;
    float gasConstant;
    float h;
    float g;
    float tension;

    // Simulation controls
    bool runSimulation;

    // Constructor
    explicit SPH(const UserInput& userInput)
        : boxSizeX(userInput.boxSizeX),
          boxSizeY(userInput.boxSizeY),
          boxSizeZ(userInput.boxSizeZ),
          particleCount(userInput.particleCount),
          restingDensity(userInput.restingDensity),
          viscosityMultiplier(userInput.viscosityMultiplier),
          mass(userInput.mass),
          gasConstant(userInput.gasConstant),
          h(userInput.h),
          g(userInput.g),
          tension(userInput.tension),
          runSimulation(userInput.runSimulation) {}

    void initParticles(int cubeWidth);
};

struct Particle
{
    glm::vec3 position, velocity, acceleration, force;
    float density;
    float pressure;
    uint16_t hash;
};

#endif //SPH_CUH
