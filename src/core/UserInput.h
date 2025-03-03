#ifndef USERINPUT_H
#define USERINPUT_H

struct UserInput {
    // Box settings
    float boxSizeX = 6.0f;
    float boxSizeY = 6.0f;
    float boxSizeZ = 5.0f;

    // Particle settings
    float particleR = 0.04f;
    int sphereSlices = 6;
    int sphereStacks = 5;

    // SPH settings
    int particleCount = 10000;
    float restingDensity = 1000.0f;
    float viscosityMultiplier = 1.0f;
    float mass = 0.2f;
    float gasConstant = 1.0f;
    float h = 0.15f;
    float g = -9.8f;
    float tension = 0.2f;

    // Simulation controls
    bool runSimulation = false;

    float dt = 0.2f;
};

#endif // USERINPUT_H
