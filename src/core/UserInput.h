#ifndef USERINPUT_H
#define USERINPUT_H

struct UserInput {
    // Box settings
    float boxSizeX = 5.0f;
    float boxSizeY = 4.0f;
    float boxSizeZ = 5.0f;

    // Particle settings
    float particleR = 0.04f;
    int sphereSlices = 6;
    int sphereStacks = 5;

    // SPH settings
    int particleCount = 50000;
    float restingDensity = 600.0f;      // Resting (target) density
    float viscosityMultiplier = 0.01f;    // Viscosity multiplier
    float mass = 1.0f;                   // Particle mass
    float pMult = 50.0f;                  // Pressure multiplier (can override gasConstant)
    float nearPMult = 1.5f;              // Near-pressure multiplier (can override gasConstant*0.5)
    float h = 0.25f;                     // Smoothing radius
    float g = -9.8f;                     // Gravity (typically negative)
    float tension = 0.2f;                // Surface tension
    float collisionDamping = 0.8f;       // Collision damping factor

    float gridCellSize = 0.055f;

    // Simulation controls
    bool runSimulation = false;
    float dt = 1.0f / 60.0f;                     // Time step

    bool rayMarchRender = false;
    bool debugSurface = false;
    float surfaceMinDensity = 420.f;
    float accumulationStepSize = 0.01f;
    float extinctionCoeffX = 1.0f;
    float extinctionCoeffY = 1.0f;
    float extinctionCoeffZ = 1.0f;
    float indexOfRefraction = 1.33f; // Maximum number of bounces (reflection/refraction)
    int maxBounces = 2;

    // Debug
    bool renderVoxelGrid = false;
    bool checkResting = false;
};

#endif // USERINPUT_H
