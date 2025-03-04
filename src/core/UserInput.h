#ifndef USERINPUT_H
#define USERINPUT_H

struct UserInput {
    // Box settings
    float boxSizeX = 12.0f;
    float boxSizeY = 8.0f;
    float boxSizeZ = 12.0f;

    // Particle settings
    float particleR = 0.04f;
    int sphereSlices = 6;
    int sphereStacks = 5;

    // SPH settings
    int particleCount = 50000;
    float restingDensity = 67.0f;      // Resting (target) density
    float viscosityMultiplier = 0.34f;    // Viscosity multiplier
    float mass = 0.5f;                   // Particle mass
    float pMult = 50.0f;                  // Pressure multiplier (can override gasConstant)
    float nearPMult = 2.25f;              // Near-pressure multiplier (can override gasConstant*0.5)
    float h = 0.33f;                     // Smoothing radius
    float g = -9.8f;                     // Gravity (typically negative)
    float tension = 0.2f;                // Surface tension
    float collisionDamping = 0.95f;       // Collision damping factor

    // Simulation controls
    bool runSimulation = false;
    float dt = 1 / 60.f;                     // Time step
};

#endif // USERINPUT_H
