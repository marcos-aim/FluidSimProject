#ifndef FLUIDSIM_SPH_H
#define FLUIDSIM_SPH_H

#include "glad/glad.h"
#include <vector>
#include <glm/glm.hpp>
#include "Window.h" // Include the user input struct

#define MAX_NEIGHBORS 64
#define SPAWN_SEPARATION 0.1f // Define particle separation
#define BOUNDARY_BUFFER 0.1f

enum class StartingPosition {
    BOTTOM_CORNER, // All particles start at the bottom corner of the box
    TOP_CORNER,    // All particles start at the top corner of the box
    TWO_CUBES,     // Particles form two separate cubes
    CENTER_CUBE,   // Particles form a single cube at the center of the box
    RANDOM_FILL    // Particles are randomly distributed throughout the box
};

class SPHSimulation {
public:
    // Constructor and Destructor
    SPHSimulation(const UserInput& input);
    ~SPHSimulation();

    // Main Functions
    void initParticles(StartingPosition startType);
    void update(float deltaTime);

    // Query Functions
    std::vector<glm::vec3> h_positions;
    const std::vector<glm::vec3>& getParticlePositions() const;
    const std::vector<float>& getTimingRecord() const;

    // Status
    bool isRunning{};
    static constexpr int maxTimings = 100;

private:
    // SPH Algorithm Steps
    void runUpdateKernels(float deltaTime);
    void computeHashes(int threadsPerBlock, int blocksPerGrid);
    void computeDensityAndPressure();
    void computeForces();
    void moveParticles(float deltaTime);
    void applyBoundaryConditions();

    // SPH Parameters
    float smoothingRadius;
    float mass;
    float gasConstant;
    float viscosity;
    float surfaceTension;
    float gravity;
    glm::vec3 boxSize{};

    // Particle Data
    int numParticles;

    // Timing
    std::vector<float> timingRecord;

    // CUDA Device Data
    glm::vec3* d_positions{};
    glm::vec3* d_velocities{};
    glm::vec3* d_accelerations{};
    float* d_densities{};
    float* d_pressures{};

    // Uniform Grid Data
    int* d_hashes{};
    int* d_indices{};
    int* d_cellStart{};
    int* d_cellEnd{};
    glm::ivec3 meshDims{};
    float cellSize;
};

// DEBUG:
void printNeighborList(int* d_neighborList, int* d_neighborCounts, int numParticles, int maxNeighbors);
void printHashTable(int* d_hashes, int* d_indices, glm::vec3* d_positions, int numParticles);


#endif // FLUIDSIM_SPH_H
