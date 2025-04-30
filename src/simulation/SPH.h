#ifndef FLUIDSIM_SPH_H
#define FLUIDSIM_SPH_H

#include "UserInput.h"

#include "glad/glad.h"
#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

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
    void updateParameters(const UserInput& input);
    void initParticles(StartingPosition startType);
    void update(float deltaTime);

    // Query Functions
    std::vector<glm::vec3> h_positions;
    std::vector<glm::vec3>& getParticlePositions();
    float3* getDevicePositions();
    int getNumParticles();
    void initDensityGrid();

    // Debug
    void updateDensityGrid();
    void downloadDensityGrid(std::vector<float>& outBuffer);
    uint3 gridDims;
    float cellSize; // from input.gridCellSize

    // Status
    bool isRunning{};

private:
    void runUpdateKernels(float deltaTime);

    // SPH Parameters
    float smoothingRadius;
    float mass;
    float pMult;
    float nearPMult;
    float viscosityMult;
    float surfaceTension;
    float gravity;
    float restingDensity;
    float dt;
    float collisionDamping;
    float3 boxSize;

    // Particle Data
    int numParticles;

    // CUDA Device Data
    float3* d_positions;
    float3* d_predicted_positions;
    float3* d_velocities;
    float2* d_densities;
    uint3* d_indices;
    unsigned int* d_start_indices;

    // Voxel grid for density sampling
    cudaArray_t d_densityArray = nullptr; // holds the 3D float array
    cudaSurfaceObject_t densitySurf = 0; // for fast writes
    cudaTextureObject_t densityTex = 0; // for ray-march sampling
};

// DEBUG:
void printNeighborList(const int* d_neighborList, const int* d_neighborCounts, int numParticles, int maxNeighbors);
void printHashTable(int* d_hashes, int* d_indices, glm::vec3* d_positions, int numParticles);


#endif // FLUIDSIM_SPH_H
