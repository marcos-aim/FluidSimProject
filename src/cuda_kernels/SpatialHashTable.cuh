#ifndef SPATIAL_HASH_TABLE_H
#define SPATIAL_HASH_TABLE_H

#include <cuda_runtime.h>
#include <vector>

class SpatialHashTable {
public:
    SpatialHashTable(int gridResolution, float cellSize, int maxParticles);
    ~SpatialHashTable();

    void insertParticles(const float* d_positions, int numParticles);
    void clearHashTable();

    /**
     * Find neighbors for each particle within a given radius.
     *
     * @param d_positions Pointer to the device array of particle positions (float3).
     * @param numParticles Number of particles in the system.
     * @param searchRadius Radius within which neighbors are searched.
     * @param d_neighborList Output: device array storing neighbors for each particle.
     *                       Size: numParticles * maxNeighbors.
     * @param maxNeighbors Maximum number of neighbors per particle.
     */
    void findNeighbors(
        const float* d_positions,
        int numParticles,
        float searchRadius,
        int* d_neighborList,
        int maxNeighbors
    );

private:
    int hashFunction(int x, int y, int z) const;

    int* d_hashKeys;      // Keys for spatial hash
    int* d_particleIds;   // Particle IDs associated with keys
    int* d_cellStart;     // Start index for each cell in the hash table
    int* d_cellEnd;       // End index for each cell in the hash table

    int gridResolution;
    float cellSize;
    int tableSize;        // Total number of cells in the hash table
    int maxParticles;     // Maximum number of particles that can be stored
};

#endif // SPATIAL_HASH_TABLE_H
