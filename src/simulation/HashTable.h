#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <cuda_runtime.h>

// Prime constants for hashing
#define HASH_K1 19349669u  // Prime number 1
#define HASH_K2 73856093u  // Prime number 2
#define HASH_K3 83492791u  // Prime number 3

// 27 neighbor offsets for 3D grid traversal stored in constant memory
__constant__ int3 offsets3D[27] = {
    { -1, -1, -1 }, { -1, -1,  0 }, { -1, -1,  1 },
    { -1,  0, -1 }, { -1,  0,  0 }, { -1,  0,  1 },
    { -1,  1, -1 }, { -1,  1,  0 }, { -1,  1,  1 },
    {  0, -1, -1 }, {  0, -1,  0 }, {  0, -1,  1 },
    {  0,  0, -1 }, {  0,  0,  0 }, {  0,  0,  1 },
    {  0,  1, -1 }, {  0,  1,  0 }, {  0,  1,  1 },
    {  1, -1, -1 }, {  1, -1,  0 }, {  1, -1,  1 },
    {  1,  0, -1 }, {  1,  0,  0 }, {  1,  0,  1 },
    {  1,  1, -1 }, {  1,  1,  0 }, {  1,  1,  1 }
};

// Converts a float3 position into an integer cell coordinate based on the given radius.
__device__ __forceinline__ int3 GetCell3D(float3 position, float radius)
{
    return make_int3(floorf(position.x / radius),
                     floorf(position.y / radius),
                     floorf(position.z / radius));
}

// Hashes the given cell coordinate to a single unsigned integer.
__device__ __forceinline__ unsigned int HashCell3D(int3 cell)
{
    return (cell.x * HASH_K1) + (cell.y * HASH_K2) + (cell.z * HASH_K3);
}

// Converts a hash value to a key within the table by taking modulo with tableSize.
__device__ __forceinline__ unsigned int KeyFromHash(unsigned int hash, unsigned int tableSize)
{
    return hash % tableSize;
}

#endif // HASHTABLE_H
