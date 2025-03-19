#ifndef BARNES_HUT_KERNELH_
#define BARNES_HUT_KERNELH_

#include <stdio.h>
#include <stdlib.h>
#include "barnesHutCuda.cuh"

// Quadrant directions for spatial partitioning
enum QuadrantDir {
    TOP_RIGHT = 1,
    TOP_LEFT = 2,
    BOTTOM_LEFT = 3,
    BOTTOM_RIGHT = 4
};

// Resets all tree nodes to initial state
__global__ void nbody_initialize_tree(Node* tree_nodes, int* mutex_array, 
                                    int node_count, int body_count);

// Finds the bounding box containing all bodies
__global__ void nbody_compute_bounds(Node* tree_nodes, Body* bodies, 
                                   int* mutex_array, int body_count);

// Builds the quadtree by recursively dividing space
__global__ void nbody_build_quadtree(Node* tree_nodes, Body* src_bodies, Body* dst_bodies,
                                   int node_idx, int node_count, int body_count, 
                                   int leaf_threshold);

// Calculates forces and updates body positions
__global__ void nbody_calculate_forces(Node* tree_nodes, Body* bodies, 
                                     int node_count, int body_count);

#endif