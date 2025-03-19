

#ifndef BARNES_HUT_KERNELH_
#define BARNES_HUT_KERNELH_

#include "barnesHutCuda.cuh"

//==============================================================================
// TYPE DEFINITIONS
//==============================================================================

/**
 * Direction enumeration for clarity in quadrant operations
 */
enum QuadrantDir {
    TOP_RIGHT = 1,
    TOP_LEFT = 2,
    BOTTOM_LEFT = 3,
    BOTTOM_RIGHT = 4
};

//==============================================================================
// KERNEL FUNCTION DECLARATIONS
//==============================================================================

/**
 * Initializes tree nodes at the beginning of simulation
 * 
 * @param tree_nodes Array of Barnes-Hut quadtree nodes
 * @param mutex_array Mutex array for thread synchronization
 * @param node_count Total number of nodes in the tree
 * @param body_count Total number of bodies in the simulation
 */
__global__ void nbody_initialize_tree(Node* tree_nodes, int* mutex_array, 
                                    int node_count, int body_count);

/**
 * Computes the bounding box for all bodies in the simulation
 * 
 * @param tree_nodes Array of Barnes-Hut quadtree nodes
 * @param bodies Array of bodies in the simulation
 * @param mutex_array Mutex array for thread synchronization
 * @param body_count Total number of bodies
 */
__global__ void nbody_compute_bounds(Node* tree_nodes, Body* bodies, 
                                   int* mutex_array, int body_count);

/**
 * Builds a quadtree for n-body simulation using the Barnes-Hut approach
 * 
 * @param tree_nodes Array of Barnes-Hut quadtree nodes
 * @param src_bodies Source array of bodies
 * @param dst_bodies Destination buffer for bodies
 * @param node_idx Starting node index
 * @param node_count Total number of nodes in the tree
 * @param body_count Total number of bodies in the simulation
 * @param leaf_threshold Threshold index for leaf nodes
 */
__global__ void nbody_build_quadtree(Node* tree_nodes, Body* src_bodies, Body* dst_bodies,
                                   int node_idx, int node_count, int body_count, 
                                   int leaf_threshold);

/**
 * Calculates forces on bodies using Barnes-Hut approximation and updates positions
 * 
 * @param tree_nodes Array of Barnes-Hut quadtree nodes
 * @param bodies Array of bodies to update
 * @param node_count Total number of nodes in the tree
 * @param body_count Total number of bodies in the simulation
 */
__global__ void nbody_calculate_forces(Node* tree_nodes, Body* bodies, 
                                     int node_count, int body_count);

#endif