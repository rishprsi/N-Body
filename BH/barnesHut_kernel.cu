#ifndef BARNES_HUT_KERNEL_
#define BARNES_HUT_KERNEL_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "constants.h"
#include "barnesHutCuda.cuh"
#include "barnesHut_kernel.cuh"

//==============================================================================
// TYPES AND CONSTANTS
//==============================================================================

// Physics parameters
struct NBodyPhysicsParams {
    double gravityConstant;      // Gravitational constant G
    double timestep;             // Time step dt
    double approxThreshold;      // Barnes-Hut threshold theta
    double collisionThreshold;   // Collision detection threshold
    double softeningFactor;      // Softening factor epsilon
};

//==============================================================================
// UTILITY FUNCTIONS
//==============================================================================

// Calculate distance between two points
__device__ inline double nbody_distance(const Vector& point_a, const Vector& point_b) {
    double delta_x = point_a.x - point_b.x;
    double delta_y = point_a.y - point_b.y;
    return sqrt(delta_x * delta_x + delta_y * delta_y);
}

// Check collision between two bodies
__device__ inline bool nbody_detect_collision(const Vector& pos_a, double radius_a, 
                                           const Vector& pos_b) {
    double min_dist = 2.0 * radius_a + COLLISION_TH;
    return nbody_distance(pos_a, pos_b) < min_dist;
}

// Calculate midpoint between two vectors
__device__ inline Vector nbody_midpoint(const Vector& pt1, const Vector& pt2) {
    Vector result;
    result.x = 0.5 * (pt1.x + pt2.x);
    result.y = 0.5 * (pt1.y + pt2.y);
    return result;
}

// Determine quadrant for a point
__device__ int nbody_determine_quadrant(const Vector& min_bound, const Vector& max_bound, 
                                     double x_pos, double y_pos) {
    Vector mid = nbody_midpoint(min_bound, max_bound);
    
    if (x_pos < mid.x) {
        return (y_pos > mid.y) ? QuadrantDir::TOP_LEFT : QuadrantDir::BOTTOM_LEFT;
    } else {
        return (y_pos > mid.y) ? QuadrantDir::TOP_RIGHT : QuadrantDir::BOTTOM_RIGHT;
    }
}

//==============================================================================
// INITIALIZATION PHASE
//==============================================================================

// Initialize tree nodes
__global__ void nbody_initialize_tree(
    Node* tree_nodes,     // Tree nodes array
    int* mutex_array,     // Mutex array
    int node_count,       // Total nodes
    int body_count        // Total bodies
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < node_count) {
        // Set default values
        tree_nodes[thread_id].topLeft = {INFINITY, -INFINITY};
        tree_nodes[thread_id].botRight = {-INFINITY, INFINITY};
        tree_nodes[thread_id].centerMass = {-1.0, -1.0};
        tree_nodes[thread_id].totalMass = 0.0;
        tree_nodes[thread_id].isLeaf = true;
        tree_nodes[thread_id].start = -1;
        tree_nodes[thread_id].end = -1;
        
        mutex_array[thread_id] = 0;
    }

    // Set up root node
    if (thread_id == 0) {
        tree_nodes[0].start = 0;
        tree_nodes[0].end = body_count - 1;
    }
}

// Compute simulation bounding box
__global__ void nbody_compute_bounds(
    Node* tree_nodes,     // Tree nodes array
    Body* bodies,         // Bodies array
    int* mutex_array,     // Mutex array
    int body_count        // Total bodies
) {
    // Shared memory for reduction
    __shared__ double x_min_shared[BLOCK_SIZE];
    __shared__ double y_max_shared[BLOCK_SIZE]; 
    __shared__ double x_max_shared[BLOCK_SIZE];
    __shared__ double y_min_shared[BLOCK_SIZE];

    int tx = threadIdx.x;
    int body_idx = blockIdx.x * blockDim.x + tx;

    // Initialize with extreme values
    x_min_shared[tx] = INFINITY;
    y_max_shared[tx] = -INFINITY;
    x_max_shared[tx] = -INFINITY;
    y_min_shared[tx] = INFINITY;

    __syncthreads();

    if (body_idx < body_count) {
        Body current_body = bodies[body_idx];
        
        double pos_x = current_body.position.x;
        double pos_y = current_body.position.y;
        
        x_min_shared[tx] = pos_x;
        y_max_shared[tx] = pos_y;
        x_max_shared[tx] = pos_x;
        y_min_shared[tx] = pos_y;
    }

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        
        if (tx < stride) {
            x_min_shared[tx] = fminf(x_min_shared[tx], x_min_shared[tx + stride]);
            y_max_shared[tx] = fmaxf(y_max_shared[tx], y_max_shared[tx + stride]);
            x_max_shared[tx] = fmaxf(x_max_shared[tx], x_max_shared[tx + stride]);
            y_min_shared[tx] = fminf(y_min_shared[tx], y_min_shared[tx + stride]);
        }
    }

    // Update global bounds
    if (tx == 0) {
        // Lock before update
        while (atomicCAS(mutex_array, 0, 1) != 0) { }
        
        const double DOMAIN_PADDING = 1.0e10;
        
        // Update boundaries with padding
        tree_nodes[0].topLeft.x = fminf(tree_nodes[0].topLeft.x, x_min_shared[0] - DOMAIN_PADDING);
        tree_nodes[0].topLeft.y = fmaxf(tree_nodes[0].topLeft.y, y_max_shared[0] + DOMAIN_PADDING);
        tree_nodes[0].botRight.x = fmaxf(tree_nodes[0].botRight.x, x_max_shared[0] + DOMAIN_PADDING);
        tree_nodes[0].botRight.y = fminf(tree_nodes[0].botRight.y, y_min_shared[0] - DOMAIN_PADDING);
        
        // Release lock
        atomicExch(mutex_array, 0);
    }
}

//==============================================================================
// TREE CONSTRUCTION PHASE
//==============================================================================

// Set boundaries for a child node
__device__ void nbody_set_boundaries(
    Node& child_node,             // Child node
    const Vector& min_corner,     // Min corner
    const Vector& max_corner,     // Max corner
    int quadrant                  // Quadrant 
) {
    Vector mid = nbody_midpoint(min_corner, max_corner);
    
    switch (quadrant) {
    case QuadrantDir::TOP_RIGHT:
        child_node.topLeft = {mid.x, min_corner.y};
        child_node.botRight = {max_corner.x, mid.y};
        break;
        
    case QuadrantDir::TOP_LEFT:
        child_node.topLeft = {min_corner.x, min_corner.y};
        child_node.botRight = {mid.x, mid.y};
        break;
        
    case QuadrantDir::BOTTOM_LEFT:
        child_node.topLeft = {min_corner.x, mid.y};
        child_node.botRight = {mid.x, max_corner.y};
        break;
        
    case QuadrantDir::BOTTOM_RIGHT:
        child_node.topLeft = {mid.x, mid.y};
        child_node.botRight = {max_corner.x, max_corner.y};
        break;
    }
}

// Warp-level reduction for center of mass
__device__ void nbody_warp_reduce_mass(
    volatile double* mass_vals,           // Mass values
    volatile double2* weighted_pos_vals,  // Weighted positions
    int lane_id                           // Lane ID
) {
    if (lane_id < 32) {
        // Step 1
        mass_vals[lane_id] += mass_vals[lane_id + 32];
        weighted_pos_vals[lane_id].x += weighted_pos_vals[lane_id + 32].x;
        weighted_pos_vals[lane_id].y += weighted_pos_vals[lane_id + 32].y;
        
        // Step 2
        mass_vals[lane_id] += mass_vals[lane_id + 16];
        weighted_pos_vals[lane_id].x += weighted_pos_vals[lane_id + 16].x;
        weighted_pos_vals[lane_id].y += weighted_pos_vals[lane_id + 16].y;
        
        // Step 3
        mass_vals[lane_id] += mass_vals[lane_id + 8];
        weighted_pos_vals[lane_id].x += weighted_pos_vals[lane_id + 8].x;
        weighted_pos_vals[lane_id].y += weighted_pos_vals[lane_id + 8].y;
        
        // Step 4
        mass_vals[lane_id] += mass_vals[lane_id + 4];
        weighted_pos_vals[lane_id].x += weighted_pos_vals[lane_id + 4].x;
        weighted_pos_vals[lane_id].y += weighted_pos_vals[lane_id + 4].y;
        
        // Step 5
        mass_vals[lane_id] += mass_vals[lane_id + 2];
        weighted_pos_vals[lane_id].x += weighted_pos_vals[lane_id + 2].x;
        weighted_pos_vals[lane_id].y += weighted_pos_vals[lane_id + 2].y;
        
        // Step 6 (final)
        mass_vals[lane_id] += mass_vals[lane_id + 1];
        weighted_pos_vals[lane_id].x += weighted_pos_vals[lane_id + 1].x;
        weighted_pos_vals[lane_id].y += weighted_pos_vals[lane_id + 1].y;
    }
}

// Calculate center of mass for a node
__device__ void nbody_calculate_com(
    Body* bodies,             // Bodies array
    Node& node,               // Node to update
    double* mass_shared,      // Mass values
    double2* com_shared,      // Center of mass
    int first_idx,            // First body index
    int last_idx              // Last body index
) {
    int tx = threadIdx.x;
    int body_count = last_idx - first_idx + 1;
    
    // Distribute work
    int items_per_thread = (body_count + blockDim.x - 1) / blockDim.x;
    int start_idx = first_idx + tx * items_per_thread;
    int end_idx = min(start_idx + items_per_thread - 1, last_idx);
    
    // Initialize accumulators
    double total_mass = 0.0;
    double mass_weighted_x = 0.0;
    double mass_weighted_y = 0.0;
    
    // Process assigned bodies
    for (int i = start_idx; i <= end_idx; i++) {
        Body& body = bodies[i];
        double body_mass = body.mass;
        
        total_mass += body_mass;
        mass_weighted_x += body_mass * body.position.x;
        mass_weighted_y += body_mass * body.position.y;
    }
    
    // Store in shared memory
    mass_shared[tx] = total_mass;
    com_shared[tx] = make_double2(mass_weighted_x, mass_weighted_y);
    
    // Block-level reduction
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        __syncthreads();
        
        if (tx < s) {
            mass_shared[tx] += mass_shared[tx + s];
            com_shared[tx].x += com_shared[tx + s].x;
            com_shared[tx].y += com_shared[tx + s].y;
        }
    }
    
    // Final warp reduction
    __syncthreads();
    if (tx < 32) {
        nbody_warp_reduce_mass(mass_shared, com_shared, tx);
    }
    
    // Finalize calculation
    __syncthreads();
    if (tx == 0 && mass_shared[0] > 0) {
        node.totalMass = mass_shared[0];
        node.centerMass.x = com_shared[0].x / mass_shared[0];
        node.centerMass.y = com_shared[0].y / mass_shared[0];
    }
}

// Count bodies in each quadrant
__device__ void nbody_count_per_quadrant(
    Body* bodies,              // Bodies array
    const Vector& min_corner,  // Min corner
    const Vector& max_corner,  // Max corner
    int* count_array,          // Count array
    int first_idx,             // First body index
    int last_idx               // Last body index
) {
    int tx = threadIdx.x;
    
    // Initialize counts
    if (tx < 4) {
        count_array[tx] = 0;
    }
    
    __syncthreads();
    
    // Count bodies by quadrant
    for (int i = first_idx + tx; i <= last_idx; i += blockDim.x) {
        Body body = bodies[i];
        int quadrant = nbody_determine_quadrant(min_corner, max_corner, 
                                          body.position.x, body.position.y);
        
        atomicAdd(&count_array[quadrant - 1], 1);
    }
    
    __syncthreads();
}

// Calculate offset for each quadrant
__device__ void nbody_calculate_offsets(
    int* count_array,    // Count array
    int start_index      // Starting index
) {
    int tx = threadIdx.x;
    
    if (tx < 4) {
        int offset = start_index;
        
        for (int i = 0; i < tx; i++) {
            offset += count_array[i];
        }
        
        count_array[tx + 4] = offset;
    }
    
    __syncthreads();
}

// Sort bodies by quadrant
__device__ void nbody_sort_bodies(
    Body* src_bodies,            // Source bodies
    Body* dst_bodies,            // Destination bodies
    const Vector& min_corner,    // Min corner
    const Vector& max_corner,    // Max corner
    int* count_array,            // Count array
    int first_idx,               // First body index
    int last_idx                 // Last body index
) {
    int* offset_array = &count_array[4];
    
    for (int i = first_idx + threadIdx.x; i <= last_idx; i += blockDim.x) {
        Body body = src_bodies[i];
        
        int quadrant = nbody_determine_quadrant(min_corner, max_corner, 
                                          body.position.x, body.position.y);
        
        int dest_idx = atomicAdd(&offset_array[quadrant - 1], 1);
        
        dst_bodies[dest_idx] = body;
    }
    
    __syncthreads();
}

// Build quadtree 
__global__ void nbody_build_quadtree(
    Node* tree_nodes,      // Tree nodes
    Body* src_bodies,      // Source bodies
    Body* dst_bodies,      // Destination bodies
    int node_idx,          // Start node index
    int node_count,        // Node count
    int body_count,        // Body count
    int leaf_threshold     // Leaf threshold
) {
    // Shared memory
    __shared__ int quadrant_data[8];  // Counts + offsets
    __shared__ double mass_shared[BLOCK_SIZE];
    __shared__ double2 com_shared[BLOCK_SIZE];
    
    int tx = threadIdx.x;
    node_idx += blockIdx.x;
    
    if (node_idx >= node_count) return;
    
    Node& current_node = tree_nodes[node_idx];
    int first_idx = current_node.start;
    int last_idx = current_node.end;
    Vector min_corner = current_node.topLeft;
    Vector max_corner = current_node.botRight;
    
    if (first_idx == -1 || last_idx == -1) return;
    
    // Calculate center of mass
    nbody_calculate_com(src_bodies, current_node, mass_shared, com_shared, first_idx, last_idx);
    
    // Handle leaf nodes
    if (node_idx >= leaf_threshold || first_idx == last_idx) {
        for (int i = first_idx + tx; i <= last_idx; i += blockDim.x) {
            dst_bodies[i] = src_bodies[i];
        }
        return;
    }
    
    // Count and sort bodies
    nbody_count_per_quadrant(src_bodies, min_corner, max_corner, quadrant_data, first_idx, last_idx);
    nbody_calculate_offsets(quadrant_data, first_idx);
    nbody_sort_bodies(src_bodies, dst_bodies, min_corner, max_corner, quadrant_data, first_idx, last_idx);
    
    // Create child nodes
    if (tx == 0) {
        // Child indices
        int tr_idx = (node_idx * 4) + 1;  // Top-right
        int tl_idx = (node_idx * 4) + 2;  // Top-left
        int bl_idx = (node_idx * 4) + 3;  // Bottom-left
        int br_idx = (node_idx * 4) + 4;  // Bottom-right
        
        // Set boundaries
        nbody_set_boundaries(tree_nodes[tr_idx], min_corner, max_corner, QuadrantDir::TOP_RIGHT);
        nbody_set_boundaries(tree_nodes[tl_idx], min_corner, max_corner, QuadrantDir::TOP_LEFT);
        nbody_set_boundaries(tree_nodes[bl_idx], min_corner, max_corner, QuadrantDir::BOTTOM_LEFT);
        nbody_set_boundaries(tree_nodes[br_idx], min_corner, max_corner, QuadrantDir::BOTTOM_RIGHT);
        
        current_node.isLeaf = false;
        
        // Set body ranges
        if (quadrant_data[0] > 0) {  // Top-right
            tree_nodes[tr_idx].start = first_idx;
            tree_nodes[tr_idx].end = first_idx + quadrant_data[0] - 1;
        }
        
        if (quadrant_data[1] > 0) {  // Top-left
            tree_nodes[tl_idx].start = first_idx + quadrant_data[0];
            tree_nodes[tl_idx].end = first_idx + quadrant_data[0] + quadrant_data[1] - 1;
        }
        
        if (quadrant_data[2] > 0) {  // Bottom-left
            tree_nodes[bl_idx].start = first_idx + quadrant_data[0] + quadrant_data[1];
            tree_nodes[bl_idx].end = first_idx + quadrant_data[0] + quadrant_data[1] + quadrant_data[2] - 1;
        }
        
        if (quadrant_data[3] > 0) {  // Bottom-right
            tree_nodes[br_idx].start = first_idx + quadrant_data[0] + quadrant_data[1] + quadrant_data[2];
            tree_nodes[br_idx].end = last_idx;
        }
        
        // Process children recursively
        nbody_build_quadtree<<<4, BLOCK_SIZE>>>(
            tree_nodes, dst_bodies, src_bodies, node_idx * 4 + 1, node_count, body_count, leaf_threshold);
    }
}

//==============================================================================
// FORCE CALCULATION AND INTEGRATION
//==============================================================================

// Apply gravitational force
__device__ void nbody_apply_gravity(
    const Body& body,       // Target body
    const Node& node,       // Source node
    Vector& force_out       // Output force
) {
    // Skip invalid/colliding nodes
    if (node.centerMass.x == -1 || 
        nbody_detect_collision(body.position, body.radius, node.centerMass)) {
        return;
    }
    
    // Calculate distance vector
    double r_x = node.centerMass.x - body.position.x;
    double r_y = node.centerMass.y - body.position.y;
    
    // Calculate force with softening
    double r_squared = r_x * r_x + r_y * r_y;
    double r_softened = sqrt(r_squared + E * E);
    double inv_r_cubed = 1.0 / (r_softened * r_softened * r_softened);
    
    double force_magnitude = GRAVITY * body.mass * node.totalMass * inv_r_cubed;
    
    force_out.x += r_x * force_magnitude;
    force_out.y += r_y * force_magnitude;
}

// Recursive force calculation
__device__ void nbody_tree_force(
    Node* tree_nodes,       // Tree nodes
    const Body& body,       // Target body
    Vector& force_out,      // Output force
    int node_idx,           // Current node
    int node_count,         // Total nodes
    double cell_size        // Cell size
) {
    if (node_idx >= node_count) {
        return;
    }
    
    Node node = tree_nodes[node_idx];
    
    // Leaf node case
    if (node.isLeaf) {
        nbody_apply_gravity(body, node, force_out);
        return;
    }
    
    // Apply Barnes-Hut approximation
    double distance = nbody_distance(body.position, node.centerMass);
    double size_distance_ratio = cell_size / distance;
    
    if (size_distance_ratio < THETA) {
        nbody_apply_gravity(body, node, force_out);
        return;
    }
    
    // Recursively process children
    double child_size = cell_size * 0.5;
    nbody_tree_force(tree_nodes, body, force_out, (node_idx * 4) + 1, node_count, child_size);
    nbody_tree_force(tree_nodes, body, force_out, (node_idx * 4) + 2, node_count, child_size);
    nbody_tree_force(tree_nodes, body, force_out, (node_idx * 4) + 3, node_count, child_size);
    nbody_tree_force(tree_nodes, body, force_out, (node_idx * 4) + 4, node_count, child_size);
}

// Calculate forces and update positions
__global__ void nbody_calculate_forces(
    Node* tree_nodes,       // Tree nodes
    Body* bodies,           // Bodies array
    int node_count,         // Total nodes
    int body_count          // Total bodies
) {
    int body_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double domain_size = tree_nodes[0].botRight.x - tree_nodes[0].topLeft.x;
    
    if (body_idx < body_count) {
        Body& body = bodies[body_idx];
        
        if (body.isDynamic) {
            // Reset acceleration
            body.acceleration = {0.0, 0.0};
            
            // Calculate force
            Vector net_force = {0.0, 0.0};
            nbody_tree_force(tree_nodes, body, net_force, 0, node_count, domain_size);
            
            // Convert to acceleration (F=ma → a=F/m)
            body.acceleration.x = net_force.x / body.mass;
            body.acceleration.y = net_force.y / body.mass;
            
            // Update velocity
            body.velocity.x += body.acceleration.x * DT;
            body.velocity.y += body.acceleration.y * DT;
            
            // Update position
            body.position.x += body.velocity.x * DT;
            body.position.y += body.velocity.y * DT;
        }
    }
}

#endif