/*
   Copyright 2023 Hsin-Hung Wu

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include <math.h>
#include "constants.h"
#include "fastMultipole_kernel.cuh"

// Helper device functions for complex arithmetic
__device__ Complex complexAdd(Complex a, Complex b) {
    return {a.real + b.real, a.imag + b.imag};
}

__device__ Complex complexMul(Complex a, Complex b) {
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

__device__ Complex complexScale(Complex a, double scale) {
    return {a.real * scale, a.imag * scale};
}

// Get quadrant for a position relative to a center
__device__ int getQuadrant(Vector position, Vector center) {
    int quadrant = 0;
    if (position.x >= center.x) {
        if (position.y >= center.y) {
            quadrant = 0; // Top-right
        } else {
            quadrant = 3; // Bottom-right
        }
    } else {
        if (position.y >= center.y) {
            quadrant = 1; // Top-left
        } else {
            quadrant = 2; // Bottom-left
        }
    }
    return quadrant;
}

// Custom atomic operations for double precision values
__device__ double atomicMin(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(min(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

__device__ double atomicMax(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(max(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

// Compute bounding box for all bodies
__global__ void ComputeBoundingBoxKernel(Body *bodies, Cell *cells, int *mutex, int nBodies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBodies) return;
    
    Body body = bodies[idx];
    
    // Update root cell bounds using atomic operations
    if (idx == 0) {
        // Initialize root cell
        cells[0].center = {0, 0};
        cells[0].size = NBODY_WIDTH;
        cells[0].parent = -1;
        cells[0].children[0] = cells[0].children[1] = cells[0].children[2] = cells[0].children[3] = -1;
        cells[0].bodyStart = 0;
        cells[0].bodyCount = nBodies;
        cells[0].isLeaf = true;
        cells[0].totalMass = 0.0;
        
        // Initialize multipole and local expansions
        for (int i = 0; i < P; i++) {
            cells[0].multipole[i] = {0, 0};
            cells[0].local[i] = {0, 0};
        }
        
        // Initialize bounds to extreme values
        cells[0].minBound = {INFINITY, INFINITY};
        cells[0].maxBound = {-INFINITY, -INFINITY};
    }
    
    // Ensure root cell is initialized before updating bounds
    __syncthreads();
    
    // Update bounding box using atomic operations (following BH style)
    atomicMin(&cells[0].minBound.x, body.position.x);
    atomicMin(&cells[0].minBound.y, body.position.y);
    atomicMax(&cells[0].maxBound.x, body.position.x);
    atomicMax(&cells[0].maxBound.y, body.position.y);
}

// Build the quadtree
__global__ void BuildTreeKernel(Body *bodies, Cell *cells, int *cellCount, int *sortedIndex, int *mutex, int nBodies, int maxDepth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBodies) return;
    
    // Initialize sorted index
    sortedIndex[idx] = idx;
    
    // Create root cell if this is the first thread
    if (idx == 0) {
        int rootIdx = atomicAdd(cellCount, 1);
        cells[rootIdx].center = {0, 0};
        cells[rootIdx].size = NBODY_WIDTH;
        cells[rootIdx].parent = -1;
        cells[rootIdx].bodyStart = 0;
        cells[rootIdx].bodyCount = nBodies;
        cells[rootIdx].isLeaf = (nBodies <= MAX_PARTICLES_PER_LEAF);
        
        for (int i = 0; i < 4; i++) {
            cells[rootIdx].children[i] = -1;
        }
        
        // Initialize multipole and local expansions
        for (int i = 0; i < P; i++) {
            cells[rootIdx].multipole[i] = {0, 0};
            cells[rootIdx].local[i] = {0, 0};
        }
    }
    
    __syncthreads();
    
    // Insert body into the tree
    Body body = bodies[idx];
    int cellIdx = 0; // Start at root
    int depth = 0;
    
    while (depth < maxDepth && !cells[cellIdx].isLeaf) {
        // Determine which quadrant the body belongs to
        int quadrant = getQuadrant(body.position, cells[cellIdx].center);
        
        // Check if child cell exists
        if (cells[cellIdx].children[quadrant] == -1) {
            // Create new child cell
            int newCellIdx = atomicAdd(cellCount, 1);
            cells[cellIdx].children[quadrant] = newCellIdx;
            
            // Calculate new center and size
            double halfSize = cells[cellIdx].size / 2.0;
            Vector center = cells[cellIdx].center;
            
            if (quadrant == 0) { // Top-right
                center.x += halfSize / 2.0;
                center.y += halfSize / 2.0;
            } else if (quadrant == 1) { // Top-left
                center.x -= halfSize / 2.0;
                center.y += halfSize / 2.0;
            } else if (quadrant == 2) { // Bottom-left
                center.x -= halfSize / 2.0;
                center.y -= halfSize / 2.0;
            } else { // Bottom-right
                center.x += halfSize / 2.0;
                center.y -= halfSize / 2.0;
            }
            
            // Initialize new cell
            cells[newCellIdx].center = center;
            cells[newCellIdx].size = halfSize;
            cells[newCellIdx].parent = cellIdx;
            cells[newCellIdx].bodyStart = 0;
            cells[newCellIdx].bodyCount = 0;
            cells[newCellIdx].isLeaf = true;
            
            for (int i = 0; i < 4; i++) {
                cells[newCellIdx].children[i] = -1;
            }
            
            // Initialize multipole and local expansions
            for (int i = 0; i < P; i++) {
                cells[newCellIdx].multipole[i] = {0, 0};
                cells[newCellIdx].local[i] = {0, 0};
            }
        }
        
        // Move to child cell
        cellIdx = cells[cellIdx].children[quadrant];
        depth++;
    }
    
    // Add body to leaf cell
    int bodyIdx = atomicAdd(&cells[cellIdx].bodyCount, 1);
    if (bodyIdx == 0) {
        cells[cellIdx].bodyStart = idx;
    }
    
    // If leaf is full, mark it as non-leaf for next iteration
    if (cells[cellIdx].bodyCount > MAX_PARTICLES_PER_LEAF && depth < maxDepth) {
        cells[cellIdx].isLeaf = false;
    }
}

// Compute multipole expansions for leaf cells
__global__ void ComputeMultipolesKernel(Body *bodies, Cell *cells, int *sortedIndex, int nCells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCells) return;
    
    Cell cell = cells[idx];
    
    // Only process leaf cells
    if (cell.isLeaf && cell.bodyCount > 0) {
        // Compute multipole expansion for this cell
        computeMultipoleExpansion(bodies, cell.bodyStart, cell.bodyCount, cells[idx].multipole, cell.center);
    }
}

// Compute multipole expansion for a group of bodies
__device__ void computeMultipoleExpansion(Body *bodies, int start, int count, Complex *multipole, Vector center) {
    // Initialize multipole coefficients
    for (int i = 0; i < P; i++) {
        multipole[i] = {0, 0};
    }
    
    // Monopole term (p=0) is just the total mass
    double totalMass = 0;
    for (int i = 0; i < count; i++) {
        Body body = bodies[start + i];
        totalMass += body.mass;
    }
    multipole[0] = {totalMass, 0};
    
    // Higher order terms
    for (int i = 0; i < count; i++) {
        Body body = bodies[start + i];
        
        // Convert to complex coordinates relative to cell center
        double dx = body.position.x - center.x;
        double dy = body.position.y - center.y;
        Complex z = {dx, dy};
        
        // Compute powers of z
        Complex zpow = {1, 0}; // z^0 = 1
        
        for (int p = 1; p < P; p++) {
            // z^p = z^(p-1) * z
            zpow = complexMul(zpow, z);
            
            // Add contribution to multipole coefficient
            Complex contrib = complexScale(zpow, body.mass);
            multipole[p] = complexAdd(multipole[p], contrib);
        }
    }
}

// Translate multipole expansions from children to parent
__global__ void TranslateMultipolesKernel(Cell *cells, int nCells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCells) return;
    
    Cell cell = cells[idx];
    
    // Skip leaf cells and the root
    if (cell.isLeaf || cell.parent == -1) return;
    
    // Translate multipole expansion to parent
    int parentIdx = cell.parent;
    translateMultipole(cell.multipole, cells[parentIdx].multipole, cell.center, cells[parentIdx].center);
}

// Translate a multipole expansion from source to target
__device__ void translateMultipole(Complex *source, Complex *target, Vector sourceCenter, Vector targetCenter) {
    // Compute translation vector
    double dx = sourceCenter.x - targetCenter.x;
    double dy = sourceCenter.y - targetCenter.y;
    Complex z0 = {dx, dy};
    
    // For each target multipole coefficient
    for (int p = 0; p < P; p++) {
        Complex sum = {0, 0};
        
        // Combine source multipoles with appropriate binomial coefficients
        for (int k = 0; k <= p; k++) {
            // Binomial coefficient C(p,k)
            int binomial = 1;
            for (int i = 1; i <= k; i++) {
                binomial = binomial * (p - i + 1) / i;
            }
            
            // z0^(p-k) * source[k] * C(p,k)
            Complex zpow = {1, 0}; // z0^0 = 1
            for (int i = 0; i < p-k; i++) {
                zpow = complexMul(zpow, z0);
            }
            
            Complex term = complexMul(zpow, source[k]);
            term = complexScale(term, binomial);
            sum = complexAdd(sum, term);
        }
        
        // Add to target multipole
        target[p] = complexAdd(target[p], sum);
    }
}

// Compute local expansions
__global__ void ComputeLocalExpansionsKernel(Cell *cells, int nCells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCells) return;
    
    // Start from the root and propagate local expansions downward
    if (idx == 0) { // Root cell
        // Root's local expansion is zero (no external influences)
        for (int i = 0; i < P; i++) {
            cells[0].local[i] = {0, 0};
        }
    }
    
    __syncthreads();
    
    Cell cell = cells[idx];
    
    // Skip leaf cells
    if (cell.isLeaf) return;
    
    // For each child, translate parent's local expansion plus siblings' multipole expansions
    for (int childIdx = 0; childIdx < 4; childIdx++) {
        int child = cell.children[childIdx];
        if (child == -1) continue;
        
        // First, translate parent's local expansion to child
        for (int i = 0; i < P; i++) {
            cells[child].local[i] = cells[idx].local[i];
        }
        
        // Then, for each sibling, translate its multipole expansion to child's local expansion
        for (int siblingIdx = 0; siblingIdx < 4; siblingIdx++) {
            int sibling = cell.children[siblingIdx];
            if (sibling == -1 || sibling == child) continue;
            
            translateMultipoleToLocal(cells[sibling].multipole, cells[child].local, 
                                     cells[sibling].center, cells[child].center);
        }
    }
}

// Translate multipole expansion to local expansion
__device__ void translateMultipoleToLocal(Complex *multipole, Complex *local, Vector multipoleCenter, Vector localCenter) {
    // Compute distance between centers
    double dx = multipoleCenter.x - localCenter.x;
    double dy = multipoleCenter.y - localCenter.y;
    double r2 = dx*dx + dy*dy;
    
    // Skip if centers are too close (MAC criterion)
    if (r2 < THETA * THETA) return;
    
    Complex z0 = {dx, dy};
    double r = sqrt(r2);
    
    // For each local coefficient
    for (int p = 0; p < P; p++) {
        Complex sum = {0, 0};
        
        // Combine multipole coefficients
        for (int k = 0; k < P-p; k++) {
            // Compute (1/z0)^(k+p+1) * multipole[k] / k!
            Complex zpow = {1, 0}; // (1/z0)^0 = 1
            for (int i = 0; i < k+p+1; i++) {
                // Compute 1/z0
                double denom = z0.real*z0.real + z0.imag*z0.imag;
                Complex invz = {z0.real/denom, -z0.imag/denom};
                zpow = complexMul(zpow, invz);
            }
            
            // Factorial k!
            int factorial = 1;
            for (int i = 2; i <= k; i++) {
                factorial *= i;
            }
            
            Complex term = complexMul(zpow, multipole[k]);
            term = complexScale(term, 1.0/factorial);
            sum = complexAdd(sum, term);
        }
        
        // Add to local expansion
        local[p] = complexAdd(local[p], sum);
    }
}

// Evaluate local expansions for all bodies
__global__ void EvaluateLocalExpansionsKernel(Body *bodies, Cell *cells, int *sortedIndex, int nBodies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBodies) return;
    
    Body body = bodies[idx];
    
    // Find the leaf cell containing this body
    int cellIdx = 0; // Start at root
    while (!cells[cellIdx].isLeaf) {
        int quadrant = getQuadrant(body.position, cells[cellIdx].center);
        cellIdx = cells[cellIdx].children[quadrant];
        if (cellIdx == -1) break; // Error case
    }
    
    if (cellIdx == -1) return; // Error case
    
    // Evaluate local expansion at body position
    Vector force = {0, 0};
    evaluateLocalExpansion(cells[cellIdx].local, cells[cellIdx].center, body.position, &force);
    
    // Store force for later use in force computation
    bodies[idx].acceleration = force;
}

// Evaluate local expansion at a position
__device__ void evaluateLocalExpansion(Complex *local, Vector center, Vector position, Vector *force) {
    // Convert to complex coordinates relative to cell center
    double dx = position.x - center.x;
    double dy = position.y - center.y;
    Complex z = {dx, dy};
    
    // Evaluate local expansion
    Complex potential = {0, 0};
    Complex field = {0, 0};
    
    // Compute powers of z
    Complex zpow = {1, 0}; // z^0 = 1
    
    for (int p = 0; p < P; p++) {
        // Add term to potential: local[p] * z^p
        Complex term = complexMul(local[p], zpow);
        potential = complexAdd(potential, term);
        
        // Add term to field: p * local[p] * z^(p-1)
        if (p > 0) {
            Complex fieldTerm = complexMul(local[p], zpow);
            fieldTerm = complexScale(fieldTerm, p);
            field = complexAdd(field, fieldTerm);
        }
        
        // z^(p+1) = z^p * z
        zpow = complexMul(zpow, z);
    }
    
    // Convert complex field to force vector (F = -grad(potential))
    force->x = -field.real * GRAVITY;
    force->y = -field.imag * GRAVITY;
}

// Direct evaluation for nearby particles
__global__ void DirectEvaluationKernel(Body *bodies, Cell *cells, int *sortedIndex, int nBodies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBodies) return;
    
    Body body1 = bodies[idx];
    Vector force = {0, 0};
    
    // Find the leaf cell containing this body
    int cellIdx = 0; // Start at root
    while (!cells[cellIdx].isLeaf) {
        int quadrant = getQuadrant(body1.position, cells[cellIdx].center);
        cellIdx = cells[cellIdx].children[quadrant];
        if (cellIdx == -1) break; // Error case
    }
    
    if (cellIdx == -1) return; // Error case
    
    // Compute direct interactions with bodies in the same leaf cell
    for (int i = 0; i < cells[cellIdx].bodyCount; i++) {
        int otherIdx = cells[cellIdx].bodyStart + i;
        if (otherIdx == idx) continue; // Skip self
        
        Body body2 = bodies[otherIdx];
        Vector directForce = {0, 0};
        computeDirectForce(body1, body2, &directForce);
        
        force.x += directForce.x;
        force.y += directForce.y;
    }
    
    // Add to acceleration from local expansion
    bodies[idx].acceleration.x += force.x;
    bodies[idx].acceleration.y += force.y;
}

// Compute direct force between two bodies
__device__ void computeDirectForce(Body body1, Body body2, Vector *force) {
    double dx = body2.position.x - body1.position.x;
    double dy = body2.position.y - body1.position.y;
    double r2 = dx*dx + dy*dy;
    
    // Add softening to prevent numerical instability
    r2 = fmax(r2, E * E);
    
    double r = sqrt(r2);
    double f = GRAVITY * body1.mass * body2.mass / r2;
    
    force->x = f * dx / r;
    force->y = f * dy / r;
}

// Final force computation and integration
__global__ void ComputeForcesAndUpdateKernel(Body *bodies, int nBodies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBodies) return;
    
    Body body = bodies[idx];
    
    // Skip non-dynamic bodies
    if (!body.isDynamic) return;
    
    // Update velocity using acceleration
    body.velocity.x += body.acceleration.x * DT / body.mass;
    body.velocity.y += body.acceleration.y * DT / body.mass;
    
    // Update position using velocity
    body.position.x += body.velocity.x * DT;
    body.position.y += body.velocity.y * DT;
    
    // Reset acceleration for next iteration
    body.acceleration = {0, 0};
    
    // Write back to global memory
    bodies[idx] = body;
}

// Add ResetMutexKernel implementation
__global__ void ResetMutexKernel(int *mutex, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        mutex[idx] = 0;
    }
}

// Reset cells kernel - follows BH's ResetKernel pattern
__global__ void ResetCellsKernel(Cell *cells, int *mutex, int nCells, int nBodies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nCells) {
        cells[idx].isLeaf = true;
        cells[idx].parent = -1;
        cells[idx].children[0] = cells[idx].children[1] = cells[idx].children[2] = cells[idx].children[3] = -1;
        cells[idx].bodyStart = -1;
        cells[idx].bodyCount = 0;
        cells[idx].totalMass = 0.0;
        cells[idx].minBound = {INFINITY, INFINITY};
        cells[idx].maxBound = {-INFINITY, -INFINITY};
        
        // Initialize multipole and local expansions
        for (int i = 0; i < P; i++) {
            cells[idx].multipole[i] = {0, 0};
            cells[idx].local[i] = {0, 0};
        }
        
        mutex[idx] = 0;
    }
    
    // Set up root node to handle all bodies (like BH implementation)
    if (idx == 0) {
        cells[idx].bodyStart = 0;
        cells[idx].bodyCount = nBodies;
    }
} 