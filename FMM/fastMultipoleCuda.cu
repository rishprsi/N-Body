#include <iostream>
#include <cmath>
#include "constants.h"
#include "fastMultipole_kernel.cuh"
#include "err.h"

// Constructor
FastMultipoleCuda::FastMultipoleCuda(int n,int error_check) : nBodies(n), maxDepth(MAX_DEPTH) {
    // Use fixed size for cells like in Barnes-Hut
    maxCells = MAX_CELLS;
    nCells = 0;
    
    // Allocate host memory
    h_bodies = new Body[nBodies];
    h_cells = new Cell[maxCells];
    h_cellCount = new int;
    h_sortedIndex = new int[nBodies];
    error_flag = error_check;
    if (error_flag == 1){
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_b_naive, sizeof(Body) * n));
        h_b_naive = new Body[n];
    }
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bodies, sizeof(Body) * nBodies));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bodies_buffer, sizeof(Body) * nBodies));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cells, sizeof(Cell) * maxCells));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cellCount, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortedIndex, sizeof(int) * nBodies));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_mutex, sizeof(int) * maxCells));
    
    // Initialize cell count to zero
    *h_cellCount = 0;

    resetTimers();
}

void FastMultipoleCuda::resetTimers() {
    totalKernelTime = 0.0f;
    totalExecutionTime = 0.0f;
    iterationCount = 0;
    totalFlops = 0;
}

void FastMultipoleCuda::printPerformanceMetrics() {
    std::cout << "==== Barnes-Hut Performance Metrics (Excluding Warm-up Iteration) ====" << std::endl;
    std::cout << "Number of bodies: " << nBodies << std::endl;
    std::cout << "Total iterations measured: " << iterationCount << std::endl;
    std::cout << "Total kernel execution time: " << totalKernelTime << " ms" << std::endl;
    std::cout << "Average kernel time per iteration: " << getAverageKernelTime() << " ms" << std::endl;
    std::cout << "Total execution time (including memory transfers): " << totalExecutionTime << " ms" << std::endl;
    std::cout << "Average execution time per iteration: " << getAverageExecutionTime() << " ms" << std::endl;
    std::cout << "Memory transfer overhead: " << (totalExecutionTime - totalKernelTime) << " ms (" 
              << ((totalExecutionTime - totalKernelTime) / totalExecutionTime) * 100.0f << "%)" << std::endl;
    
    double gflops = totalFlops / (totalKernelTime * 1e6); // Convert to GFLOPS
    double effective_gflops = totalFlops / (totalExecutionTime * 1e6); // Effective GFLOPS including memory transfers
    std::cout << "Total FLOPS: " << totalFlops << std::endl;
    std::cout << "Kernel-only performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Effective performance (with memory transfers): " << effective_gflops << " GFLOPS" << std::endl;
    std::cout << "=============================================================" << std::endl;
}

// Destructor
FastMultipoleCuda::~FastMultipoleCuda() {
    // Free host memory
    delete[] h_bodies;
    delete[] h_cells;
    delete h_cellCount;
    delete[] h_sortedIndex;

    if (error_flag){
        delete[] h_b_naive;
        CHECK_CUDA_ERROR(cudaFree(d_b_naive));
    }
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_bodies));
    CHECK_CUDA_ERROR(cudaFree(d_bodies_buffer));
    CHECK_CUDA_ERROR(cudaFree(d_cells));
    CHECK_CUDA_ERROR(cudaFree(d_cellCount));
    CHECK_CUDA_ERROR(cudaFree(d_sortedIndex));
    CHECK_CUDA_ERROR(cudaFree(d_mutex));
}

// Initialize random bodies
void FastMultipoleCuda::initRandomBodies() {
    srand(time(NULL));
    
    for (int i = 0; i < nBodies - 1; i++) {
        double x = (2.0 * rand() / RAND_MAX - 1.0) * NBODY_WIDTH / 2;
        double y = (2.0 * rand() / RAND_MAX - 1.0) * NBODY_HEIGHT / 2;
        
        double vx = (2.0 * rand() / RAND_MAX - 1.0) * 1.0e4;
        double vy = (2.0 * rand() / RAND_MAX - 1.0) * 1.0e4;
        
        setBody(i, true, EARTH_MASS, EARTH_DIA, {x, y}, {vx, vy}, {0, 0});
    }
    
    // Add a central massive body
    setBody(nBodies - 1, false, SUN_MASS, SUN_DIA, {CENTERX, CENTERY}, {0, 0}, {0, 0});
}

// Initialize spiral galaxy
void FastMultipoleCuda::initSpiralBodies() {
    srand(time(NULL));
    
    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST;
    Vector centerPos = {CENTERX, CENTERY};
    
    for (int i = 0; i < nBodies - 1; i++) {
        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;
        
        double x = centerPos.x + radius * cos(angle);
        double y = centerPos.y + radius * sin(angle);
        
        Vector position = {x, y};
        double distance = sqrt((position.x - centerPos.x) * (position.x - centerPos.x) + 
                              (position.y - centerPos.y) * (position.y - centerPos.y));
        
        Vector r = {position.x - centerPos.x, position.y - centerPos.y};
        Vector a = {r.x / distance, r.y / distance};
        
        double esc = sqrt((GRAVITY * SUN_MASS) / distance);
        Vector velocity = {-a.y * esc, a.x * esc};
        
        setBody(i, true, EARTH_MASS, EARTH_DIA, position, velocity, {0, 0});
    }
    
    // Add central massive body
    setBody(nBodies - 1, false, SUN_MASS, SUN_DIA, centerPos, {0, 0}, {0, 0});
}

// Initialize colliding galaxies
void FastMultipoleCuda::initCollideGalaxy() {
    srand(time(NULL));
    
    int halfBodies = nBodies / 2;
    double maxDistance = MAX_DIST / 2;
    double minDistance = MIN_DIST;
    
    // First galaxy
    Vector center1 = {-MAX_DIST / 2, 0};
    for (int i = 0; i < halfBodies - 1; i++) {
        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;
        
        double x = center1.x + radius * cos(angle);
        double y = center1.y + radius * sin(angle);
        
        Vector position = {x, y};
        double distance = sqrt((position.x - center1.x) * (position.x - center1.x) + 
                              (position.y - center1.y) * (position.y - center1.y));
        
        Vector r = {position.x - center1.x, position.y - center1.y};
        Vector a = {r.x / distance, r.y / distance};
        
        double esc = sqrt((GRAVITY * SUN_MASS) / distance);
        Vector velocity = {-a.y * esc + 1.0e4, a.x * esc};
        
        setBody(i, true, EARTH_MASS, EARTH_DIA, position, velocity, {0, 0});
    }
    
    // Central body for first galaxy
    setBody(halfBodies - 1, true, SUN_MASS, SUN_DIA, center1, {1.0e4, 0}, {0, 0});
    
    // Second galaxy
    Vector center2 = {MAX_DIST / 2, 0};
    for (int i = halfBodies; i < nBodies - 1; i++) {
        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;
        
        double x = center2.x + radius * cos(angle);
        double y = center2.y + radius * sin(angle);
        
        Vector position = {x, y};
        double distance = sqrt((position.x - center2.x) * (position.x - center2.x) + 
                              (position.y - center2.y) * (position.y - center2.y));
        
        Vector r = {position.x - center2.x, position.y - center2.y};
        Vector a = {r.x / distance, r.y / distance};
        
        double esc = sqrt((GRAVITY * SUN_MASS) / distance);
        Vector velocity = {-a.y * esc - 1.0e4, a.x * esc};
        
        setBody(i, true, EARTH_MASS, EARTH_DIA, position, velocity, {0, 0});
    }
    
    // Central body for second galaxy
    setBody(nBodies - 1, true, SUN_MASS, SUN_DIA, center2, {-1.0e4, 0}, {0, 0});
}

// Initialize solar system
void FastMultipoleCuda::initSolarSystem() {
    // Earth
    setBody(0, true, 5.9740e24, 1.3927e6, {1.4960e11, 0}, {0, 2.9800e4}, {0, 0});
    // Mars
    setBody(1, true, 6.4190e23, 1.3927e6, {2.2790e11, 0}, {0, 2.4100e4}, {0, 0});
    // Mercury
    setBody(2, true, 3.3020e23, 1.3927e6, {5.7900e10, 0}, {0, 4.7900e4}, {0, 0});
    // Venus
    setBody(3, true, 4.8690e24, 1.3927e6, {1.0820e11, 0}, {0, 3.5000e4}, {0, 0});
    // Sun
    setBody(4, false, 1.9890e30, 1.3927e6, {CENTERX, CENTERY}, {0, 0}, {0, 0});
}

// Helper to set body properties
void FastMultipoleCuda::setBody(int i, bool isDynamic, double mass, double radius, Vector position, Vector velocity, Vector acceleration) {
    h_bodies[i].isDynamic = isDynamic;
    h_bodies[i].mass = mass;
    h_bodies[i].radius = radius;
    h_bodies[i].position = position;
    h_bodies[i].velocity = velocity;
    h_bodies[i].acceleration = acceleration;
}

// Setup method to initialize the simulation
void FastMultipoleCuda::setup(int sim) {
    if (sim == 0) {
        initSpiralBodies();
    } else if (sim == 1) {
        initRandomBodies();
    } else if (sim == 2) {
        initCollideGalaxy();
    } else {
        initSolarSystem();
    }
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, nBodies * sizeof(Body), cudaMemcpyHostToDevice));
    
    // Initialize cell count and reset cells
    *h_cellCount = 1; // Start with just the root cell
    CHECK_CUDA_ERROR(cudaMemcpy(d_cellCount, h_cellCount, sizeof(int), cudaMemcpyHostToDevice));
    
    // Reset cells array
    int blockSize = BLOCK_SIZE;
    int gridSize = (maxCells + blockSize - 1) / blockSize;
    ResetCellsKernel<<<gridSize, blockSize>>>(d_cells, d_mutex, maxCells, nBodies);
    CHECK_LAST_CUDA_ERROR();
}

// Main update method
void FastMultipoleCuda::update() {
    // Reset cell count
    *h_cellCount = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(d_cellCount, h_cellCount, sizeof(int), cudaMemcpyHostToDevice));
    
    // Reset mutex array
    int blockSize = BLOCK_SIZE;
    int gridSize = (maxCells + blockSize - 1) / blockSize;
    ResetMutexKernel<<<gridSize, blockSize>>>(d_mutex, maxCells);
    CHECK_LAST_CUDA_ERROR();

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Record start time for overall update
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    // Execute FMM algorithm steps
    buildTree();
    computeMultipoles();
    translateMultipoles();
    computeLocalExpansions();
    evaluateLocalExpansions();
    directEvaluation();

    // Record end time
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    totalKernelTime += milliseconds;
    iterationCount++;
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

// Build the tree structure
void FastMultipoleCuda::buildTree() {
    // Compute bounding box
    int blockSize = BLOCK_SIZE;
    int gridSize = (nBodies + blockSize - 1) / blockSize;
    ComputeBoundingBoxKernel<<<gridSize, blockSize>>>(d_bodies, d_cells, d_mutex, nBodies);
    CHECK_LAST_CUDA_ERROR();
    
    // Build the tree
    blockSize = 256;
    gridSize = (nBodies + blockSize - 1) / blockSize;
    BuildTreeKernel<<<gridSize, blockSize>>>(d_bodies, d_cells, d_cellCount, d_sortedIndex, d_mutex, nBodies, maxDepth);
    CHECK_LAST_CUDA_ERROR();
    
    // Get the number of cells
    CHECK_CUDA_ERROR(cudaMemcpy(h_cellCount, d_cellCount, sizeof(int), cudaMemcpyDeviceToHost));
    nCells = *h_cellCount;
}

// Compute multipole expansions
void FastMultipoleCuda::computeMultipoles() {
    int blockSize = BLOCK_SIZE;
    int gridSize = (nCells + blockSize - 1) / blockSize;
    
    ComputeMultipolesKernel<<<gridSize, blockSize>>>(d_bodies, d_cells, d_sortedIndex, nCells);
    CHECK_LAST_CUDA_ERROR();
}

// Translate multipoles from children to parents
void FastMultipoleCuda::translateMultipoles() {
    int blockSize = BLOCK_SIZE;
    int gridSize = (nCells + blockSize - 1) / blockSize;
    
    TranslateMultipolesKernel<<<gridSize, blockSize>>>(d_cells, nCells);
    CHECK_LAST_CUDA_ERROR();
}

// Compute local expansions
void FastMultipoleCuda::computeLocalExpansions() {
    int blockSize = BLOCK_SIZE;
    int gridSize = (nCells + blockSize - 1) / blockSize;
    
    ComputeLocalExpansionsKernel<<<gridSize, blockSize>>>(d_cells, nCells);
    CHECK_LAST_CUDA_ERROR();
}

// Evaluate local expansions
void FastMultipoleCuda::evaluateLocalExpansions() {
    int blockSize = BLOCK_SIZE;
    int gridSize = (nBodies + blockSize - 1) / blockSize;
    
    EvaluateLocalExpansionsKernel<<<gridSize, blockSize>>>(d_bodies, d_cells, d_sortedIndex, nBodies);
    CHECK_LAST_CUDA_ERROR();
}

// Direct evaluation for nearby particles
void FastMultipoleCuda::directEvaluation() {
    int blockSize = BLOCK_SIZE;
    int gridSize = (nBodies + blockSize - 1) / blockSize;
    
    DirectEvaluationKernel<<<gridSize, blockSize>>>(d_bodies, d_cells, d_sortedIndex, nBodies);
    CHECK_LAST_CUDA_ERROR();
    
    // Update positions and velocities
    ComputeForcesAndUpdateKernel<<<gridSize, blockSize>>>(d_bodies, nBodies);
    CHECK_LAST_CUDA_ERROR();
}

// Read bodies from device
void FastMultipoleCuda::readDeviceBodies() {
    CHECK_CUDA_ERROR(cudaMemcpy(h_bodies, d_bodies, nBodies * sizeof(Body), cudaMemcpyDeviceToHost));
}

// Get bodies
Body* FastMultipoleCuda::getBodies() {
    return h_bodies;
}

void FastMultipoleCuda::runNaive(){
    int blockSize = BLOCK_SIZE;
    dim3 gridSize = ceil((float)nBodies / blockSize);
    force_tile_kernel<<<gridSize, blockSize>>>(d_b_naive, nBodies);
    
}

Body* FastMultipoleCuda::readNaiveDeviceBodies()
{
    CHECK_CUDA_ERROR(cudaMemcpy(h_b_naive, d_b_naive, sizeof(Body) * nBodies, cudaMemcpyDeviceToHost));
    return h_b_naive;
}