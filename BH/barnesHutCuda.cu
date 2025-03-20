#include <iostream>
#include <cmath>
#include "barnesHut_kernel.cuh"
#include "constants.h"
#include "err.h"

BarnesHutCuda::BarnesHutCuda(int n) : nBodies(n)
{
    nNodes = MAX_NODES;
    leafLimit = MAX_NODES - N_LEAF;
    h_b = new Body[n];
    h_node = new Node[nNodes];

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_b, sizeof(Body) * n));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_node, sizeof(Node) * nNodes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_mutex, sizeof(int) * nNodes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_b_buffer, sizeof(Body) * n));
    
    resetTimers();
}

void BarnesHutCuda::resetTimers() {
    totalTime = 0.0f;
    treeInitTime = 0.0f;
    boundingBoxTime = 0.0f;
    treeConstructTime = 0.0f;
    forceCalcTime = 0.0f;
    totalFlops = 0;
}

void BarnesHutCuda::printPerformanceMetrics() {
    std::cout << "==== Barnes-Hut Performance Metrics ====" << std::endl;
    std::cout << "Number of bodies: " << nBodies << std::endl;
    std::cout << "Total execution time: " << totalTime << " ms" << std::endl;
    std::cout << "Tree initialization time: " << treeInitTime << " ms (" 
              << (treeInitTime / totalTime) * 100.0f << "%)" << std::endl;
    std::cout << "Bounding box computation time: " << boundingBoxTime << " ms (" 
              << (boundingBoxTime / totalTime) * 100.0f << "%)" << std::endl;
    std::cout << "Tree construction time: " << treeConstructTime << " ms (" 
              << (treeConstructTime / totalTime) * 100.0f << "%)" << std::endl;
    std::cout << "Force calculation time: " << forceCalcTime << " ms (" 
              << (forceCalcTime / totalTime) * 100.0f << "%)" << std::endl;
    
    double gflops = totalFlops / (totalTime * 1e6); // Convert to GFLOPS
    std::cout << "Total FLOPS: " << totalFlops << " (" << gflops << " GFLOPS)" << std::endl;
    std::cout << "=======================================" << std::endl;
}

BarnesHutCuda::~BarnesHutCuda()
{
    delete[] h_b;
    delete[] h_node;
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_node));
    CHECK_CUDA_ERROR(cudaFree(d_mutex));
    CHECK_CUDA_ERROR(cudaFree(d_b_buffer));
}

void BarnesHutCuda::resetCUDA()
{
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Record start time
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    int blockSize = BLOCK_SIZE;
    dim3 gridSize = ceil((float)nNodes / blockSize);
    nbody_initialize_tree<<<gridSize, blockSize>>>(d_node, d_mutex, nNodes, nBodies);
    
    // Record end time
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    treeInitTime += milliseconds;
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

void BarnesHutCuda::computeBoundingBoxCUDA()
{
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    int blockSize = BLOCK_SIZE;
    dim3 gridSize = ceil((float)nBodies / blockSize);
    nbody_compute_bounds<<<gridSize, blockSize>>>(d_node, d_b, d_mutex, nBodies);
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    boundingBoxTime += milliseconds;
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

void BarnesHutCuda::constructQuadTreeCUDA()
{
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    int blockSize = BLOCK_SIZE;
    dim3 gridSize = ceil((float)nBodies / blockSize);
    nbody_build_quadtree<<<1, blockSize>>>(d_node, d_b, d_b_buffer, 0, nNodes, nBodies, leafLimit);
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    treeConstructTime += milliseconds;
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

void BarnesHutCuda::computeForceCUDA()
{
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    int blockSize = 32;
    dim3 gridSize = ceil((float)nBodies / blockSize);
    nbody_calculate_forces<<<gridSize, blockSize>>>(d_node, d_b, nNodes, nBodies);
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    forceCalcTime += milliseconds;
    
    // Estimate FLOPS for force calculation (approximate)
    long long flopsPerStep = nBodies * log2(nBodies);
    totalFlops += flopsPerStep;
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

void BarnesHutCuda::initRandomBodies()
{
    srand(time(NULL));
    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST;
    Vector centerPos = {CENTERX, CENTERY};
    for (int i = 0; i < nBodies - 1; ++i)
    {

        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        // Generate random distance from center within the given max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Calculate coordinates of the point
        double x = centerPos.x + radius * std::cos(angle);
        double y = centerPos.y + radius * std::sin(angle);
        Vector position = {x, y};
        h_b[i].isDynamic = true;
        h_b[i].mass = EARTH_MASS;
        h_b[i].radius = EARTH_DIA;
        h_b[i].position = position;
        h_b[i].velocity = {0.0, 0.0};
        h_b[i].acceleration = {0.0, 0.0};
    }
    h_b[nBodies - 1].isDynamic = false;
    h_b[nBodies - 1].mass = SUN_MASS;
    h_b[nBodies - 1].radius = SUN_DIA;
    h_b[nBodies - 1].position = centerPos;
    h_b[nBodies - 1].velocity = {0.0, 0.0};
    h_b[nBodies - 1].acceleration = {0.0, 0.0};
}

void BarnesHutCuda::initSpiralBodies()
{

    srand(time(NULL));
    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST;
    Vector centerPos = {CENTERX, CENTERY};
    for (int i = 0; i < nBodies - 1; ++i)
    {

        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        // Generate random distance from center within the given max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Calculate coordinates of the point
        double x = centerPos.x + radius * std::cos(angle);
        double y = centerPos.y + radius * std::sin(angle);
        Vector position = {x, y};

        double distance = sqrt(pow(x - centerPos.x, 2) + pow(y - centerPos.y, 2));
        Vector r = {position.x - centerPos.x, position.y - centerPos.y};
        Vector a = {r.x / distance, r.y / distance};

        // Calculate velocity vector components
        double esc = sqrt((GRAVITY * SUN_MASS) / (distance));
        Vector velocity = {-a.y * esc, a.x * esc};

        h_b[i].isDynamic = true;
        h_b[i].mass = EARTH_MASS;
        h_b[i].radius = EARTH_DIA;
        h_b[i].position = position;
        h_b[i].velocity = velocity;
        h_b[i].acceleration = {0.0, 0.0};
    }
    h_b[nBodies - 1].isDynamic = false;
    h_b[nBodies - 1].mass = SUN_MASS;
    h_b[nBodies - 1].radius = SUN_DIA;
    h_b[nBodies - 1].position = centerPos;
    h_b[nBodies - 1].velocity = {0.0, 0.0};
    h_b[nBodies - 1].acceleration = {0.0, 0.0};
}

void BarnesHutCuda::initCollideGalaxy()
{

    srand(time(NULL));
    double maxDistance = MAX_DIST / 4.0;
    double minDistance = MIN_DIST;
    Vector centerPos = {-NBODY_WIDTH / 6.0, CENTERY};

    int galaxy1 = nBodies / 2;

    for (int i = 0; i < galaxy1 - 1; ++i)
    {

        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        // Generate random distance from center within the given max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Calculate coordinates of the point
        double x = centerPos.x + radius * std::cos(angle);
        double y = centerPos.y + radius * std::sin(angle);
        Vector position = {x, y};

        double distance = sqrt(pow(x - centerPos.x, 2) + pow(y - centerPos.y, 2));
        Vector r = {position.x - centerPos.x, position.y - centerPos.y};
        Vector a = {r.x / distance, r.y / distance};

        // Calculate velocity vector components
        double esc = sqrt((GRAVITY * SUN_MASS) / (distance));
        Vector velocity = {-a.y * esc, a.x * esc};

        h_b[i].isDynamic = true;
        h_b[i].mass = EARTH_MASS;
        h_b[i].radius = EARTH_DIA;
        h_b[i].position = position;
        h_b[i].velocity = velocity;
        h_b[i].acceleration = {0.0, 0.0};
    }
    h_b[galaxy1 - 1].isDynamic = true;
    h_b[galaxy1 - 1].mass = SUN_MASS;
    h_b[galaxy1 - 1].radius = SUN_DIA;
    h_b[galaxy1 - 1].position = centerPos;
    h_b[galaxy1 - 1].velocity = {0.0, 0.0};
    h_b[galaxy1 - 1].acceleration = {0.0, 0.0};

    centerPos = {NBODY_WIDTH / 6.0, CENTERY};

    for (int i = galaxy1; i < nBodies - 1; ++i)
    {

        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        // Generate random distance from center within the given max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Calculate coordinates of the point
        double x = centerPos.x + radius * std::cos(angle);
        double y = centerPos.y + radius * std::sin(angle);
        Vector position = {x, y};

        double distance = sqrt(pow(x - centerPos.x, 2) + pow(y - centerPos.y, 2));
        Vector r = {position.x - centerPos.x, position.y - centerPos.y};
        Vector a = {r.x / distance, r.y / distance};

        // Calculate velocity vector components
        double esc = sqrt((GRAVITY * SUN_MASS) / (distance));
        Vector velocity = {-a.y * esc, a.x * esc};

        h_b[i].isDynamic = true;
        h_b[i].mass = EARTH_MASS;
        h_b[i].radius = EARTH_DIA;
        h_b[i].position = position;
        h_b[i].velocity = velocity;
        h_b[i].acceleration = {0.0, 0.0};
    }
    h_b[nBodies - 1].isDynamic = true;
    h_b[nBodies - 1].mass = SUN_MASS;
    h_b[nBodies - 1].radius = SUN_DIA;
    h_b[nBodies - 1].position = centerPos;
    h_b[nBodies - 1].velocity = {0.0, 0.0};
    h_b[nBodies - 1].acceleration = {0.0, 0.0};
}

void BarnesHutCuda::setBody(int i, bool isDynamic, double mass, double radius, Vector position, Vector velocity, Vector acceleration)
{
    h_b[i].isDynamic = isDynamic;
    h_b[i].mass = mass;
    h_b[i].radius = radius;
    h_b[i].position = position;
    h_b[i].velocity = velocity;
    h_b[i].acceleration = acceleration;
}

void BarnesHutCuda::initSolarSystem()
{
    setBody(0, true, 5.9740e24, 1.3927e6, {1.4960e11, 0}, {0, 2.9800e4}, {0, 0});
    setBody(1, true, 6.4190e23, 1.3927e6, {2.2790e11, 0}, {0, 2.4100e4}, {0, 0});
    setBody(2, true, 3.3020e23, 1.3927e6, {5.7900e10, 0}, {0, 4.7900e4}, {0, 0});
    setBody(3, true, 4.8690e24, 1.3927e6, {1.0820e11, 0}, {0, 3.5000e4}, {0, 0});
    setBody(4, false, 1.9890e30, 1.3927e6, {CENTERX, CENTERY}, {0, 0}, {0, 0});
}

Body *BarnesHutCuda::getBodies()
{

    return h_b;
}

void BarnesHutCuda::readDeviceBodies()
{
    CHECK_CUDA_ERROR(cudaMemcpy(h_b, d_b, sizeof(Body) * nBodies, cudaMemcpyDeviceToHost));
}

void BarnesHutCuda::setup(int sim)
{
    if (sim == 0)
    {
        initSpiralBodies();
    }
    else if (sim == 1)
    {
        initRandomBodies();
    }
    else if (sim == 2)
    {
        initCollideGalaxy();
    }
    else
    {
        initSolarSystem();
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, sizeof(Body) * nBodies, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_node, h_node, sizeof(Node) * nNodes, cudaMemcpyHostToDevice));
}
void BarnesHutCuda::update()
{
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Record start time for overall update
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    resetCUDA();
    computeBoundingBoxCUDA();
    constructQuadTreeCUDA();
    computeForceCUDA();
    
    // Record end time
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    totalTime += milliseconds;
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}
