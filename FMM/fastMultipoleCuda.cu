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

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "constants.h"
#include "err.h"
#include "fastMultipoleCuda.cuh"
#include "fastMultipole_kernel.cuh"

// Global video writer for visualization
cv::VideoWriter video;

// Helper function to store frames
void storeFrame(Body *bodies, int nBodies, int frameNum) {
    cv::Mat img(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    
    if (frameNum == 0) {
        std::string filename = "fmm_simulation.avi";
        video.open(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, img.size(), true);
        if (!video.isOpened()) {
            std::cerr << "Could not open the output video file for write" << std::endl;
            return;
        }
    }
    
    for (int i = 0; i < nBodies; i++) {
        Body &b = bodies[i];
        double x = (b.position.x - (-NBODY_WIDTH / 2)) * WINDOW_WIDTH / NBODY_WIDTH;
        double y = (b.position.y - (-NBODY_HEIGHT / 2)) * WINDOW_HEIGHT / NBODY_HEIGHT;
        
        if (x >= 0 && x < WINDOW_WIDTH && y >= 0 && y < WINDOW_HEIGHT) {
            double radius = std::max(1.0, std::log10(b.mass) / 3.0);
            cv::circle(img, cv::Point(x, y), radius, cv::Scalar(255, 255, 255), -1);
        }
    }
    
    video.write(img);
}

// Check command line arguments
bool checkArgs(int nBodies, int sim, int iter) {
    if (nBodies < 1) {
        std::cout << "ERROR: need to have at least 1 body" << std::endl;
        return false;
    }

    if (sim < 0 || sim > 3) {
        std::cout << "ERROR: simulation doesn't exist" << std::endl;
        return false;
    }

    if (iter < 1) {
        std::cout << "ERROR: need to have at least 1 iteration" << std::endl;
        return false;
    }

    return true;
}

// Constructor
FastMultipoleCuda::FastMultipoleCuda(int n) : nBodies(n), maxDepth(MAX_DEPTH) {
    // Use fixed size for cells like in Barnes-Hut
    maxCells = MAX_CELLS;
    nCells = 0;
    
    // Allocate host memory
    h_bodies = new Body[nBodies];
    h_cells = new Cell[maxCells];
    h_cellCount = new int;
    h_sortedIndex = new int[nBodies];
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bodies, sizeof(Body) * nBodies));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bodies_buffer, sizeof(Body) * nBodies));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cells, sizeof(Cell) * maxCells));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cellCount, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_sortedIndex, sizeof(int) * nBodies));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_mutex, sizeof(int) * maxCells));
    
    // Initialize cell count to zero
    *h_cellCount = 0;
}

// Destructor
FastMultipoleCuda::~FastMultipoleCuda() {
    // Free host memory
    delete[] h_bodies;
    delete[] h_cells;
    delete h_cellCount;
    delete[] h_sortedIndex;
    
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
    
    // Execute FMM algorithm steps
    buildTree();
    computeMultipoles();
    translateMultipoles();
    computeLocalExpansions();
    evaluateLocalExpansions();
    directEvaluation();
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