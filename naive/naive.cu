#ifndef DIRECT_SUM_KERNEL_H_
#define DIRECT_SUM_KERNEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "constants.h"
#include "err.h"

#define BLOCK_SIZE 256

// Holds Vector coordinates for the object
typedef struct
{
    double x;
    double y;
} Vector;

// Holds all the properties for the object
typedef struct
{
    bool isDynamic;
    double mass;
    double radius;
    Vector position;
    Vector velocity;
    Vector acceleration;

} Obj;

// Performance metrics
double totalKernelTime = 0.0;
double totalExecutionTime = 0.0;
double totalOperations = 0.0;
int numIterations = 0;

// Creates a scaled window for the video
Vector scaleToWindow(Vector pos)
{

    double scaleX = WINDOW_HEIGHT / NBODY_HEIGHT;
    double scaleY = WINDOW_WIDTH / NBODY_WIDTH;
    return {(pos.x - 0) * scaleX + WINDOW_WIDTH / 2, (pos.y - 0) * scaleY + WINDOW_HEIGHT / 2};
}

// For each iteration stores the frame generated
void storeFrame(Obj *objects, int n, int id, cv::VideoWriter &video)
{
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    cv::Scalar color;
    int radius;

    // Goes through all the objects to save it's position
    for (int i = 0; i < n; i++)
    {
        Vector pos = scaleToWindow(objects[i].position);
        cv::Point center(pos.x, pos.y);

        if (objects[i].mass >= HBL) // Distinguish large and small objects
        {
            color = cv::Scalar(0, 0, 255); // Red for stars
            radius = 5;
        }
        else
        {
            color = cv::Scalar(255, 255, 255); // White for planets
            radius = 1;
        }
        cv::circle(image, center, radius, color, -1);
    }
    // Writes to video
    video.write(image);
}

// Export positions to CSV
void exportPositionsToCSV(Obj *objects, int n, const std::string &filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return;
    }
    
    // Write header
    outputFile << "id,x,y,mass,velocity_x,velocity_y" << std::endl;
    
    // Write object data
    for (int i = 0; i < n; i++) {
        outputFile << i << ","
                  << objects[i].position.x << ","
                  << objects[i].position.y << ","
                  << objects[i].mass << ","
                  << objects[i].velocity.x << ","
                  << objects[i].velocity.y << std::endl;
    }
    
    outputFile.close();
}

// Creates a galxy like simulation where objects have initial velocities and a center mass
Obj *initSpiralObjects(int n)
{

    Obj *objects = new Obj[n];
    srand(time(NULL));
    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST;
    Vector centerPos = {CENTERX, CENTERY};

    // Generate all objects
    for (int i = 0; i < n - 1; ++i)
    {

        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        // Generate random distance from center within the given max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Calculate coordinates of the point
        double x = centerPos.x + radius * std::cos(angle);
        double y = centerPos.y + radius * std::sin(angle);

        Vector position = {x, y};

        // Calculates distance from the center
        double distance = sqrt(pow(x - centerPos.x, 2) + pow(y - centerPos.y, 2));
        Vector r = {position.x - centerPos.x, position.y - centerPos.y};
        Vector a = {r.x / distance, r.y / distance};

        // Calculate velocity vector components
        double esc = sqrt((GRAVITY * SUN_MASS) / (distance));
        Vector velocity = {-a.y * esc, a.x * esc};

        // Gives each object earth's mass and eart's diameter for consistency
        objects[i].isDynamic = true;
        objects[i].mass = EARTH_MASS;
        objects[i].radius = EARTH_DIA;
        objects[i].position = position;
        objects[i].velocity = velocity;
        objects[i].acceleration = {0.0, 0.0};
    }
    // Creates teh center of mass with Sun's properties
    objects[n - 1].isDynamic = false;
    objects[n - 1].mass = SUN_MASS;
    objects[n - 1].radius = SUN_DIA;
    objects[n - 1].position = centerPos;
    objects[n - 1].velocity = {0.0, 0.0};
    objects[n - 1].acceleration = {0.0, 0.0};
    return objects;
}

// Sets properties for a particular object
void setObject(Obj *objects, int i, bool isDynamic, double mass, double radius, Vector position, Vector velocity, Vector acceleration)
{
    objects[i].isDynamic = isDynamic;
    objects[i].mass = mass;
    objects[i].radius = radius;
    objects[i].position = position;
    objects[i].velocity = velocity;
    objects[i].acceleration = acceleration;
}

// Initializes a cusom system to play around with different masses and distances
Obj *initCustomSystem()
{

    Obj *objects = new Obj[5];
    setObject(objects, 0, true, 5.9740e24, 1.3927e6, {1.4960e11, 0}, {0, 2.9800e4}, {0, 0});
    setObject(objects, 1, true, 6.4190e23, 1.3927e6, {2.2790e11, 0}, {0, 2.4100e4}, {0, 0});
    setObject(objects, 2, true, 3.3020e23, 1.3927e6, {5.7900e10, 0}, {0, 4.7900e4}, {0, 0});
    setObject(objects, 3, true, 4.8690e24, 1.3927e6, {1.0820e11, 0}, {0, 3.5000e4}, {0, 0});
    setObject(objects, 4, false, 1.9890e30, 1.3927e6, {CENTERX, CENTERY}, {0, 0}, {0, 0});
    return objects;
}

// Create a colliding galaxies simulation
Obj *initCollidingGalaxies(int n)
{
    Obj *objects = new Obj[n];
    srand(time(NULL));
    
    int half = n / 2;
    
    // First galaxy (centered at left)
    Vector center1 = {CENTERX - 0.5e11, CENTERY};
    for (int i = 0; i < half - 1; ++i)
    {
        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        double radius = (MAX_DIST - MIN_DIST) * (rand() / (double)RAND_MAX) + MIN_DIST;
        
        double x = center1.x + radius * std::cos(angle);
        double y = center1.y + radius * std::sin(angle);
        
        Vector position = {x, y};
        double distance = sqrt(pow(x - center1.x, 2) + pow(y - center1.y, 2));
        Vector r = {position.x - center1.x, position.y - center1.y};
        Vector a = {r.x / distance, r.y / distance};
        
        double esc = sqrt((GRAVITY * SUN_MASS) / (distance));
        Vector velocity = {-a.y * esc + 1.5e4, a.x * esc}; // Add horizontal velocity
        
        objects[i].isDynamic = true;
        objects[i].mass = EARTH_MASS;
        objects[i].radius = EARTH_DIA;
        objects[i].position = position;
        objects[i].velocity = velocity;
        objects[i].acceleration = {0.0, 0.0};
    }
    
    // Central star of first galaxy
    objects[half - 1].isDynamic = false;
    objects[half - 1].mass = SUN_MASS;
    objects[half - 1].radius = SUN_DIA;
    objects[half - 1].position = center1;
    objects[half - 1].velocity = {1.5e4, 0.0}; // Moving right
    objects[half - 1].acceleration = {0.0, 0.0};
    
    // Second galaxy (centered at right)
    Vector center2 = {CENTERX + 0.5e11, CENTERY};
    for (int i = half; i < n - 1; ++i)
    {
        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        double radius = (MAX_DIST - MIN_DIST) * (rand() / (double)RAND_MAX) + MIN_DIST;
        
        double x = center2.x + radius * std::cos(angle);
        double y = center2.y + radius * std::sin(angle);
        
        Vector position = {x, y};
        double distance = sqrt(pow(x - center2.x, 2) + pow(y - center2.y, 2));
        Vector r = {position.x - center2.x, position.y - center2.y};
        Vector a = {r.x / distance, r.y / distance};
        
        double esc = sqrt((GRAVITY * SUN_MASS) / (distance));
        Vector velocity = {-a.y * esc - 1.5e4, a.x * esc}; // Add horizontal velocity
        
        objects[i].isDynamic = true;
        objects[i].mass = EARTH_MASS;
        objects[i].radius = EARTH_DIA;
        objects[i].position = position;
        objects[i].velocity = velocity;
        objects[i].acceleration = {0.0, 0.0};
    }
    
    // Central star of second galaxy
    objects[n - 1].isDynamic = false;
    objects[n - 1].mass = SUN_MASS;
    objects[n - 1].radius = SUN_DIA;
    objects[n - 1].position = center2;
    objects[n - 1].velocity = {-1.5e4, 0.0}; // Moving left
    objects[n - 1].acceleration = {0.0, 0.0};
    
    return objects;
}

// Checks validity of initial 
bool argCheck(int nBodies, int sim, int iter)
{
    if (sim < 0 || sim > 2)
    {
        std::cout << "ERROR: simulation isn't valid (0=spiral, 1=solar system, 2=colliding)" << std::endl;
        return false;
    }

    if (iter <= 1)
    {
        std::cout << "ERROR: need to more than one iterations" << std::endl;
        return false;
    }
    if (nBodies < 1)
    {
        std::cout << "ERROR: need to have at least 1 object in the system" << std::endl;
        return false;
    }

    return true;
}

// Kenel to compute distance between two positions for force degradation
__device__ double getDistance(Vector pos1, Vector pos2)
{

    return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2));
}

// Checks for collision with other objects
__device__ bool doesCollide(Obj &b1, Obj &b2)
{
    return b1.radius + b2.radius + COLLISION_TH > getDistance(b1.position, b2.position);
}

// Calculate FLOPS for naive N-body
double calculateFlops(int n) {
    return (double)n * (double)n * 20;
}

// Main kernel to calculate current position, velocity and acceleration
__global__ void ForceTiledKernel(Obj *objects, int n)
{
    __shared__ Obj Bds[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int i = bx * blockDim.x + tx;

    if (i < n)
    {
        Obj &bi = objects[i];
        // Initializes needed variables
        double fx = 0.0, fy = 0.0;
        bi.acceleration = {0.0, 0.0};
        for (int tile = 0; tile < gridDim.x; ++tile)
        {
            // Updates objects in shared memory for quick access
            Bds[tx] = objects[tile * blockDim.x + tx];
            __syncthreads();
            // Cumulates forcess from each other particle in the simulation using tiling
            for (int b = 0; b < BLOCK_SIZE; ++b)
            {
                int j = tile * blockDim.x + b;
                if (j < n)
                {
                    Obj bj = Bds[b];
                    if (!doesCollide(bi, bj) && bi.isDynamic)
                    {
                        // Calculates distances and changes the force based on the gravity and distance of the object
                        Vector rij = {bj.position.x - bi.position.x, bj.position.y - bi.position.y};
                        double r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (E * E));
                        double f = (GRAVITY * bi.mass * bj.mass) / (r * r * r + (E * E));
                        Vector force = {rij.x * f, rij.y * f};
                        fx += (force.x / bi.mass);
                        fy += (force.y / bi.mass);
                    }
                }
            }
            __syncthreads();
        }
        // Calculates current position, acceleration and velocity of the object
        bi.acceleration.x += fx;
        bi.acceleration.y += fy;
        bi.velocity.x += bi.acceleration.x * DT;
        bi.velocity.y += bi.acceleration.y * DT;
        bi.position.x += bi.velocity.x * DT;
        bi.position.y += bi.velocity.y * DT;
    }
}

// Print performance metrics
void printPerformanceMetrics() {
    std::cout << "\nPerformance Metrics for Naive N-Body Simulation:" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Total iterations measured: " << numIterations << std::endl;
    std::cout << "Total kernel execution time: " << totalKernelTime << " ms" << std::endl;
    std::cout << "Average kernel execution time: " << totalKernelTime / numIterations << " ms/iteration" << std::endl;
    std::cout << "Total execution time (incl. memory transfers): " << totalExecutionTime << " ms" << std::endl;
    std::cout << "Average execution time: " << totalExecutionTime / numIterations << " ms/iteration" << std::endl;
    
    // Calculate GFLOPS
    double totalFlops = totalOperations;
    double kernelGflops = totalFlops / (totalKernelTime * 1e6);
    double effectiveGflops = totalFlops / (totalExecutionTime * 1e6);
    
    std::cout << "Total floating-point operations: " << std::fixed << totalFlops << " ops" << std::endl;
    std::cout << "Kernel performance: " << kernelGflops << " GFLOPS" << std::endl;
    std::cout << "Effective performance: " << effectiveGflops << " GFLOPS" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
}

int main(int argc, char **argv)
{
    int nBodies = NUM_BODIES;
    int sim = 0;
    int iters = 300;
    int exportFreq = 10;
    
    if (argc >= 4)
    {
        nBodies = atoi(argv[1]);
        sim = atoi(argv[2]);
        iters = atoi(argv[3]);
    }

    if (!argCheck(nBodies, sim, iters))
        return -1;
        
    std::cout << "Running naive N-body simulation with " << nBodies << " bodies for " 
              << iters << " iterations" << std::endl;

    std::string simType;
    switch(sim) {
        case 0: simType = "spiral"; break;
        case 1: simType = "solar"; break;
        case 2: simType = "colliding"; break;
        default: simType = "custom";
    }
    
    system("mkdir -p videos");
    
    std::string videoFilename = "videos/" + simType + "_" + std::to_string(nBodies) + ".avi";
    cv::VideoWriter video(videoFilename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));
    
    Obj *h_bodies;
    // Initializes objects
    if (sim == 0)
    {
        h_bodies = initSpiralObjects(nBodies);
    }
    else if (sim == 1)
    {
        nBodies = 5;
        h_bodies = initCustomSystem();
    }
    else if (sim == 2)
    {
        h_bodies = initCollidingGalaxies(nBodies);
    }

    // Create output directory
    system("mkdir -p naive_output_data");

    // Memory allocation
    int bytes = nBodies * sizeof(Obj);

    Obj *d_bodies;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bodies, bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, bytes, cudaMemcpyHostToDevice));

    // Kernel configuration
    int blockSize = BLOCK_SIZE;
    int gridSize = (nBodies + blockSize - 1) / blockSize;

    // Reset performance counters
    totalKernelTime = 0.0;
    totalExecutionTime = 0.0;
    totalOperations = 0.0;
    numIterations = 0;

    // CUDA events for timing
    cudaEvent_t kernelStart, kernelStop, fullStart, fullStop;
    float kernelTimeMs, executionTimeMs;
    
    // Run first iteration (iteration 0) - warmup without timing
    ForceTiledKernel<<<gridSize, blockSize>>>(d_bodies, nBodies);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaMemcpy(h_bodies, d_bodies, bytes, cudaMemcpyDeviceToHost));
    storeFrame(h_bodies, nBodies, 0, video);
    
    // Export initial state
    std::string filename = "naive_output_data/positions_0.csv";
    exportPositionsToCSV(h_bodies, nBodies, filename);

    // Run the simulation for timed iterations
    std::cout << "Starting timed iterations..." << std::endl;
    
    for (int it = 1; it < iters; it++)
    {
        // Create timing events
        CHECK_CUDA_ERROR(cudaEventCreate(&kernelStart));
        CHECK_CUDA_ERROR(cudaEventCreate(&kernelStop));
        CHECK_CUDA_ERROR(cudaEventCreate(&fullStart));
        CHECK_CUDA_ERROR(cudaEventCreate(&fullStop));
        
        // Start full execution timer (including memory transfers)
        CHECK_CUDA_ERROR(cudaEventRecord(fullStart));
        
        // Start kernel timer
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStart));
        
        // Run kernel
        ForceTiledKernel<<<gridSize, blockSize>>>(d_bodies, nBodies);
        CHECK_LAST_CUDA_ERROR();
        
        // End kernel timer
        CHECK_CUDA_ERROR(cudaEventRecord(kernelStop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(kernelStop));
        
        // Copy results back to host
        CHECK_CUDA_ERROR(cudaMemcpy(h_bodies, d_bodies, bytes, cudaMemcpyDeviceToHost));
        
        // End full execution timer
        CHECK_CUDA_ERROR(cudaEventRecord(fullStop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(fullStop));
        
        // Calculate elapsed time
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTimeMs, kernelStart, kernelStop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&executionTimeMs, fullStart, fullStop));
        
        // Update performance metrics
        totalKernelTime += kernelTimeMs;
        totalExecutionTime += executionTimeMs;
        totalOperations += calculateFlops(nBodies);
        numIterations++;
        
        // Cleanup timing events
        CHECK_CUDA_ERROR(cudaEventDestroy(kernelStart));
        CHECK_CUDA_ERROR(cudaEventDestroy(kernelStop));
        CHECK_CUDA_ERROR(cudaEventDestroy(fullStart));
        CHECK_CUDA_ERROR(cudaEventDestroy(fullStop));
        
        // Store frame and export positions
        storeFrame(h_bodies, nBodies, it, video);
        
        if (it % exportFreq == 0) {
            std::string filename = "naive_output_data/positions_" + std::to_string(it) + ".csv";
            exportPositionsToCSV(h_bodies, nBodies, filename);
        }
    }
    
    // Print performance metrics
    printPerformanceMetrics();
    
    // Export performance data to CSV
    std::ofstream perfOutput("naive_performance_results.csv");
    if (perfOutput.is_open()) {
        perfOutput << "bodies,iterations,total_kernel_time_ms,avg_kernel_time_ms,total_execution_time_ms,avg_execution_time_ms,total_flops,kernel_gflops,effective_gflops" << std::endl;
        double kernelGflops = totalOperations / (totalKernelTime * 1e6);
        double effectiveGflops = totalOperations / (totalExecutionTime * 1e6);
        
        perfOutput << nBodies << ","
                  << numIterations << ","
                  << totalKernelTime << ","
                  << (totalKernelTime / numIterations) << ","
                  << totalExecutionTime << ","
                  << (totalExecutionTime / numIterations) << ","
                  << totalOperations << ","
                  << kernelGflops << ","
                  << effectiveGflops << std::endl;
        
        perfOutput.close();
        std::cout << "Performance data written to naive_performance_results.csv" << std::endl;
    } else {
        std::cerr << "ERROR: Could not open naive_performance_results.csv for writing" << std::endl;
    }

    video.release();

    // free memories
    CHECK_CUDA_ERROR(cudaFree(d_bodies));
    delete[] h_bodies;

    return 0;
}

#endif