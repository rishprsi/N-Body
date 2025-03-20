#ifndef DIRECT_SUM_KERNEL_H_
#define DIRECT_SUM_KERNEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "constants.h"
#include "err.h"

#define BLOCK_SIZE 256

// Initializes OpenCV for creating a video
cv::VideoWriter video("nbody.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));

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

// Creates a scaled window for the video
Vector scaleToWindow(Vector pos)
{

    double scaleX = WINDOW_HEIGHT / NBODY_HEIGHT;
    double scaleY = WINDOW_WIDTH / NBODY_WIDTH;
    return {(pos.x - 0) * scaleX + WINDOW_WIDTH / 2, (pos.y - 0) * scaleY + WINDOW_HEIGHT / 2};
}

// For each iteration stores the frame generated
void storeFrame(Obj *objects, int n, int id)
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

// Checks validity of initial 
bool argCheck(int nBodies, int sim, int iter)
{
    if (sim < 0 || sim > 1)
    {
        std::cout << "ERROR: simulation isn't valid" << std::endl;
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

int main(int argc, char **argv)
{
    int nBodies = NUM_BODIES;
    int sim = 0;
    int iters = 300;
    if (argc == 4)
    {
        nBodies = atoi(argv[1]);
        sim = atoi(argv[2]);
        iters = atoi(argv[3]);
    }

    if (!argCheck(nBodies, sim, iters))
        return -1;

    Obj *h_bodies;
    // Initializes objects
    if (sim == 0)
    {
        h_bodies = initSpiralObjects(nBodies);
    }
    else
    {
        nBodies = 5;
        h_bodies = initCustomSystem();
    }

    // Memory allocation
    int bytes = nBodies * sizeof(Obj);

    Obj *d_bodies;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bodies, bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, bytes, cudaMemcpyHostToDevice));

    // Kernel configuration
    int blockSize = BLOCK_SIZE;
    int gridSize = ceil((double)nBodies / blockSize);
    int it = 0;

    // Run the simulation for iters iterations
    while (it < iters) // main loop
    {
        ForceTiledKernel<<<gridSize, blockSize>>>(d_bodies, nBodies);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaMemcpy(h_bodies, d_bodies, bytes, cudaMemcpyDeviceToHost));
        storeFrame(h_bodies, nBodies, ++it);
    }
    video.release();

    // free memories
    CHECK_CUDA_ERROR(cudaFree(d_bodies));
    free(h_bodies);

    CHECK_LAST_CUDA_ERROR();
    return 0;
}

#endif