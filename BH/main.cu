#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "barnesHutCuda.cuh"
#include "constants.h"
#include "err.h"

cv::VideoWriter video("nbody.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));

Vector scaleToWindow(Vector pos)
{

    double scaleX = WINDOW_HEIGHT / NBODY_HEIGHT;
    double scaleY = WINDOW_WIDTH / NBODY_WIDTH;
    return {(pos.x - 0) * scaleX + WINDOW_WIDTH / 2, (pos.y - 0) * scaleY + WINDOW_HEIGHT / 2};
}

void storeFrame(Body *bodies, int n, int id, int error_check, Body *naive_bodies)
{
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    cv::Scalar color; // White color
    int radius;
    
    for (int i = 0; i < n; i++)
    {
        Vector pos = scaleToWindow(bodies[i].position);
        cv::Point center(pos.x, pos.y);
        // std::cout << " " << pos.x <<" "<< pos.y;
        // stars will be red and planets will be white
        if (bodies[i].mass >= HBL)
        {
            color = cv::Scalar(0, 0, 255);
            radius = 5;
        }
        else
        {
            color = cv::Scalar(255, 255, 255);
            radius = 1;
        }
        cv::circle(image, center, radius, color, -1);
    }
    // std::cout << std::endl;
    if (error_check == 1){
        for (int i = 0; i < n; i++)
    {
        Vector pos2 = scaleToWindow(naive_bodies[i].position);
        cv::Point center(pos2.x, pos2.y);
        // std::cout << " " << pos2.x <<" "<< pos2.y;
        // stars will be red and planets will be white
        if (naive_bodies[i].mass >= HBL)
        {
            color = cv::Scalar(0, 0, 255);
            radius = 5;
        }
        else
        {
            color = cv::Scalar(255, 255,0 );
            radius = 1;
        }
        cv::circle(image, center, radius, color, -1);
    }
    // std::cout << std::endl;
    }
    
    video.write(image);
}

bool checkArgs(int nBodies, int sim, int iter)
{

    if (nBodies < 1)
    {
        std::cout << "ERROR: need to have at least 1 body" << std::endl;
        return false;
    }

    if (sim < 0 || sim > 3)
    {
        std::cout << "ERROR: simulation doesn't exist" << std::endl;
        return false;
    }

    if (iter < 1)
    {
        std::cout << "ERROR: need to have at least 1 iteration" << std::endl;
        return false;
    }

    return true;
}

// Calculates Divergence between Direct sum and Barnes Hut
double getAvgDivergence(Body *bodies, Body *naive_bodies, int numBodies) {
    double totalDivergence = 0.0;
    Body *temp1 = new Body[numBodies];
    Body *temp2 = new Body[numBodies];

    

    for (int i = 0; i < numBodies; ++i) {
        temp1[bodies[i].id] = bodies[i];
        temp2[naive_bodies[i].id] = naive_bodies[i];
    }
    bodies = temp1;
    naive_bodies = temp2;
    
    for (int i = 0; i < numBodies; ++i) {
        double dx = bodies[i].position.x - naive_bodies[i].position.x;
        double dy = bodies[i].position.y - naive_bodies[i].position.y;
        double divergence = sqrt(dx * dx + dy * dy);
        
        totalDivergence += divergence;
    }
    
    return numBodies > 0 ? totalDivergence / numBodies : 0.0;
}

int main(int argc, char **argv)
{
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    int nBodies = NUM_BODIES;
    int sim = 0;
    int iters = 300;
    int error_check = 0;
    
    if (argc >= 4)
    {
        nBodies = atoi(argv[1]);
        sim = atoi(argv[2]);
        iters = atoi(argv[3]);
        error_check = atoi(argv[4]);
    }else if (argc==3){
        nBodies = atoi(argv[1]);
        sim = atoi(argv[2]);
        iters = atoi(argv[3]);
    }

    if (!checkArgs(nBodies, sim, iters))
        return -1;

    if (sim == 3)
        nBodies = 5;
        
    std::cout << "Running Barnes-Hut simulation with " << nBodies << " bodies for " 
              << iters << " iterations" << std::endl;

    BarnesHutCuda *bh = new BarnesHutCuda(nBodies,error_check);
    bh->setup(sim);
    Body *bodies;
    

    // First iteration (iteration 0) - warm-up run without timing
    bh->resetTimers();
    bh->update();
    bh->readDeviceBodies();
    bodies = bh->getBodies();
    if (error_check==1){
        bh->runNaive();
        storeFrame(bodies,nBodies,0,error_check, bh->readNaiveDeviceBodies());
    }else{
        storeFrame(bodies, nBodies, 0,error_check,{});
    }
    
        
    // Reset timers after the warm-up iteration
    bh->resetTimers();
    std::cout << "Starting timed iterations (excluding warm-up iteration)..." << std::endl;
    
    // CUDA events for timing entire execution including memory transfers
    cudaEvent_t start, stop;
    float executionTimeMs;

    
    
    // Start measuring from iteration 1 onwards
    for (int i = 1; i < iters; ++i)
    {
        // Start timing for full execution including memory transfers
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        
        // Computational phase
        bh->update();
        
        // Memory transfer phase
        bh->readDeviceBodies();
        
        // Record end time for full execution
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&executionTimeMs, start, stop));
        
        // Add the measured time to our total
        bh->addExecutionTime(executionTimeMs);

        
        
        // Clean up CUDA events
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));
        
        // Visualization
        bodies = bh->getBodies();
        

        if (error_check==1){
            bh->runNaive();
            storeFrame(bodies, nBodies, i,error_check, bh->readNaiveDeviceBodies());
        }else{
            storeFrame(bodies, nBodies, i,error_check, {});
        }
    }

    if (error_check==1){
        Body *naive_bodies = bh->readNaiveDeviceBodies();
        double error = getAvgDivergence(bodies,naive_bodies,nBodies);
        double maxDistance = sqrt(NBODY_WIDTH * NBODY_WIDTH + NBODY_HEIGHT * NBODY_HEIGHT);
        std::cout << "Average error rate is: " << error << std::endl;
        std::cout << "Normalized error rate is: " << error/maxDistance << std::endl;
    }

    // Adjust the iteration count in performance metrics to reflect only the timed iterations
    int timedIterations = iters - 1;
    
    bh->printPerformanceMetrics();
    
    // Export final performance data
    std::ofstream perfOutput("performance_results.csv");
    if (perfOutput.is_open()) {
        perfOutput << "bodies,iterations,total_kernel_time_ms,avg_kernel_time_ms,total_execution_time_ms,avg_execution_time_ms,total_flops,kernel_gflops,effective_gflops" << std::endl;
        double kernelGflops = bh->getTotalFlops() / (bh->getTotalKernelTime() * 1e6);
        double effectiveGflops = bh->getTotalFlops() / (bh->getTotalExecutionTime() * 1e6);
        
        perfOutput << nBodies << ","
                  << timedIterations << ","
                  << bh->getTotalKernelTime() << ","
                  << bh->getAverageKernelTime() << ","
                  << bh->getTotalExecutionTime() << ","
                  << bh->getAverageExecutionTime() << ","
                  << bh->getTotalFlops() << ","
                  << kernelGflops << ","
                  << effectiveGflops << std::endl;
        
        perfOutput.close();
        std::cout << "Performance data written to performance_results.csv" << std::endl;
    } else {
        std::cerr << "ERROR: Could not open performance_results.csv for writing" << std::endl;
    }

    video.release();
    delete bh;
    return 0;
}
