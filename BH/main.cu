#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "barnesHutCuda.cuh"
#include "constants.h"

cv::VideoWriter video("nbody.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));

Vector scaleToWindow(Vector pos)
{

    double scaleX = WINDOW_HEIGHT / NBODY_HEIGHT;
    double scaleY = WINDOW_WIDTH / NBODY_WIDTH;
    return {(pos.x - 0) * scaleX + WINDOW_WIDTH / 2, (pos.y - 0) * scaleY + WINDOW_HEIGHT / 2};
}

void storeFrame(Body *bodies, int n, int id)
{
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    cv::Scalar color; // White color
    int radius;
    for (int i = 0; i < n; i++)
    {
        Vector pos = scaleToWindow(bodies[i].position);
        cv::Point center(pos.x, pos.y);

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
    video.write(image);
}

void exportPositionsToCSV(Body *bodies, int n, const std::string &filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return;
    }
    
    // Write header
    outputFile << "id,x,y,mass,velocity_x,velocity_y" << std::endl;
    
    // Write body data
    for (int i = 0; i < n; i++) {
        outputFile << i << ","
                  << bodies[i].position.x << ","
                  << bodies[i].position.y << ","
                  << bodies[i].mass << ","
                  << bodies[i].velocity.x << ","
                  << bodies[i].velocity.y << std::endl;
    }
    
    outputFile.close();
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

int main(int argc, char **argv)
{
    int nBodies = NUM_BODIES;
    int sim = 0;
    int iters = 300;
    int exportFreq = 10;  // Export every 10 frames
    
    if (argc >= 4)
    {
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

    BarnesHutCuda *bh = new BarnesHutCuda(nBodies);
    bh->setup(sim);
    
    system("mkdir -p output_data");
    
    // Reset timers before the main simulation loop
    bh->resetTimers();

    for (int i = 0; i < iters; ++i)
    {
        bh->update();
        bh->readDeviceBodies();
        storeFrame(bh->getBodies(), nBodies, i);
        
        if (i % exportFreq == 0) {
            std::string filename = "output_data/positions_" + std::to_string(i) + ".csv";
            exportPositionsToCSV(bh->getBodies(), nBodies, filename);
        }
    }

    bh->printPerformanceMetrics();
    
    // Export final performance data
    std::ofstream perfOutput("performance_results.csv");
    if (perfOutput.is_open()) {
        perfOutput << "bodies,iterations,total_time_ms,tree_init_ms,bounding_box_ms,tree_construct_ms,force_calc_ms,total_flops,gflops" << std::endl;
        perfOutput << nBodies << ","
                  << iters << ","
                  << bh->getTotalTime() << ","
                  << bh->getTreeInitTime() << ","
                  << bh->getBoundingBoxTime() << ","
                  << bh->getTreeConstructTime() << ","
                  << bh->getForceCalcTime() << ","
                  << bh->getTotalFlops() << ","
                  << (bh->getTotalFlops() / (bh->getTotalTime() * 1e6)) << std::endl;
        perfOutput.close();
        std::cout << "Performance data written to performance_results.csv" << std::endl;
    } else {
        std::cerr << "ERROR: Could not open performance_results.csv for writing" << std::endl;
    }

    video.release();
    delete bh;
    return 0;
}
