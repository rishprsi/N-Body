

#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "constants.h"
#include "fastMultipoleCuda.cuh"

cv::VideoWriter video("nbody.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));
extern void storeFrame(Body *bodies, int nBodies, int frameNum);
extern bool checkArgs(int nBodies, int sim, int iter);

int main(int argc, char **argv) {
    int nBodies = NUM_BODIES;
    int sim = 0;
    int iters = 300;
    
    if (argc == 4) {
        nBodies = atoi(argv[1]);
        sim = atoi(argv[2]);
        iters = atoi(argv[3]);
    }
    
    if (!checkArgs(nBodies, sim, iters))
        return -1;
    
    if (sim == 3)
        nBodies = 5; // Solar system has fixed number of bodies
    
    FastMultipoleCuda *fmm = new FastMultipoleCuda(nBodies);
    fmm->setup(sim);
    
    for (int i = 0; i < iters; ++i) {
        fmm->update();
        fmm->readDeviceBodies();
        storeFrame(fmm->getBodies(), nBodies, i);
    }
    
    video.release();
    delete fmm;
    
    return 0;
} 