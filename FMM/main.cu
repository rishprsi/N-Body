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
#include <cstdlib>
#include "constants.h"
#include "fastMultipoleCuda.cuh"

extern cv::VideoWriter video;
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