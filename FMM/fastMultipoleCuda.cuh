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

#ifndef FAST_MULTIPOLE_CUDA_H_
#define FAST_MULTIPOLE_CUDA_H_

typedef struct
{
    double x;
    double y;
} Vector;

typedef struct
{
    bool isDynamic;
    double mass;
    double radius;
    Vector position;
    Vector velocity;
    Vector acceleration;
} Body;

// Complex number for multipole expansions
typedef struct
{
    double real;
    double imag;
} Complex;

// Cell structure for FMM
typedef struct
{
    Vector center;
    double size;
    int parent;
    int children[4];
    int bodyStart;
    int bodyCount;
    Complex multipole[P];
    Complex local[P];
    bool isLeaf;
} Cell;

class FastMultipoleCuda
{
private:
    int nBodies;
    int nCells;
    int maxCells;
    int maxDepth;
    
    Body *h_bodies;
    Body *d_bodies;
    Body *d_bodies_buffer;
    
    Cell *h_cells;
    Cell *d_cells;
    
    int *h_cellCount;
    int *d_cellCount;
    
    int *h_sortedIndex;
    int *d_sortedIndex;
    
    // Simulation initialization methods
    void initRandomBodies();
    void initSpiralBodies();
    void initCollideGalaxy();
    void initSolarSystem();
    void setBody(int i, bool isDynamic, double mass, double radius, Vector position, Vector velocity, Vector acceleration);
    
    // FMM algorithm steps
    void buildTree();
    void computeMultipoles();
    void translateMultipoles();
    void computeLocalExpansions();
    void evaluateLocalExpansions();
    void directEvaluation();
    
public:
    FastMultipoleCuda(int n);
    ~FastMultipoleCuda();
    
    void setup(int sim);
    void update();
    void readDeviceBodies();
    Body* getBodies();
};

#endif 