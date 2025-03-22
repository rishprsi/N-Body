#ifndef FAST_MULTIPOLE_KERNEL_H_
#define FAST_MULTIPOLE_KERNEL_H_

#include <stdio.h>
#include <stdlib.h>
#include "fastMultipoleCuda.cuh"

// Tree construction kernels
__global__ void ComputeBoundingBoxKernel(Body *bodies, Cell *cells, int *mutex, int nBodies);
__global__ void BuildTreeKernel(Body *bodies, Cell *cells, int *cellCount, int *sortedIndex, int *mutex, int nBodies, int maxDepth);

// Multipole computation kernels
__global__ void ComputeMultipolesKernel(Body *bodies, Cell *cells, int *sortedIndex, int nCells);
__global__ void TranslateMultipolesKernel(Cell *cells, int nCells);

// Local expansion kernels
__global__ void ComputeLocalExpansionsKernel(Cell *cells, int nCells);
__global__ void EvaluateLocalExpansionsKernel(Body *bodies, Cell *cells, int *sortedIndex, int nBodies);

// Direct evaluation for nearby particles
__global__ void DirectEvaluationKernel(Body *bodies, Cell *cells, int *sortedIndex, int nBodies);

// Final force computation and integration
__global__ void ComputeForcesAndUpdateKernel(Body *bodies, int nBodies);

// Helper device functions
__device__ int getQuadrant(Vector position, Vector center);
__device__ void computeMultipoleExpansion(Body *bodies, int start, int count, Complex *multipole, Vector center);
__device__ void translateMultipole(Complex *source, Complex *target, Vector sourceCenter, Vector targetCenter);
__device__ void translateMultipoleToLocal(Complex *multipole, Complex *local, Vector multipoleCenter, Vector localCenter);
__device__ void evaluateLocalExpansion(Complex *local, Vector center, Vector position, Vector *force);
__device__ void computeDirectForce(Body body1, Body body2, Vector *force);

// Add ResetMutexKernel to the header
__global__ void ResetMutexKernel(int *mutex, int n);

// Reset cells kernel
__global__ void ResetCellsKernel(Cell *cells, int *mutex, int nCells, int nBodies);

// Direct Sum calculations for error checks
__global__ void force_tile_kernel(Body *bodies, int nBodies);

#endif 