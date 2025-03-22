#ifndef FAST_MULTIPOLE_CUDA_H_
#define FAST_MULTIPOLE_CUDA_H_

typedef struct
{
    double x;
    double y;
} Vector;

typedef struct
{
    int id;
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
    bool isLeaf;
    Vector center;
    double size;
    int parent;
    int children[4];
    int bodyStart;
    int bodyCount;
    Complex multipole[P];
    Complex local[P];
    Vector minBound;  // Add minBound for bounding box
    Vector maxBound;  // Add maxBound for bounding box
    double totalMass; // Add totalMass for center of mass calculation
} Cell;

class FastMultipoleCuda
{
private:
    int nBodies;
    int nCells;
    int maxCells;
    int maxDepth;
    int error_flag;
    
    Body *h_bodies;
    Body *h_b_naive;
    Body *d_bodies;
    Body *d_b_naive;
    Body *d_bodies_buffer;
    
    Cell *h_cells;
    Cell *d_cells;
    
    int *h_cellCount;
    int *d_cellCount;
    
    int *h_sortedIndex;
    int *d_sortedIndex;
    
    int *d_mutex;  // Add mutex for synchronization

    // Performance metrics
    float totalKernelTime;
    float totalExecutionTime;  // Total time including memory transfers
    int iterationCount;
    long long totalFlops;
    
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
    FastMultipoleCuda(int n, int error_check);
    ~FastMultipoleCuda();
    
    void setup(int sim);
    void update();
    void readDeviceBodies();
    Body* getBodies();

    // Performance metrics getters
    float getTotalKernelTime() const { return totalKernelTime; }
    float getTotalExecutionTime() const { return totalExecutionTime; }
    void addExecutionTime(float ms) { totalExecutionTime += ms; }
    int getIterationCount() const { return iterationCount; }
    float getAverageKernelTime() const { return iterationCount > 0 ? totalKernelTime / iterationCount : 0; }
    float getAverageExecutionTime() const { return iterationCount > 0 ? totalExecutionTime / iterationCount : 0; }
    long long getTotalFlops() const { return totalFlops; }
    void resetTimers();
    void printPerformanceMetrics();
    void runNaive();
    Body *readNaiveDeviceBodies();
};

#endif 