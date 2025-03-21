#ifndef BARNES_HUT_CUDA_H_
#define BARNES_HUT_CUDA_H_

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

typedef struct
{
    Vector topLeft;
    Vector botRight;
    Vector centerMass;
    double totalMass;
    bool isLeaf;
    int start;
    int end;

} Node;

class BarnesHutCuda
{
    int nBodies;
    int nNodes;
    int leafLimit;
    int error_flag;

    Body *h_b;
    Body *h_b_naive;
    Node *h_node;

    Body *d_b;
    Body *d_b_naive;
    Body *d_b_buffer;
    Node *d_node;
    int *d_mutex;

    // Performance metrics
    float totalKernelTime;
    float totalExecutionTime;  // Total time including memory transfers
    int iterationCount;
    long long totalFlops;

    void initRandomBodies();
    void initSpiralBodies();
    void initCollideGalaxy();
    void initSolarSystem();
    void setBody(int i, bool isDynamic, double mass, double radius, Vector position, Vector velocity, Vector acceleration);
    void resetCUDA();
    void computeBoundingBoxCUDA();
    void constructQuadTreeCUDA();
    void computeForceCUDA();
   

public:
    BarnesHutCuda(int n,int error_check);
    ~BarnesHutCuda();
    void update();
    void setup(int sim);
    void readDeviceBodies();
    Body *getBodies();
    
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