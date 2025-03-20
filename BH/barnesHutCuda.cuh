#ifndef BARNES_HUT_CUDA_H_
#define BARNES_HUT_CUDA_H_

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

    Body *h_b;
    Node *h_node;

    Body *d_b;
    Body *d_b_buffer;
    Node *d_node;
    int *d_mutex;

    // Performance metrics
    float totalTime;
    float treeInitTime;
    float boundingBoxTime;
    float treeConstructTime;
    float forceCalcTime;
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
    BarnesHutCuda(int n);
    ~BarnesHutCuda();
    void update();
    void setup(int sim);
    void readDeviceBodies();
    Body *getBodies();
    
    // Performance metrics getters
    float getTotalTime() const { return totalTime; }
    float getTreeInitTime() const { return treeInitTime; }
    float getBoundingBoxTime() const { return boundingBoxTime; }
    float getTreeConstructTime() const { return treeConstructTime; }
    float getForceCalcTime() const { return forceCalcTime; }
    long long getTotalFlops() const { return totalFlops; }
    void resetTimers();
    void printPerformanceMetrics();
};

#endif