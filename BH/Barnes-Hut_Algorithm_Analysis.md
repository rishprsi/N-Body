# Barnes-Hut Algorithm Implementation Analysis

## 1. Algorithm Overview

The Barnes-Hut algorithm is a hierarchical approximation method that reduces the computational complexity of N-body simulations from O(N²) to O(N log N). The algorithm works by:

1. Dividing space into a quadtree (in 2D) or octree (in 3D)
2. Approximating distant groups of bodies as single bodies with combined mass
3. Using this approximation when calculating gravitational forces

### Algorithm Visualization

```mermaid
graph TD
    A[Initialize Bodies] --> B[Build Quadtree]
    B --> C[Calculate Forces]
    C --> D[Update Positions]
    D --> E[Repeat for Next Timestep]
    E -->|Next iteration| B
    
    subgraph "Build Quadtree"
    B1[Reset Nodes] --> B2[Compute Bounding Box]
    B2 --> B3[Create Recursive Tree Structure]
    B3 --> B4[Calculate Center of Mass for Each Node]
    end
    
    subgraph "Calculate Forces"
    C1[For Each Body] --> C2{Is Node Far Enough?}
    C2 -->|Yes| C3[Use Node's Center of Mass]
    C2 -->|No| C4[Recursively Check Children]
    C3 --> C5[Accumulate Force]
    C4 --> C5
    end
```

## 2. Main Program Flow

```mermaid
flowchart TD
    Setup[BarnesHutCuda::setup] --> |Initialize| Update[BarnesHutCuda::update]
    Update --> |Step 1| Reset[ResetKernel]
    Reset --> |Step 2| BoundingBox[ComputeBoundingBoxKernel]
    BoundingBox --> |Step 3| QuadTree[ConstructQuadTreeKernel]
    QuadTree --> |Step 4| Force[ComputeForceKernel]
    Force --> |Next Iteration| Update
    
    subgraph "Setup Phase"
    Setup --> |Simulation Type| G1[initSpiralBodies]
    Setup --> |Simulation Type| G2[initRandomBodies]
    Setup --> |Simulation Type| G3[initCollideGalaxy]
    Setup --> |Simulation Type| G4[initSolarSystem]
    end
    
    subgraph "Force Calculation Phase"
    Force --> |For each body| F1[Reset acceleration]
    F1 --> F2[Tree traversal]
    F2 --> F3[Apply Barnes-Hut approximation]
    F3 --> F4[Update velocities and positions]
    end
```

## 3. Data Structures

### Body
```cpp
typedef struct {
    bool isDynamic;      // Whether body can move
    double mass;         // Mass of body
    double radius;       // Radius of body
    Vector position;     // Position (x,y)
    Vector velocity;     // Velocity (vx,vy)
    Vector acceleration; // Acceleration (ax,ay)
} Body;
```

### Node (Quadtree Node)
```cpp
typedef struct {
    Vector topLeft;      // Top-left coordinate of region
    Vector botRight;     // Bottom-right coordinate of region
    Vector centerMass;   // Center of mass of region
    double totalMass;    // Total mass of region
    bool isLeaf;         // Whether node is a leaf
    int start;           // Start index in body array
    int end;             // End index in body array
} Node;
```

### Quadtree Structure Visualization

```mermaid
graph TD
    R[Root Node] --> Q1[Quadrant 1<br>Top-Right]
    R --> Q2[Quadrant 2<br>Top-Left]
    R --> Q3[Quadrant 3<br>Bottom-Left]
    R --> Q4[Quadrant 4<br>Bottom-Right]
    
    Q1 --> Q11[Q1-SubQuad1]
    Q1 --> Q12[Q1-SubQuad2]
    Q1 --> Q13[Q1-SubQuad3]
    Q1 --> Q14[Q1-SubQuad4]
    
    Q2 --> Q21[Q2-SubQuad1]
    Q2 --> Q22[Q2-SubQuad2]
    Q2 --> Q23[Q2-SubQuad3]
    Q2 --> Q24[Q2-SubQuad4]
    
    class R rootNode
    class Q1,Q2,Q3,Q4 internalNode
    class Q11,Q12,Q13,Q14,Q21,Q22,Q23,Q24 leafNode
    
    classDef rootNode fill:#f96,stroke:#333,stroke-width:2px
    classDef internalNode fill:#9cf,stroke:#333,stroke-width:1px
    classDef leafNode fill:#9f9,stroke:#333,stroke-width:1px
```

### Memory Layout

In the Barnes-Hut implementation, the quadtree is represented as a flat array of nodes where:
- The root node is at index 0
- For a node at index i, its four children are at indices 4i+1, 4i+2, 4i+3, and 4i+4
- Bodies are organized in memory to match their spatial arrangement in the quadtree

```
┌─────────────────────────────────────────────────────────────────┐
│                          Node Array                             │
├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┤
│  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │ ... │     │
│Root │Child│Child│Child│Child│Grand│Grand│Grand│Grand│     │     │
│     │  1  │  2  │  3  │  4  │child│child│child│child│     │     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

┌─────────────────────────────────────────────────────────────────┐
│                         Body Array                              │
├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┤
│  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │ ... │     │
│Q1   │Q1   │Q2   │Q2   │Q3   │Q3   │Q4   │Q4   │     │     │     │
│Bodies│    │Bodies│    │Bodies│    │Bodies│    │     │     │     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

## 4. Kernel Analysis

### 4.1 ResetKernel

**Purpose:** Initialize all quadtree nodes to default values before building a new tree.

**Invocation:**
```cpp
int blockSize = BLOCK_SIZE;
dim3 gridSize = ceil((float)nNodes / blockSize);
ResetKernel<<<gridSize, blockSize>>>(d_node, d_mutex, nNodes, nBodies);
```

**Inputs:**
- `Node *node`: Array of nodes representing the quadtree
- `int *mutex`: Array of mutex locks for synchronization
- `int nNodes`: Total number of nodes available in the tree
- `int nBodies`: Total number of bodies in the simulation

**Outputs:**
- `node`: Initialized with default values
- `mutex`: Initialized to 0

**Function:**
```cpp
__global__ void ResetKernel(Node *node, int *mutex, int nNodes, int nBodies) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < nNodes) {
        // Initialize with default values
        node[b].topLeft = {INFINITY, -INFINITY};
        node[b].botRight = {-INFINITY, INFINITY};
        node[b].centerMass = {-1, -1};
        node[b].totalMass = 0.0;
        node[b].isLeaf = true;
        node[b].start = -1;
        node[b].end = -1;
        mutex[b] = 0;
    }

    // Set up root node to contain all bodies
    if (b == 0) {
        node[b].start = 0;
        node[b].end = nBodies - 1;
    }
}
```

**Parallelization:** Each thread initializes one node in the quadtree, with the first thread (b=0) performing additional work to set up the root node.

**Thread/Block Organization:**

```mermaid
graph TD
    subgraph Grid
        B1[Block 0] --> T01[Thread 0<br>Node 0]
        B1 --> T11[Thread 1<br>Node 1]
        B1 --> T21[Thread 2<br>Node 2]
        B1 --> D1[...]
        B1 --> T2551[Thread 255<br>Node 255]
        
        B2[Block 1] --> T02[Thread 0<br>Node 256]
        B2 --> T12[Thread 1<br>Node 257]
        B2 --> T22[Thread 2<br>Node 258]
        B2 --> D2[...]
        B2 --> T2552[Thread 255<br>Node 511]
        
        D3[...] --> D4[...]
    end
    
    classDef rootThread fill:#f96,stroke:#333
    class T01 rootThread
```

### 4.2 ComputeBoundingBoxKernel

**Purpose:** Calculate the minimum bounding box that contains all bodies in the simulation.

**Invocation:**
```cpp
int blockSize = BLOCK_SIZE;
dim3 gridSize = ceil((float)nBodies / blockSize);
ComputeBoundingBoxKernel<<<gridSize, blockSize>>>(d_node, d_b, d_mutex, nBodies);
```

**Inputs:**
- `Node *node`: Array of nodes (only root node is modified)
- `Body *bodies`: Array of all bodies in the simulation
- `int *mutex`: Array of mutex locks for synchronization
- `int nBodies`: Total number of bodies

**Outputs:**
- `node[0]`: Updated with bounding box coordinates

**Function:**
```cpp
__global__ void ComputeBoundingBoxKernel(Node *node, Body *bodies, int *mutex, int nBodies) {
    __shared__ double topLeftX[BLOCK_SIZE];
    __shared__ double topLeftY[BLOCK_SIZE];
    __shared__ double botRightX[BLOCK_SIZE];
    __shared__ double botRightY[BLOCK_SIZE];

    int tx = threadIdx.x;
    int b = blockIdx.x * blockDim.x + tx;

    // Initialize shared memory for reduction
    topLeftX[tx] = INFINITY;
    topLeftY[tx] = -INFINITY;
    botRightX[tx] = -INFINITY;
    botRightY[tx] = INFINITY;

    __syncthreads();

    // Load body positions
    if (b < nBodies) {
        Body body = bodies[b];
        topLeftX[tx] = body.position.x;
        topLeftY[tx] = body.position.y;
        botRightX[tx] = body.position.x;
        botRightY[tx] = body.position.y;
    }

    // Parallel reduction for min/max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tx < s) {
            topLeftX[tx] = fminf(topLeftX[tx], topLeftX[tx + s]);
            topLeftY[tx] = fmaxf(topLeftY[tx], topLeftY[tx + s]);
            botRightX[tx] = fmaxf(botRightX[tx], botRightX[tx + s]);
            botRightY[tx] = fminf(botRightY[tx], botRightY[tx + s]);
        }
    }

    // Update root node bounds with atomic operations
    if (tx == 0) {
        while (atomicCAS(mutex, 0, 1) != 0) { /* spin wait */ }
        
        // Add padding to bounding box
        node[0].topLeft.x = fminf(node[0].topLeft.x, topLeftX[0] - 1.0e10);
        node[0].topLeft.y = fmaxf(node[0].topLeft.y, topLeftY[0] + 1.0e10);
        node[0].botRight.x = fmaxf(node[0].botRight.x, botRightX[0] + 1.0e10);
        node[0].botRight.y = fminf(node[0].botRight.y, botRightY[0] - 1.0e10);
        
        atomicExch(mutex, 0);
    }
}
```

**Parallel Reduction Visualization:**

```mermaid
graph TD
    subgraph "Block with 8 threads (simplified)"
        S0["Step 0: Initial Values<br>topLeftX: [3,1,7,4,2,8,5,6]"]
        S1["Step 1: s=4<br>topLeftX: [min(3,2),min(1,8),min(7,5),min(4,6),2,8,5,6]<br>= [2,1,5,4,2,8,5,6]"]
        S2["Step 2: s=2<br>topLeftX: [min(2,5),min(1,4),5,4,2,8,5,6]<br>= [2,1,5,4,2,8,5,6]"]
        S3["Step 3: s=1<br>topLeftX: [min(2,1),1,5,4,2,8,5,6]<br>= [1,1,5,4,2,8,5,6]"]
        
        S0 --> S1
        S1 --> S2
        S2 --> S3
    end
    
    R["Result: Minimum X = 1"]
    
    S3 --> R
```

### 4.3 ConstructQuadTreeKernel

**Purpose:** Recursively build the quadtree by dividing space and grouping bodies.

**Invocation:**
```cpp
int blockSize = BLOCK_SIZE;
dim3 gridSize = ceil((float)nBodies / blockSize);
ConstructQuadTreeKernel<<<1, blockSize>>>(d_node, d_b, d_b_buffer, 0, nNodes, nBodies, leafLimit);
```

**Thread Collaboration Model for Quadtree Construction:**

```mermaid
graph TD
    A[One Block per Node] --> B[All Threads in Block<br>Collaborate on One Node]
    B --> C[Different Thread Groups<br>Handle Different Tasks]
    
    C --> D[Center of Mass Calculation]
    C --> E[Body Counting]
    C --> F[Body Grouping]
    
    G[Thread 0] --> H[Special Responsibilities]
    H --> I[Set Up Child Nodes]
    H --> J[Launch Child Kernels]
    
    K[Block-Level Synchronization] --> L["__syncthreads()"]
    L --> M[Ensures All Threads Complete<br>Each Phase Before Proceeding]
```

**ConstructQuadTreeKernel Parallelization Strategy:**

The `ConstructQuadTreeKernel` uses a cooperative threading model where:

1. Each node in the quadtree is processed by an entire thread block
2. Different thread groups within the block handle different subtasks
3. Thread 0 has special responsibilities for node management
4. All threads synchronize between phases with `__syncthreads()`

```cpp
__global__ void ConstructQuadTreeKernel(Node *node, Body *bodies, Body *buffer, 
                                       int nodeIndex, int nNodes, int nBodies, int leafLimit) {
    __shared__ int count[8];  // Shared memory visible to all threads in block
    int tx = threadIdx.x;     // Thread ID within the block
    
    // Each block processes a different node
    nodeIndex += blockIdx.x;
    
    if (nodeIndex >= nNodes) return;
    // ... rest of the kernel
}
```

**Overall Quadtree Construction Flow:**

```mermaid
flowchart TD
    A[Start with root node<br>containing all bodies] --> B[Compute center of mass]
    B --> C{Is terminal node?}
    C -->|Yes| D[Copy bodies to buffer<br>and return]
    C -->|No| E[Count bodies per quadrant]
    E --> F[Calculate offsets]
    F --> G[Group bodies by quadrant]
    G --> H[Set up child nodes]
    H --> I[Launch child kernels]
    
    subgraph "CountBodies Function"
    E1[Initialize counters] --> E2[Each thread processes<br>multiple bodies]
    E2 --> E3[Determine quadrant<br>for each body]
    E3 --> E4[Atomically increment<br>counter for quadrant]
    end
    
    subgraph "GroupBodies Function"
    G1[Each thread processes<br>multiple bodies] --> G2[Determine quadrant<br>for each body]
    G2 --> G3[Get destination index<br>using atomic operation]
    G3 --> G4[Place body in<br>correct position]
    end
```

**Quadrant Numbering:**

```
┌───────┬───────┐
│       │       │
│   2   │   1   │
│Top-Left│Top-Right│
│       │       │
├───────┼───────┤
│       │       │
│   3   │   4   │
│Bottom-Left│Bottom-Right│
│       │       │
└───────┴───────┘
```

**Parallel Execution Flow for Tree Construction:**

```mermaid
sequenceDiagram
    participant Host
    participant Block as Thread Block
    participant Thread0 as Thread 0
    participant AllThreads as All Threads
    
    Host->>Block: Launch ConstructQuadTreeKernel for Root Node
    
    AllThreads->>AllThreads: Compute Center of Mass (parallel reduction)
    
    Note over Block: Terminal Node Check
    
    AllThreads->>AllThreads: Count bodies per quadrant
    Note over AllThreads: Each thread classifies multiple bodies<br>Uses atomic operations to update counters
    
    Thread0->>Thread0: Calculate quadrant offsets
    
    AllThreads->>AllThreads: Group bodies by quadrant
    Note over AllThreads: Each thread places bodies<br>at unique locations using atomic counters
    
    Thread0->>Thread0: Set up child nodes<br>Boundaries and body ranges
    
    Thread0->>Host: Launch 4 child kernels (one per quadrant)
    
    Host->>Block: Launch 4 new thread blocks
    
    Note over Host,Block: Recursive process continues<br>until leaf nodes are reached
```

**Helper Functions in Detail:**

#### CountBodies Detailed Parallelization

```mermaid
graph TD
    A["Initialize Counters<br>count[0..3] = 0"] --> B["Distribute Bodies<br>Among Threads"]
    
    subgraph "Thread Processing"
        C[Thread 0] --> C1["Process bodies 0, 256, 512..."]
        D[Thread 1] --> D1["Process bodies 1, 257, 513..."]
        E[Thread 2] --> E1["Process bodies 2, 258, 514..."]
        F["Thread N"] --> F1["Process bodies N, N+256, N+512..."]
    end
    
    B --> C
    B --> D
    B --> E
    B --> F
    
    C1 --> G["For Each Body:<br>Determine Quadrant"]
    D1 --> G
    E1 --> G
    F1 --> G
    
    G --> H{Which Quadrant?}
    H -->|Top-Right| I["atomicAdd(&count[0], 1)"]
    H -->|Top-Left| J["atomicAdd(&count[1], 1)"]
    H -->|Bottom-Left| K["atomicAdd(&count[2], 1)"]
    H -->|Bottom-Right| L["atomicAdd(&count[3], 1)"]
    
    I --> M["__syncthreads()"]
    J --> M
    K --> M
    L --> M
    
    M --> N["Final Counts in count[0..3]"]
```

**CountBodies Code Analysis with Explanations:**

```cpp
__device__ void CountBodies(Body *bodies, Vector topLeft, Vector botRight, 
                          int *count, int start, int end, int nBodies) {
    // Calculate midpoints to divide space into quadrants
    double midX = (topLeft.x + botRight.x) / 2.0;
    double midY = (topLeft.y + botRight.y) / 2.0;
    
    int tx = threadIdx.x;
    
    // Thread 0 initializes counters
    if (tx == 0) {
        count[0] = count[1] = count[2] = count[3] = 0;
        count[4] = count[5] = count[6] = count[7] = 0; // Extra space for offsets
    }
    
    __syncthreads(); // Ensure counters are initialized before all threads proceed
    
    // Strided loop - each thread processes multiple bodies
    // Example: With 256 threads, thread 0 processes bodies 0, 256, 512...
    for (int i = start + tx; i <= end; i += blockDim.x) {
        Body body = bodies[i];
        Vector pos = body.position;
        
        // Determine which quadrant this body belongs to
        int quadrant;
        if (pos.x >= midX && pos.y >= midY)       quadrant = 0; // Top-Right
        else if (pos.x < midX && pos.y >= midY)   quadrant = 1; // Top-Left
        else if (pos.x < midX && pos.y < midY)    quadrant = 2; // Bottom-Left
        else                                      quadrant = 3; // Bottom-Right
        
        // Atomically increment counter for this quadrant
        // This prevents race conditions when multiple threads
        // try to increment the same counter simultaneously
        atomicAdd(&count[quadrant], 1);
    }
    
    __syncthreads(); // Wait for all threads to finish counting
}
```

**Key Parallelization Techniques in CountBodies:**

1. **Strided Processing**: Each thread handles bodies at regular intervals
   - With 256 threads, thread 0 processes bodies 0, 256, 512...
   - Thread 1 processes bodies 1, 257, 513...
   - This maximizes thread utilization

2. **Atomic Operations**: `atomicAdd()` prevents race conditions
   - When multiple threads try to increment the same counter
   - Ensures each body is counted exactly once per quadrant

3. **Barrier Synchronization**: `__syncthreads()` ensures all threads
   - Start with initialized counters
   - Complete counting before proceeding to the next phase

#### ComputeOffset Function

```cpp
__device__ void ComputeOffset(int *count, int start) {
    // Thread 0 calculates offsets
    if (threadIdx.x == 0) {
        // count[0..3] contains the count of bodies in each quadrant
        // Store starting offsets in count[4..7]
        count[4] = start;                       // Top-Right quadrant starts at 'start'
        count[5] = start + count[0];            // Top-Left quadrant
        count[6] = start + count[0] + count[1]; // Bottom-Left quadrant
        count[7] = start + count[0] + count[1] + count[2]; // Bottom-Right quadrant
    }
    
    __syncthreads(); // Ensure offsets are calculated before threads use them
}
```

#### GroupBodies Detailed Parallelization

```mermaid
graph TD
    A["Initialize counters with starting offsets"] --> B["Distribute bodies among threads"]
    
    B --> T0["Thread 0: Process bodies 0, 256, 512..."]
    B --> T1["Thread 1: Process bodies 1, 257, 513..."]
    B --> T2["Thread 2: Process bodies 2, 258, 514..."]
    B --> TN["Thread N: Process bodies N, N+256, N+512..."]
    
    T0 --> P["For each assigned body:"]
    T1 --> P
    T2 --> P
    TN --> P
    
    P --> Q["Determine quadrant for body"]
    Q --> R["Get destination index with atomicAdd"]
    R --> S["Place body in buffer at unique position"]
    
    S --> Z["Final Result: Bodies grouped by quadrant"]
    
    Z --> Z1["Q1 bodies in contiguous memory"]
    Z --> Z2["Q2 bodies in contiguous memory"]
    Z --> Z3["Q3 bodies in contiguous memory"]
    Z --> Z4["Q4 bodies in contiguous memory"]
```

**GroupBodies Code Analysis with Explanations:**

```cpp
__device__ void GroupBodies(Body *bodies, Body *buffer, Vector topLeft, Vector botRight, 
                         int *count, int start, int end, int nBodies) {
    // Calculate midpoints to divide space into quadrants
    double midX = (topLeft.x + botRight.x) / 2.0;
    double midY = (topLeft.y + botRight.y) / 2.0;
    
    int tx = threadIdx.x;
    
    // count[4..7] now holds starting offsets for each quadrant
    // These were calculated in ComputeOffset
    
    __syncthreads(); // Ensure offsets are ready before all threads proceed
    
    // Strided loop - each thread processes multiple bodies
    for (int i = start + tx; i <= end; i += blockDim.x) {
        Body body = bodies[i];
        Vector pos = body.position;
        
        // Determine which quadrant this body belongs to
        int quadrant;
        if (pos.x >= midX && pos.y >= midY)       quadrant = 0; // Top-Right
        else if (pos.x < midX && pos.y >= midY)   quadrant = 1; // Top-Left
        else if (pos.x < midX && pos.y < midY)    quadrant = 2; // Bottom-Left
        else                                      quadrant = 3; // Bottom-Right
        
        // Atomically get destination index and increment counter
        int index = atomicAdd(&count[quadrant + 4], 1);
        
        // Place body in correct position in buffer
        buffer[index] = body;
    }
    
    __syncthreads(); // Wait for all threads to finish grouping
}
```

**Key Parallelization Techniques in GroupBodies:**

1. **Atomic Destination Indexing**: `atomicAdd()` gives each thread a unique destination
   - As threads classify bodies, they atomically increment offset counters
   - Each body gets a unique position in the output array
   - Ensures no two bodies are placed at the same location

2. **Memory Access Pattern Optimization**:
   - Bodies are reordered to match their spatial locality
   - Bodies in the same quadrant are stored contiguously
   - Improves cache coherence for later quadtree traversal
   - Reduces memory latency during force calculation

3. **Load Balancing**:
   - Strided processing ensures even distribution of work
   - Each thread processes approximately the same number of bodies

**Memory Transformation During Tree Construction:**

```
Initial Body Array (random order):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │  9  │
│ Q4  │ Q1  │ Q3  │ Q1  │ Q2  │ Q4  │ Q3  │ Q2  │ Q1  │ Q2  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

After GroupBodies (sorted by quadrant):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │  9  │
│ Q1  │ Q1  │ Q1  │ Q2  │ Q2  │ Q2  │ Q3  │ Q3  │ Q4  │ Q4  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
   │     │     │     │     │     │     │     │     │     │
   └─────┴─────┴─────┘     │     │     │     │     │     │
           │                │     │     │     │     │     │
        Child 1          Child 2  │  Child 3  │  Child 4  │
                              │     │     │     │     │     │
                              └─────┴─────┘     └─────┴─────┘
```

This spatial organization improves data locality, enabling more efficient memory access patterns during the tree traversal phase of the algorithm.

#### ComputeCenterMass

```mermaid
flowchart TD
    A[Start] --> B[Initialize shared memory]
    B --> C[Each thread processes<br>multiple bodies]
    C --> D[Accumulate mass and<br>weighted positions]
    D --> E[Parallel reduction<br>to combine results]
    E --> F[Calculate final<br>center of mass]
    F --> G[Thread 0 updates<br>node properties]
```

### 4.4 ComputeForceKernel

**Purpose:** Calculate gravitational forces between bodies using the quadtree for approximation, and update positions and velocities.

**Barnes-Hut Approximation Criterion Visualization:**

```mermaid
graph TD
    A[For each body] --> B{Is node a leaf?}
    B -->|Yes| C[Calculate direct force]
    B -->|No| D{Is node far enough?<br>width/distance < THETA}
    D -->|Yes| E[Use approximation:<br>Calculate force from<br>node's center of mass]
    D -->|No| F[Recursively visit<br>all four child nodes]
    
    C --> G[Update acceleration]
    E --> G
    F --> G
    
    G --> H[Update velocity<br>and position]
```

**Distance vs. Approximation:**

```
                           ┌───────────────────┐
                           │                   │
                           │                   │
                           │     Node of       │
                           │     width s       │
                           │                   │
                           │                   │
                           └───────────────────┘
                                    ↑
                                    │
                                    │ distance d
                                    │
                                    │
                                    •
                               Body position

If s/d < THETA: Use approximation (treat node as a single body)
If s/d >= THETA: Recursively check children
```

**Integration (Update Rule):**

```mermaid
graph LR
    A["a(t) = F/m"] --> B["v(t+Δt) = v(t) + a(t)·Δt"]
    B --> C["x(t+Δt) = x(t) + v(t+Δt)·Δt"]
```

## 5. Galaxy Collision Simulation

The galaxy collision simulation creates two spiral galaxies on a collision course:

```mermaid
graph TD
    A[Initialize Two Galaxies] --> B[Set First Galaxy Position<br>at -NBODY_WIDTH/6]
    A --> C[Set Second Galaxy Position<br>at +NBODY_WIDTH/6]
    B --> D[Create Spiral Pattern<br>of Bodies]
    C --> E[Create Spiral Pattern<br>of Bodies]
    B --> F[Set Massive Central Body<br>for First Galaxy]
    C --> G[Set Massive Central Body<br>for Second Galaxy]
    D --> H[Calculate Orbital Velocities<br>for First Galaxy]
    E --> I[Calculate Orbital Velocities<br>for Second Galaxy]
```

**Orbital Velocity Calculation:**

For a body at position (x,y) relative to galaxy center:
1. Calculate distance: `r = sqrt((x-centerX)² + (y-centerY)²)`
2. Create unit vector pointing from center to body: `r̂ = (x-centerX,y-centerY)/r`
3. Calculate escape velocity: `v_esc = sqrt(G*M/r)`
4. Set orbital velocity perpendicular to radius: `v = (-r̂.y, r̂.x) * v_esc`

This creates a stable spiral galaxy where bodies orbit the central mass.

## 6. Memory Management and Data Flow

```mermaid
sequenceDiagram
    participant Host
    participant GPU
    
    Host->>Host: Initialize bodies<br>(initCollideGalaxy, etc.)
    Host->>GPU: Copy bodies to device<br>(cudaMemcpy h_b → d_b)
    
    loop Each simulation step
        Host->>GPU: ResetKernel
        Note over GPU: Reset all nodes
        
        Host->>GPU: ComputeBoundingBoxKernel
        Note over GPU: Find simulation boundary
        
        Host->>GPU: ConstructQuadTreeKernel
        Note over GPU: Build quadtree<br>Sort bodies by position
        
        Host->>GPU: ComputeForceKernel
        Note over GPU: Calculate forces<br>Update positions
    end
    
    GPU->>Host: Copy bodies back to host<br>(cudaMemcpy d_b → h_b)
    Host->>Host: Store frame/visualization
```

## 7. Key Algorithm Parameters and Performance Considerations

```mermaid
graph TD
    A[Barnes-Hut Performance] --> B[Algorithmic Parameters]
    A --> C[Hardware Optimization]
    
    B --> B1[THETA<br>Accuracy vs. Speed]
    B --> B2[MAX_NODES<br>Memory Usage]
    B --> B3[DT<br>Simulation Stability]
    B --> B4[E<br>Numerical Stability]
    
    C --> C1[Thread Organization]
    C --> C2[Memory Access Patterns]
    C --> C3[Shared Memory Usage]
    C --> C4[Recursive Implementation]
    
    C1 --> C1a[Reset: 1 Thread per Node]
    C1 --> C1b[BoundingBox: Parallel Reduction]
    C1 --> C1c[QuadTree: Cooperative Thread Block]
    C1 --> C1d[Force: 1 Thread per Body]
    
    C2 --> C2a[Body Reordering by Quadrant]
    C2 --> C2b[Contiguous Memory Access]
    
    C3 --> C3a[Shared Memory for Reduction]
    C3 --> C3b[Shared Memory for Quadrant Counts]
    
    C4 --> C4a[Tree Building: Kernel Launches]
    C4 --> C4b[Force Calculation: Standard Recursion]
```

### THETA Parameter Effect on Approximation

```
Low THETA (0.1): More accurate, more traversal           High THETA (1.0): Less accurate, less traversal
┌──────┬──────┐                                          ┌──────┬──────┐
│      │      │                                          │      │      │
│ Visit│ Visit│                                          │Approx│Approx│
├──────┼──────┤                                          │      │      │
│      │      │                                          │      │      │
│ Visit│ Visit│                                          │      │      │
│      │      │                                          │      │      │
└──────┴──────┘                                          └──────┴──────┘
``` 