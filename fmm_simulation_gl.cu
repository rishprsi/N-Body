// fmm_simulation_gl.cu

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cublas_v2.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

// Error checking macros
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        std::cerr << "CUDA error (" << __FILE__ << ":" << __LINE__ << "): " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(err); \
    } \
}

#define CUBLAS_CHECK(call) { \
    cublasStatus_t err = call; \
    if(err != CUBLAS_STATUS_SUCCESS){ \
        std::cerr << "CUBLAS error (" << __FILE__ << ":" << __LINE__ << ")." << std::endl; \
        exit(err); \
    } \
}

// Simulation parameters
const int NUM_PARTICLES = 1024;   // Total number of particles
const float DOMAIN_MIN = 0.0f;
const float DOMAIN_MAX = 1.0f;
const float DT = 0.01f;           // Time step for simulation

// Particle structure with position, charge and velocity.
// For visualization, we use the first three floats (x, y, z) as the position.
struct Particle {
    float x, y, z;
    float charge;   // still kept from original simulation (could be used for color, etc.)
    float vx, vy, vz;
};

// Global simulation device pointer
Particle* d_particles = nullptr;

// ---------------------------------------------------------------------------
// CUDA Kernel: Update particle positions using a simple Euler integration.
// Particles that leave the domain are wrapped around.
__global__ void updatePositionsKernel(Particle* particles, int numParticles, float dt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < numParticles) {
        Particle p = particles[idx];
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
        // Wrap-around if out of bounds
        float domainSize = DOMAIN_MAX - DOMAIN_MIN;
        if (p.x < DOMAIN_MIN) p.x += domainSize;
        if (p.y < DOMAIN_MIN) p.y += domainSize;
        if (p.z < DOMAIN_MIN) p.z += domainSize;
        if (p.x > DOMAIN_MAX) p.x -= domainSize;
        if (p.y > DOMAIN_MAX) p.y -= domainSize;
        if (p.z > DOMAIN_MAX) p.z -= domainSize;
        particles[idx] = p;
    }
}

// ---------------------------------------------------------------------------
// Global OpenGL objects for CUDA-OpenGL interop
GLuint vbo = 0;                                  // OpenGL Vertex Buffer Object
struct cudaGraphicsResource* cuda_vbo_resource;    // CUDA Graphics Resource (to map OpenGL buffer)

// Create a VBO to store particle data. We allocate enough storage for NUM_PARTICLES
// each of size sizeof(Particle). We will render only the first three floats (x,y,z).
void createVBO(GLuint* vbo, unsigned int size) {
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // Register this buffer object with CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, *vbo, cudaGraphicsMapFlagsWriteDiscard));
}

// Delete the VBO and unregister it from CUDA.
void deleteVBO(GLuint* vbo) {
    if(cuda_vbo_resource) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_vbo_resource));
        cuda_vbo_resource = nullptr;
    }
    if(*vbo) {
        glDeleteBuffers(1, vbo);
        *vbo = 0;
    }
}

// ---------------------------------------------------------------------------
// Simulation initialization: allocate and initialize particle data.
void initSimulation() {
    // Allocate particles on host
    std::vector<Particle> particles_host(NUM_PARTICLES);
    for (int i = 0; i < NUM_PARTICLES; i++){
        // Random positions in [DOMAIN_MIN, DOMAIN_MAX]
        particles_host[i].x = DOMAIN_MIN + static_cast<float>(rand()) / RAND_MAX * (DOMAIN_MAX - DOMAIN_MIN);
        particles_host[i].y = DOMAIN_MIN + static_cast<float>(rand()) / RAND_MAX * (DOMAIN_MAX - DOMAIN_MIN);
        particles_host[i].z = DOMAIN_MIN + static_cast<float>(rand()) / RAND_MAX * (DOMAIN_MAX - DOMAIN_MIN);
        particles_host[i].charge = 1.0f;  // uniform charge
        
        // Random velocities in a small range so the motion is visible
        particles_host[i].vx = 0.001f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        particles_host[i].vy = 0.001f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        particles_host[i].vz = 0.001f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
    }
    
    // Allocate device memory and copy the initial particle data
    CUDA_CHECK(cudaMalloc((void**)&d_particles, NUM_PARTICLES * sizeof(Particle)));
    CUDA_CHECK(cudaMemcpy(d_particles, particles_host.data(), NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// OpenGL display callback: update the VBO from simulation and render particles.
void display() {
    // Map the OpenGL VBO so we can update it with particle data from d_particles
    size_t numBytes;
    Particle* d_vbo_ptr = nullptr;
    CUDA_CHECK(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &numBytes, cuda_vbo_resource));
    
    // Copy particle data from simulation device memory to the mapped VBO
    CUDA_CHECK(cudaMemcpy(d_vbo_ptr, d_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToDevice));
    
    // Unmap the resource so that OpenGL can use it for rendering
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
    
    // Clear screen and set up view
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // For simplicity, use an orthographic projection that fits the domain
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Enable vertex array and bind our VBO.
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // Tell OpenGL the first three floats are the position; stride is sizeof(Particle)
    glVertexPointer(3, GL_FLOAT, sizeof(Particle), (void*)0);
    
    // Draw all particles as points
    glPointSize(3.0f);
    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    
    glutSwapBuffers();
}

// ---------------------------------------------------------------------------
// GLUT idle callback: update simulation and request redisplay.
void idle() {
    // Launch kernel to update particle positions
    int blockSize = 256;
    int gridSize = (NUM_PARTICLES + blockSize - 1) / blockSize;
    updatePositionsKernel<<<gridSize, blockSize>>>(d_particles, NUM_PARTICLES, DT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Request display update
    glutPostRedisplay();
}

// ---------------------------------------------------------------------------
// OpenGL initialization: set up clear color and other states.
void initGL() {
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);
}

// ---------------------------------------------------------------------------
// Cleanup function to free CUDA and OpenGL resources.
void cleanup() {
    if(d_particles) {
        CUDA_CHECK(cudaFree(d_particles));
        d_particles = nullptr;
    }
    deleteVBO(&vbo);
}

// ---------------------------------------------------------------------------
// Main function: initialize GLUT, GLEW, simulation and enter main loop.
int main(int argc, char** argv) {
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("CUDA-OpenGL N-Body Simulation");
    
    // Initialize GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Set up OpenGL state
    initGL();
    
    // Initialize simulation (allocate and set initial particle data)
    initSimulation();
    
    // Create a VBO for particle data and register it with CUDA.
    createVBO(&vbo, NUM_PARTICLES * sizeof(Particle));
    
    // Register GLUT callbacks
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    
    // Set cleanup to be called on exit.
    atexit(cleanup);
    
    // Enter the GLUT main loop
    glutMainLoop();
    
    return 0;
}
