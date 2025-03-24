# N-Body Simulation Project

Implementations of various N-body simulation algorithms:
- Barnes-Hut (BH): O(n log n) tree-based approximation
- Direct Summation (Naive): O(n²) exact calculation
- Fast Multipole Method (FMM): O(n) multipole expansion-based approximation

## Requirements
All Platforms:
- OpenCV

On Windows:
- OpenGL (not used)
- Visual Studio 2019

## Building the Project On Windows
Each algorithm is in its own directory. To run a simulation:

- Open the solution file inside the chosen algorithm's directory.
- Go to Project -> Properties.
- Under Configuration Properties -> Debugging, go to command line arguments (see Parameters) and edit as desired.
- Run the simulation with "Local Windows Debugger."


## Building the Project On Linux

Each algorithm is in its own directory. Build them individually:

### Barnes-Hut

```bash
cd BH
mkdir -p build
cd build
cmake ..
make
```

### Naive Implementation

```bash
cd naive
mkdir -p build
cd build
cmake ..
make
```

### Fast Multipole Method

```bash
cd FMM
mkdir -p build
cd build
cmake ..
make
```

## Running the Simulations

### Barnes-Hut (BH)

```bash
cd BH/build
./BH [NUM_BODIES] [SIM_TYPE] [ITERATIONS] [ERROR_CHECK]
```

### Naive Implementation

```bash
cd naive/build
./Naive [NUM_BODIES] [SIM_TYPE] [ITERATIONS]
```

### Fast Multipole Method (FMM)

```bash
cd FMM/build
./FastMultipole [NUM_BODIES] [SIM_TYPE] [ITERATIONS] [ERROR_CHECK]
```

## Parameters

- `NUM_BODIES`: Number of bodies (default: 300)
- `SIM_TYPE`: Simulation type (default: 0)
  - 0: Spiral galaxy
  - 1: Random distribution
  - 2: Colliding galaxies
  - 3: Solar system (sets bodies to 5)
- `ITERATIONS`: Number of simulation steps (default: 300)
- `ERROR_CHECK`: Enable error checking against naive algorithm (0/1, BH & FMM only)

## Examples

```bash
# Barnes-Hut spiral galaxy simulation with 10,000 bodies
cd BH/build
./BH 10000 0 100 0

# Naive direct summation with 5,000 bodies
cd naive/build
./Naive 5000 0 100

# FMM colliding galaxies simulation with 50,000 bodies
cd FMM/build
./FastMultipole 50000 2 200 0
```

## Performance Benchmarks

Each algorithm has its own benchmark script:

```bash
# Barnes-Hut benchmarks
cd BH
./run_benchmarks.sh

# Naive benchmarks
cd naive
./run_naive_benchmarks.sh

# FMM benchmarks
cd FMM
./run_benchmarks.sh
```

The benchmarks will:
1. Run simulations with various body counts
2. Test different simulation configurations
3. Generate CSV performance data
4. Print a summary of results

## Performance Metrics

The application collects:
- Kernel execution times
- Memory transfer overhead
- FLOPS calculations
- Visualization of simulation


## Demos
Demos for each type of simulation are on Google Drive, and the links to each can be found below:

- [Spiral](https://drive.google.com/file/d/1rrUcyskZfSkmFZX80QeRIY8FLLrlIPBh/view?usp=sharing)
- [Random Bodies](https://drive.google.com/file/d/18F7LHWiciP6AH_SibAQ1ORl2z69vqTui/view?usp=sharing)
- [Galaxy Collision](https://drive.google.com/file/d/1gI1YKynvtYgaKmnyQFxjWZGkhzM0pyNM/view?usp=sharing)
- [Solar System](https://drive.google.com/file/d/1xzfObLiw8sC9THEVyLmxeeVNvty54jMp/view?usp=sharing)


