#!/bin/bash
# Benchmark script for running multiple simulations and collecting results

# Navigate to script directory
cd "$(dirname "$0")"

# Build the code
echo "Building BarnesHut..."
mkdir -p build
cd build
cmake ..
make -j4

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

cd ..

# Create output directory if it doesn't exist
mkdir -p benchmark_results

# Create consolidated results file with header
RESULTS_FILE="benchmark_results/consolidated_benchmarks.csv"
echo "simulation_type,bodies,iterations,total_kernel_time_ms,avg_kernel_time_ms,total_execution_time_ms,avg_execution_time_ms,total_flops,kernel_gflops,effective_gflops" > $RESULTS_FILE

# Array of body counts to test
BODY_COUNTS=(1000 5000 10000 50000 100000 500000 1000000 5000000)

# Number of iterations to run
ITERATIONS=300

# Run spiral galaxy simulations (type 0)
echo "Running spiral galaxy benchmarks..."
for bodies in "${BODY_COUNTS[@]}"; do
    echo "  Testing with $bodies bodies..."
    
    # Run the Barnes-Hut simulation
    ./build/BarnesHut $bodies 0 $ITERATIONS
    
    # If performance_results.csv exists, append to consolidated file with simulation type
    if [ -f "performance_results.csv" ]; then
        # Skip header line and add simulation type to each row
        tail -n +2 performance_results.csv | sed "s/^/spiral,/" >> $RESULTS_FILE
        echo "  Results added to $RESULTS_FILE"
    else
        echo "  Error: performance_results.csv not found!"
    fi
done

# Run collision simulations (type 2)
echo "Running collision simulations benchmarks..."
for bodies in "${BODY_COUNTS[@]}"; do
    echo "  Testing with $bodies bodies..."
    
    # Run the Barnes-Hut simulation
    ./build/BarnesHut $bodies 2 $ITERATIONS
    
    # If performance_results.csv exists, append to consolidated file with simulation type
    if [ -f "performance_results.csv" ]; then
        # Skip header line and add simulation type to each row
        tail -n +2 performance_results.csv | sed "s/^/collision,/" >> $RESULTS_FILE
        echo "  Results added to $RESULTS_FILE"
    else
        echo "  Error: performance_results.csv not found!"
    fi
done

echo "Benchmarks complete! Results saved to $RESULTS_FILE"

# Show summary of results
echo ""
echo "Summary of results:"
echo "--------------------------------------------------------"
echo "Simulation Type | Bodies | Avg Kernel (ms) | Avg Execution (ms)"
echo "--------------------------------------------------------"
awk -F, 'NR>1 {printf "%-15s | %6d | %14.2f | %18.2f\n", $1, $2, $5, $7}' $RESULTS_FILE
echo "--------------------------------------------------------"