#!/bin/bash

# Directory for results
mkdir -p results

# Array of body sizes to test
body_sizes=(1000 5000 10000)

# Run naive simulations
echo "Running naive simulations..."
for bodies in "${body_sizes[@]}"; do
    echo "Running naive simulation with $bodies bodies..."
    # Simulation type 0 = spiral, iterations = 2
    ncu ./naive/naive/build/NaiveNBody $bodies 0 2 > "results/naive_${bodies}_bodies.txt"
    echo "Done. Results saved to results/naive_${bodies}_bodies.txt"
    ncu --metrics l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,l1tex__t_sector_pipe_lsu_mem_local_op_ld_hit_rate.pct,sm__sass_data_bytes_mem_shared.avg,tpc__average_registers_per_thread.ratio ./naive/naive/build/NaiveNBody $bodies 0 2 > "results/naive_${bodies}_bodies_metrics.txt"
    echo "Done. Results saved to results/naive_${bodies}_bodies_metrics.txt"
done

# Run Barnes-Hut simulations
echo "Running Barnes-Hut simulations..."
for bodies in "${body_sizes[@]}"; do
    echo "Running BH simulation with $bodies bodies..."
    # Simulation type 0 = spiral, iterations = 2
    ncu ./BH/build/BarnesHut $bodies 0 2 > "results/bh_${bodies}_bodies.txt"
    echo "Done. Results saved to results/bh_${bodies}_bodies.txt"
    ncu --metrics l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,l1tex__t_sector_pipe_lsu_mem_local_op_ld_hit_rate.pct,sm__sass_data_bytes_mem_shared.avg,tpc__average_registers_per_thread.ratio ./BH/build/BarnesHut $bodies 0 2 > "results/bh_${bodies}_bodies_metrics.txt"
    echo "Done. Results saved to results/bh_${bodies}_bodies_metrics.txt"
done

echo "All simulations completed."
echo "Results are available in the results directory." 