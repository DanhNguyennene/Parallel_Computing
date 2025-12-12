#!/bin/bash

# ==============================================
# MPI Testing Script for HPCC Cluster
# Network: 10.1.8.0/24
# Clone repo: git clone https://github.com/DanhNguyennene/CO3067_251_Group_04.git
# ==============================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Use HOME directory - same on all nodes
BASE_DIR="$HOME/CO3067_251_Group_04"
HOSTFILE="$BASE_DIR/hostfile"

# Output directory INSIDE the repo so it gets saved
OUTPUT_DIR="$BASE_DIR/results_MPI_HPCC_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "MPI TESTS - DISTRIBUTED CLUSTER"
echo "=============================================="
echo "Date: $(date)"
echo "Output: $OUTPUT_DIR"
echo "Base Dir: $BASE_DIR"
echo "=============================================="

# ==============================================
# Handle --clean option: remove repos and results from all nodes
# ==============================================
if [ "$1" == "--clean" ]; then
    echo ""
    echo "=============================================="
    echo "CLEANING ALL NODES"
    echo "=============================================="
    
    if [ -f "$HOSTFILE" ]; then
        NODES=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' "$HOSTFILE" | sort -u)
    else
        NODES=""
    fi
    
    for node in $NODES; do
        echo -n "  Cleaning $node: "
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" "
            rm -rf ~/CO3067_251_Group_04
            echo 'done'
        " 2>/dev/null || echo "FAILED"
    done
    
    echo -n "  Cleaning local: "
    rm -rf "$BASE_DIR"
    echo "done"
    
    echo ""
    echo "All nodes cleaned!"
    exit 0
fi

# Check if repo exists, if not clone it
if [ ! -d "$BASE_DIR" ]; then
    echo "Cloning repository..."
    cd "$HOME"
    git clone https://github.com/DanhNguyennene/CO3067_251_Group_04.git
fi

cd "$BASE_DIR"
git pull origin main 2>/dev/null || true

# ==============================================
# Create/check hostfile - OpenMPI format
# ==============================================
if [ ! -f "$HOSTFILE" ]; then
    echo "ERROR: No hostfile found at $HOSTFILE"
    echo "Please create a hostfile with format:"
    echo "  hostname_or_ip slots=N"
    echo ""
    echo "Example:"
    echo "  10.0.0.1 slots=4"
    echo "  10.0.0.2 slots=4"
    exit 1
fi

echo "Hostfile:"
cat "$HOSTFILE"
echo ""

NODES=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' "$HOSTFILE" | sort -u)
NODE_COUNT=$(echo "$NODES" | wc -w)
echo "Found $NODE_COUNT nodes in hostfile"

# ==============================================
# Test SSH connectivity to all nodes
# ==============================================
echo ""
echo "Testing SSH connectivity..."
REACHABLE_NODES=""
for node in $NODES; do
    echo -n "  $node: "
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" "echo OK" 2>/dev/null; then
        REACHABLE_NODES="$REACHABLE_NODES $node"
    else
        echo "FAILED - Check SSH keys and connectivity"
    fi
done
echo ""

REACHABLE_COUNT=$(echo $REACHABLE_NODES | wc -w)
if [ "$REACHABLE_COUNT" -eq 0 ]; then
    echo "ERROR: No nodes are reachable via SSH!"
    echo ""
    echo "To fix SSH connectivity:"
    echo "  1. Generate SSH key if needed: ssh-keygen -t rsa"
    echo "  2. Copy key to each node: ssh-copy-id user@10.0.0.X"
    echo "  3. Test connection: ssh 10.0.0.X"
    echo ""
    echo "Running tests on LOCAL machine only..."
    USE_HOSTFILE=false
    MPI_OPTS="--oversubscribe"
else
    echo "$REACHABLE_COUNT nodes reachable"
    USE_HOSTFILE=true
    MPI_OPTS="--hostfile $HOSTFILE --mca btl tcp,self --mca btl_tcp_if_include 10.0.0.0/24"
fi

# ==============================================
# Setup code on reachable nodes
# ==============================================
if [ "$USE_HOSTFILE" = true ]; then
    echo ""
    echo "Setting up code on reachable nodes..."
    for node in $REACHABLE_NODES; do
        echo -n "  $node: "
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" "
            if [ ! -d ~/CO3067_251_Group_04 ]; then
                cd \$HOME && git clone https://github.com/DanhNguyennene/CO3067_251_Group_04.git 2>/dev/null
                echo 'cloned'
            else
                cd ~/CO3067_251_Group_04 && git pull origin main 2>/dev/null
                echo 'updated'
            fi
        " 2>/dev/null || echo "FAILED"
    done
    echo ""
fi

# System Information
{
    echo "=== CLUSTER SYSTEM INFORMATION ==="
    echo "Master Hostname: $(hostname)"
    echo "Date: $(date)"
    echo "User: $(whoami)"
    echo "Nodes in hostfile: $NODE_COUNT"
    echo "Reachable nodes: $REACHABLE_COUNT"
    echo "Using hostfile: $USE_HOSTFILE"
    echo "Local CPU Cores: $(nproc)"
    echo "MPI Version:"
    mpirun --version 2>&1 | head -2
    echo ""
    if [ "$USE_HOSTFILE" = true ]; then
        echo "Node Details:"
        for node in $REACHABLE_NODES; do
            cores=$(ssh -o BatchMode=yes -o ConnectTimeout=3 "$node" "nproc" 2>/dev/null || echo "N/A")
            echo "  $node: $cores cores"
        done
    fi
    echo ""
} | tee "$OUTPUT_DIR/system_info.txt"

cd "$BASE_DIR"

# Function to compile on all reachable nodes
compile_on_all_nodes() {
    local dir=$1
    if [ "$USE_HOSTFILE" = true ]; then
        echo "Compiling $dir on all nodes..."
        for node in $REACHABLE_NODES; do
            ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" "cd $BASE_DIR/$dir && make clean && make" 2>&1 | tail -1 &
        done
        wait
        echo "Compilation complete on all nodes"
    fi
}

# ==============================================
# Test MPI Naive
# ==============================================
echo ""
echo "=============================================="
echo "Testing: MPI Naive Matrix Multiplication"
echo "=============================================="

cd "$BASE_DIR/mpi-naive"
make clean && make
compile_on_all_nodes "mpi-naive"

if [ -f ./mpi_program ]; then
    {
        echo "=== MPI Naive Results ==="
        echo "Platform: Distributed Cluster"
        echo "Date: $(date)"
        echo "Using hostfile: $USE_HOSTFILE"
        echo ""
        
        for size in 100 1000 4000; do
            echo "=========================================="
            echo "Matrix Size: ${size}x${size}"
            echo "=========================================="
            
            # Test different process counts
            for procs in 1 2 4 8 16; do
                if [ $((size % procs)) -eq 0 ]; then
                    echo ""
                    echo "--- Processes: $procs ---"
                    
                    if [ $size -le 1000 ]; then
                        mpirun $MPI_OPTS -np $procs ./mpi_program $size 1
                    else
                        mpirun $MPI_OPTS -np $procs ./mpi_program $size 0
                    fi
                fi
            done
            echo ""
        done
    } 2>&1 | tee "$OUTPUT_DIR/mpi_naive_results.txt"
else
    echo "Build failed for MPI Naive" | tee "$OUTPUT_DIR/mpi_naive_results.txt"
fi

# ==============================================
# Test MPI Strassen
# ==============================================
echo ""
echo "=============================================="
echo "Testing: MPI Strassen Matrix Multiplication"
echo "=============================================="

cd "$BASE_DIR/mpi-strassen"
make clean && make
compile_on_all_nodes "mpi-strassen"

if [ -f ./mpi_program ]; then
    {
        echo "=== MPI Strassen Results ==="
        echo "Platform: Distributed Cluster"
        echo "Date: $(date)"
        echo "Using hostfile: $USE_HOSTFILE"
        echo ""
        
        for size in 100 1000 4000; do
            echo "=========================================="
            echo "Matrix Size: ${size}x${size}"
            echo "=========================================="
            
            echo ""
            echo "--- Processes: 7 (required by Strassen algorithm) ---"
            
            if [ $size -le 1000 ]; then
                mpirun $MPI_OPTS -np 7 ./mpi_program $size 1
            else
                mpirun $MPI_OPTS -np 7 ./mpi_program $size 0
            fi
            echo ""
        done
    } 2>&1 | tee "$OUTPUT_DIR/mpi_strassen_results.txt"
else
    echo "Build failed for MPI Strassen" | tee "$OUTPUT_DIR/mpi_strassen_results.txt"
fi

# ==============================================
# Test Hybrid MPI + OpenMP
# ==============================================
echo ""
echo "=============================================="
echo "Testing: Hybrid MPI+OpenMP Strassen"
echo "=============================================="

cd "$BASE_DIR/hybrid-strassen"
make clean && make
compile_on_all_nodes "hybrid-strassen"

if [ -f ./main ]; then
    {
        echo "=== Hybrid MPI+OpenMP Results ==="
        echo "Platform: Distributed Cluster"
        echo "Date: $(date)"
        echo "Using hostfile: $USE_HOSTFILE"
        echo ""
        
        for size in 100 1000 4000; do
            echo "=========================================="
            echo "Matrix Size: ${size}x${size}"
            echo "=========================================="
            
            for threads in 1 2 4 8; do
                echo ""
                echo "--- Processes: 7, Threads: $threads ---"
                
                export OMP_NUM_THREADS=$threads
                
                if [ $size -le 1000 ]; then
                    mpirun $MPI_OPTS -np 7 -x OMP_NUM_THREADS=$threads ./main $size 1 $threads 128
                else
                    mpirun $MPI_OPTS -np 7 -x OMP_NUM_THREADS=$threads ./main $size 0 $threads 128
                fi
            done
            echo ""
        done
    } 2>&1 | tee "$OUTPUT_DIR/hybrid_strassen_results.txt"
else
    echo "Build failed for Hybrid Strassen" | tee "$OUTPUT_DIR/hybrid_strassen_results.txt"
fi

# ==============================================
# Generate Summary
# ==============================================
echo ""
echo "Generating summary..."

{
    echo "=== MPI BENCHMARK SUMMARY ==="
    echo "Date: $(date)"
    echo "Nodes: $NODE_COUNT (Reachable: $REACHABLE_COUNT)"
    echo "Using hostfile: $USE_HOSTFILE"
    echo ""
    echo "--- MPI Naive ---"
    grep -E "(Matrix Size|Processes:|Total execution time|PASSED|FAILED)" "$OUTPUT_DIR/mpi_naive_results.txt" 2>/dev/null || echo "No results"
    echo ""
    echo "--- MPI Strassen ---"
    grep -E "(Matrix Size|Processes:|Total execution time|PASSED|FAILED|Strassen completed)" "$OUTPUT_DIR/mpi_strassen_results.txt" 2>/dev/null || echo "No results"
    echo ""
    echo "--- Hybrid MPI+OpenMP ---"
    grep -E "(Matrix Size|Processes:|Threads:|Total execution time|PASSED|FAILED|completed)" "$OUTPUT_DIR/hybrid_strassen_results.txt" 2>/dev/null || echo "No results"
} > "$OUTPUT_DIR/SUMMARY.txt"

echo ""
echo "=============================================="
echo "CLUSTER TESTING COMPLETE!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"

echo ""
echo "To push results to GitHub, run:"
echo "  cd $BASE_DIR"
echo "  git add ."
echo "  git commit -m 'Cluster results'"
echo "  git push origin main"