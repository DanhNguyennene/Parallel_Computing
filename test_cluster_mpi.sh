#!/bin/bash

# ==============================================
# MPI Testing Script for WireGuard Cluster
# Network: 10.0.0.0/24 (2 nodes: 10.0.0.1, 10.0.0.2)
# Clone repo: git clone https://github.com/DanhNguyennene/CO3067_251_Group_04.git
# Supports both OpenMPI and MPICH
# ==============================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Use HOME directory - same on all nodes
BASE_DIR="$HOME/mpi_cluster_para_assignment"
HOSTFILE="$BASE_DIR/hostfile"

# Output directory INSIDE the repo so it gets saved
OUTPUT_DIR="$BASE_DIR/results_MPI_WireGuard_${TIMESTAMP}"

# Detect MPI implementation
if mpirun --version 2>&1 | grep -q "Open MPI"; then
    MPI_TYPE="openmpi"
    MPI_OPTS="-hostfile $HOSTFILE --mca btl tcp,self --mca btl_tcp_if_include 10.0.0.0/24 --mca oob_tcp_if_include 10.0.0.0/24"
else
    MPI_TYPE="mpich"
    # MPICH uses -f for hostfile and doesn't need mca options
    MPI_OPTS="-f $HOSTFILE"
fi

echo "Detected MPI: $MPI_TYPE"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "MPI TESTS - WireGuard CLUSTER (10.0.0.0/24)"
echo "=============================================="
echo "Date: $(date)"
echo "Output: $OUTPUT_DIR"
echo "Base Dir: $BASE_DIR"
echo "MPI Type: $MPI_TYPE"
echo "=============================================="

# ==============================================
# Handle --clean option: remove repos and results from all nodes
# ==============================================
if [ "$1" == "--clean" ]; then
    echo ""
    echo "=============================================="
    echo "CLEANING ALL NODES"
    echo "=============================================="
    
    # Read hostfile if exists
    if [ -f "$HOSTFILE" ]; then
        NODES=$(grep -oE '10\.0\.0\.[0-9]+' "$HOSTFILE" | sort -u)
    else
        # Default nodes
        NODES="10.0.0.1 10.0.0.2"
    fi
    
    for node in $NODES; do
        echo -n "  Cleaning $node: "
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" "
            rm -rf ~/mpi_cluster_para_assignment
            echo 'done'
        " 2>/dev/null || echo "FAILED"
    done
    
    # Clean local too
    echo -n "  Cleaning local: "
    rm -rf "$BASE_DIR"
    echo "done"
    
    echo ""
    echo "All nodes cleaned!"
    exit 0
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "Cloning repository..."
    cd "$HOME"
    git clone https://github.com/DanhNguyennene/CO3067_251_Group_04.git
fi

cd "$BASE_DIR"
git pull origin main 2>/dev/null || true

# Always recreate hostfile to ensure correct format for detected MPI type
echo "Creating hostfile for $MPI_TYPE..."
if [ "$MPI_TYPE" == "openmpi" ]; then
    # OpenMPI format: hostname slots=N
    cat > "$HOSTFILE" << 'EOF'
10.0.0.1 slots=4
10.0.0.2 slots=4
EOF
else
    # MPICH format: hostname:num_procs
    cat > "$HOSTFILE" << 'EOF'
10.0.0.1:4
10.0.0.2:4
EOF
fi

echo "Hostfile:"
cat "$HOSTFILE"
echo ""

NODES=$(grep -oE '10\.0\.0\.[0-9]+' "$HOSTFILE" | sort -u)
NODE_COUNT=$(echo "$NODES" | wc -l)
echo "Found $NODE_COUNT nodes in hostfile"

echo ""
echo "Setting up code on all nodes..."
for node in $NODES; do
    echo -n "  $node: "
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" "
        rm -rf ~/CO3067_251_Group_04
        cd \$HOME && git clone https://github.com/DanhNguyennene/CO3067_251_Group_04.git 2>/dev/null
        echo 'fresh clone'
    " 2>/dev/null || echo "FAILED (skipping)"
done
echo ""

# System Information
{
    echo "=== WireGuard CLUSTER SYSTEM INFORMATION ==="
    echo "Master Hostname: $(hostname)"
    echo "Date: $(date)"
    echo "User: $(whoami)"
    echo "Nodes: $NODE_COUNT"
    echo "Local CPU Cores: $(nproc)"
    echo "MPI Version:"
    mpirun --version | head -2
    echo ""
    echo "Available Nodes:"
    for node in $NODES; do
        cores=$(ssh -o BatchMode=yes -o ConnectTimeout=3 "$node" "nproc" 2>/dev/null || echo "N/A")
        echo "  $node: $cores cores"
    done
    echo ""
} | tee "$OUTPUT_DIR/system_info.txt"

cd "$BASE_DIR"

# Function to compile on all nodes
compile_on_all_nodes() {
    local dir=$1
    echo "Compiling $dir on all nodes..."
    for node in $NODES; do
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" "cd $BASE_DIR/$dir && make clean && make" 2>&1 | tail -1 &
    done
    wait
    echo "Compilation complete on all nodes"
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
        echo "Platform: WireGuard Cluster (10.0.0.0/24)"
        echo "Date: $(date)"
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
        echo "Platform: WireGuard Cluster (10.0.0.0/24)"
        echo "Date: $(date)"
        echo ""
        
        # MPI Strassen requires exactly 7 processes (one for each of the 7 Strassen sub-problems)
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
        echo "Platform: WireGuard Cluster (10.0.0.0/24)"
        echo "Date: $(date)"
        echo ""
        
        # Hybrid Strassen requires exactly 7 MPI processes, but can vary OpenMP threads
        for size in 100 1000 4000; do
            echo "=========================================="
            echo "Matrix Size: ${size}x${size}"
            echo "=========================================="
            
            # Test with 7 MPI processes and different OpenMP thread counts
            for threads in 1 2 4 8; do
                echo ""
                echo "--- Processes: 7, Threads: $threads ---"
                
                # Set OMP_NUM_THREADS - syntax differs between OpenMPI and MPICH
                if [ "$MPI_TYPE" == "openmpi" ]; then
                    ENV_OPT="-x OMP_NUM_THREADS=$threads"
                else
                    ENV_OPT="-env OMP_NUM_THREADS $threads"
                fi
                
                if [ $size -le 1000 ]; then
                    mpirun $MPI_OPTS -np 7 $ENV_OPT ./main $size 1 $threads 128
                else
                    mpirun $MPI_OPTS -np 7 $ENV_OPT ./main $size 0 $threads 128
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
    echo "=== MPI BENCHMARK SUMMARY (WireGuard Cluster) ==="
    echo "Date: $(date)"
    echo "Nodes: $NODE_COUNT"
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
echo "WireGuard CLUSTER TESTING COMPLETE!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"

echo ""
echo "To push results to GitHub, run:"
echo "  cd $BASE_DIR"
echo "  git add ."
echo "  git commit -m 'WireGuard cluster results'"
echo "  git push origin main"
ls -la "$OUTPUT_DIR/"
