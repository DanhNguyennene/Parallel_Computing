#!/bin/bash

# ==============================================
# MPI Testing Script for WireGuard Cluster
# Network: 10.0.0.0/24 (2 nodes with different usernames)
#   - 10.0.0.1: user danhvuive
#   - 10.0.0.2: user danhbuonba
# Clone repo: git clone https://github.com/DanhNguyennene/CO3067_251_Group_04.git
# ==============================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Local paths (on this machine)
BASE_DIR="$HOME/CO3067_251_Group_04"
HOSTFILE="$BASE_DIR/hostfile"

# Remote path - use ~ which expands to remote user's home on each node
REMOTE_REPO_DIR="~/CO3067_251_Group_04"

# Output directory INSIDE the repo so it gets saved
OUTPUT_DIR="$BASE_DIR/results_MPI_WireGuard_${TIMESTAMP}"

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
        NODES_WITH_USER=$(grep -oE '[a-zA-Z0-9_]+@[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' "$HOSTFILE" | sort -u)
    else
        NODES_WITH_USER="danhvuive@10.0.0.1 danhbuonba@10.0.0.2"
    fi
    
    for node_with_user in $NODES_WITH_USER; do
        echo -n "  Cleaning $node_with_user: "
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$node_with_user" "
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
# Create/check hostfile - OpenMPI format with user@host
# ==============================================
if [ ! -f "$HOSTFILE" ]; then
    echo "Creating hostfile with user mappings..."
    cat > "$HOSTFILE" << 'EOF'
danhvuive@10.0.0.1 slots=4
danhbuonba@10.0.0.2 slots=4
EOF
fi

echo "Hostfile:"
cat "$HOSTFILE"
echo ""

# Extract just IPs for connectivity testing, and full user@host for MPI
NODES_WITH_USER=$(grep -oE '[a-zA-Z0-9_]+@[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' "$HOSTFILE" | sort -u)
NODES=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' "$HOSTFILE" | sort -u)
NODE_COUNT=$(echo "$NODES" | wc -w)
echo "Found $NODE_COUNT nodes in hostfile"

# ==============================================
# Test SSH connectivity to all nodes
# ==============================================
echo ""
echo "Testing SSH connectivity..."
REACHABLE_NODES=""
REACHABLE_NODES_WITH_USER=""
for node_with_user in $NODES_WITH_USER; do
    node=$(echo "$node_with_user" | cut -d'@' -f2)
    echo -n "  $node_with_user: "
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$node_with_user" "echo OK" 2>/dev/null; then
        REACHABLE_NODES="$REACHABLE_NODES $node"
        REACHABLE_NODES_WITH_USER="$REACHABLE_NODES_WITH_USER $node_with_user"
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
    echo "  2. Copy key to each node: ssh-copy-id danhvuive@10.0.0.1"
    echo "                           ssh-copy-id danhbuonba@10.0.0.2"
    echo "  3. Test connection: ssh danhvuive@10.0.0.1"
    echo ""
    echo "Running tests on LOCAL machine only..."
    USE_HOSTFILE=false
    MPI_OPTS="--oversubscribe"
else
    echo "$REACHABLE_COUNT nodes reachable"
    USE_HOSTFILE=true
    MPI_OPTS="--hostfile $HOSTFILE --mca btl tcp,self --mca btl_tcp_if_include 10.0.0.0/24 --mca oob_tcp_if_include 10.0.0.0/24"
fi

# ==============================================
# Setup code on reachable nodes
# ==============================================
if [ "$USE_HOSTFILE" = true ]; then
    echo ""
    echo "Setting up code on reachable nodes..."
    for node_with_user in $REACHABLE_NODES_WITH_USER; do
        echo -n "  $node_with_user: "
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$node_with_user" "
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
        for node_with_user in $REACHABLE_NODES_WITH_USER; do
            cores=$(ssh -o BatchMode=yes -o ConnectTimeout=3 "$node_with_user" "nproc" 2>/dev/null || echo "N/A")
            echo "  $node_with_user: $cores cores"
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
        for node_with_user in $REACHABLE_NODES_WITH_USER; do
            # Use ~ which expands to remote user's home directory
            ssh -o BatchMode=yes -o ConnectTimeout=10 "$node_with_user" "cd ~/CO3067_251_Group_04/$dir && make clean && make" 2>&1 | tail -1 &
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