#include "mpi-naive.h"

void initializeMatrices(int N, int rank, std::vector<int> &A, std::vector<int> &B, std::vector<int> &C)
{
    if (rank == 0)
    {
        A.resize(N * N);
        B.resize(N * N);
        C.resize(N * N);

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 1 + std::rand() % 9;
                B[i * N + j] = 1 + std::rand() % 9;
            }
        }
    }
    else
    {
        B.resize(N * N);
    }
}

void distributeMatrices(int N, int rank, const std::vector<int> &A, std::vector<int> &local_a, std::vector<int> &B, int rows_per_proc)
{
    MPI_Scatter(
        (rank == 0 ? A.data() : nullptr),
        rows_per_proc * N,
        MPI_INT,
        local_a.data(),
        rows_per_proc * N,
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    MPI_Bcast(
        B.data(),
        N * N,
        MPI_INT,
        0,
        MPI_COMM_WORLD);
}

void localMatrixComputation(int N, int rows_per_proc, const std::vector<int> &local_a, const std::vector<int> &B, std::vector<int> &local_c, double &local_time)
{
    double local_start = MPI_Wtime();

    for (int i = 0; i < rows_per_proc; i++)
    {
        for (int k = 0; k < N; k++)
        {
            int a_ik = local_a[i * N + k];
            for (int j = 0; j < N; j++)
            {
                local_c[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }

    double local_end = MPI_Wtime();
    local_time = local_end - local_start;
}

void blockCyclicMatrixMultiply(int N, int rank, int size, const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, double& comp_time, int block_size)
{
    int num_blocks = (N + block_size - 1) / block_size;
    int my_blocks = 0;
    
    for (int b = rank; b < num_blocks * num_blocks; b += size) {
        my_blocks++;
    }
    
    std::vector<int> local_A_blocks;
    std::vector<int> local_B_blocks;
    std::vector<int> local_C_blocks(my_blocks * block_size * block_size, 0);
    std::vector<int> block_rows;
    std::vector<int> block_cols;
    
    if (rank == 0) {
        for (int b = 0; b < num_blocks * num_blocks; b++) {
            int block_row = b / num_blocks;
            int block_col = b % num_blocks;
            int owner = b % size;
            
            int actual_rows = std::min(block_size, N - block_row * block_size);
            int actual_cols = std::min(block_size, N - block_col * block_size);
            
            std::vector<int> block_a(actual_rows * actual_cols);
            std::vector<int> block_b(actual_rows * actual_cols);
            
            for (int i = 0; i < actual_rows; i++) {
                for (int j = 0; j < actual_cols; j++) {
                    int global_i = block_row * block_size + i;
                    int global_j = block_col * block_size + j;
                    block_a[i * actual_cols + j] = A[global_i * N + global_j];
                    block_b[i * actual_cols + j] = B[global_i * N + global_j];
                }
            }
            
            if (owner == 0) {
                local_A_blocks.insert(local_A_blocks.end(), block_a.begin(), block_a.end());
                local_B_blocks.insert(local_B_blocks.end(), block_b.begin(), block_b.end());
                block_rows.push_back(block_row);
                block_cols.push_back(block_col);
            } else {
                MPI_Send(&block_row, 1, MPI_INT, owner, 0, MPI_COMM_WORLD);
                MPI_Send(&block_col, 1, MPI_INT, owner, 1, MPI_COMM_WORLD);
                MPI_Send(&actual_rows, 1, MPI_INT, owner, 2, MPI_COMM_WORLD);
                MPI_Send(block_a.data(), actual_rows * actual_cols, MPI_INT, owner, 3, MPI_COMM_WORLD);
                MPI_Send(block_b.data(), actual_rows * actual_cols, MPI_INT, owner, 4, MPI_COMM_WORLD);
            }
        }
    } else {
        for (int i = 0; i < my_blocks; i++) {
            int block_row, block_col, actual_size;
            MPI_Recv(&block_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&block_col, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&actual_size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            std::vector<int> block_a(actual_size * actual_size);
            std::vector<int> block_b(actual_size * actual_size);
            
            MPI_Recv(block_a.data(), actual_size * actual_size, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(block_b.data(), actual_size * actual_size, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            local_A_blocks.insert(local_A_blocks.end(), block_a.begin(), block_a.end());
            local_B_blocks.insert(local_B_blocks.end(), block_b.begin(), block_b.end());
            block_rows.push_back(block_row);
            block_cols.push_back(block_col);
        }
    }
    
    double start = MPI_Wtime();
    
    int offset = 0;
    for (int b = 0; b < my_blocks; b++) {
        int actual_size = std::min(block_size, N - block_rows[b] * block_size);
        
        for (int i = 0; i < actual_size; i++) {
            for (int k = 0; k < actual_size; k++) {
                int a_ik = local_A_blocks[offset + i * actual_size + k];
                for (int j = 0; j < actual_size; j++) {
                    local_C_blocks[offset + i * actual_size + j] += a_ik * local_B_blocks[offset + k * actual_size + j];
                }
            }
        }
        offset += actual_size * actual_size;
    }
    
    comp_time = MPI_Wtime() - start;
    
    if (rank != 0) {
        for (int b = 0; b < my_blocks; b++) {
            int actual_size = std::min(block_size, N - block_rows[b] * block_size);
            MPI_Send(&block_rows[b], 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
            MPI_Send(&block_cols[b], 1, MPI_INT, 0, 6, MPI_COMM_WORLD);
            MPI_Send(&local_C_blocks[b * block_size * block_size], actual_size * actual_size, MPI_INT, 0, 7, MPI_COMM_WORLD);
        }
    } else {
        offset = 0;
        for (int b = 0; b < my_blocks; b++) {
            int actual_size = std::min(block_size, N - block_rows[b] * block_size);
            for (int i = 0; i < actual_size; i++) {
                for (int j = 0; j < actual_size; j++) {
                    int global_i = block_rows[b] * block_size + i;
                    int global_j = block_cols[b] * block_size + j;
                    C[global_i * N + global_j] = local_C_blocks[offset + i * actual_size + j];
                }
            }
            offset += actual_size * actual_size;
        }
        
        for (int p = 1; p < size; p++) {
            int p_blocks = 0;
            for (int b = p; b < num_blocks * num_blocks; b += size) {
                p_blocks++;
            }
            
            for (int b = 0; b < p_blocks; b++) {
                int block_row, block_col;
                MPI_Recv(&block_row, 1, MPI_INT, p, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&block_col, 1, MPI_INT, p, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                int actual_size = std::min(block_size, N - block_row * block_size);
                std::vector<int> recv_block(actual_size * actual_size);
                MPI_Recv(recv_block.data(), actual_size * actual_size, MPI_INT, p, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                for (int i = 0; i < actual_size; i++) {
                    for (int j = 0; j < actual_size; j++) {
                        int global_i = block_row * block_size + i;
                        int global_j = block_col * block_size + j;
                        C[global_i * N + global_j] = recv_block[i * actual_size + j];
                    }
                }
            }
        }
    }
}

void gatherResults(int N, int rank, int rows_per_proc, const std::vector<int> &local_c, std::vector<int> &C)
{
    MPI_Gather(local_c.data(), rows_per_proc * N, MPI_INT, (rank == 0 ? C.data() : nullptr), rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);
}

double computeMaxLocalTime(double local_time, int /* rank */)
{
    double max_local_time = 0.0;
    MPI_Reduce(&local_time, &max_local_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return max_local_time;
}

void serialVerify(int N, const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C_verify)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C_verify[i * N + j] = sum;
        }
    }
}

bool verifyResults(int N, const std::vector<int>& C, const std::vector<int>& C_verify, int rank)
{
    if (rank != 0)
        return true; // Only rank 0 verifies

    long long diff_sum = 0;
    long long ref_sum = 0;
    int max_errors_to_show = 5;
    int error_count = 0;

    for (int i = 0; i < N * N; ++i)
    {
        long long diff = std::abs(static_cast<long long>(C[i]) - static_cast<long long>(C_verify[i]));
        if (diff > 0 && error_count < max_errors_to_show)
        {
            int row = i / N;
            int col = i % N;
            std::cout << "  Error at (" << row << "," << col << "): "
                      << "got " << C[i] << ", expected " << C_verify[i]
                      << ", diff=" << diff << std::endl;
            error_count++;
        }
        diff_sum += diff * diff;
        ref_sum += static_cast<long long>(C_verify[i]) * static_cast<long long>(C_verify[i]);
    }

    double rel_error = std::sqrt(static_cast<double>(diff_sum) / (static_cast<double>(ref_sum) + 1e-12));
    std::cout << "\nRelative L2 error: " << std::scientific << rel_error << std::endl;

    if (error_count > 0)
    {
        std::cout << "Total errors found: " << error_count;
        if (error_count >= max_errors_to_show)
            std::cout << " (showing first " << max_errors_to_show << ")";
        std::cout << std::endl;
    }

    bool passed = (rel_error < 1e-6);
    if (passed)
    {
        std::cout << "✓ PASSED - Results are correct!" << std::endl;
    }
    else
    {
        std::cout << "✗ FAILED - Results differ significantly!" << std::endl;
    }

    return passed;
}
