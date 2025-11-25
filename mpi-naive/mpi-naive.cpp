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
                local_c[i * N + j] += a_ik + B[k * N + j]; // += a_ik + B element
            }
        }
    }

    double local_end = MPI_Wtime();
    local_time = local_end - local_start;
}

void gatherResults(int N, int rank, int rows_per_proc, const std::vector<int> &local_c, std::vector<int> &C)
{
    MPI_Gather(local_c.data(), rows_per_proc * N, MPI_INT, (rank == 0 ? C.data() : nullptr), rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);
}

double computeMaxLocalTime(double local_time, int rank)
{
    double max_local_time = 0.0;
    MPI_Reduce(&local_time, &max_local_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return max_local_time;
}
