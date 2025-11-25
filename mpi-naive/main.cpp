#include "mpi-naive.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " <N>\n";
        }
        MPI_Finalize();
        return 1;
    }

    int N = std::atoi(argv[1]); // Size of the square matrix

    if (N % num_procs != 0)
    {
        if (rank == 0)
            std::cerr << "Error: N must be divisible by number of processes\n";
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / num_procs;
    std::vector<int> A, B, C;
    std::vector<int> local_a(rows_per_proc * N);
    std::vector<int> local_c(rows_per_proc * N, 0);

    initializeMatrices(N, rank, A, B, C);
    double start_time = MPI_Wtime();
    distributeMatrices(N, rank, A, local_a, B, rows_per_proc);

    double local_time;
    localMatrixComputation(N, rows_per_proc, local_a, B, local_c, local_time);

    gatherResults(N, rank, rows_per_proc, local_c, C);

    if (rank == 0)
    {
        double total_time = MPI_Wtime() - start_time; // Re-measure total outside
        std::cout << "\nTotal execution time: " << total_time << " seconds\n";
    }

    double max_local_time = computeMaxLocalTime(local_time, rank);
    if (rank == 0)
    {
        std::cout << "Maximum local computation time among processes: " << max_local_time << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
