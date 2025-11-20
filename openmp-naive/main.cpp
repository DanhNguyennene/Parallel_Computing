#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <omp.h>
#include <cstring>
#include <algorithm>

#define LOWER_B 0.0
#define UPPER_B 1.0

#define THRESHOLD 200
#define OMP_NUM_THREAD 16

class Timer{
    std::chrono::high_resolution_clock::time_point start_;
    public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    float elapse(){
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float>(end - start_).count();
    }
};

std::vector<float> createRandomMatrix(int size, int seed){
    std::vector<float> matrix(size*size);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(LOWER_B, UPPER_B);
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = dist(rng);
    }
    return matrix;
}

void serialVerify(int n, const float* A, const float* B, float* C){
    std::fill(C, C + n*n, 0.0f);
    for(int i = 0; i < n; ++i){
        for(int k = 0; k < n; ++k){
            float a_ik = A[i * n + k];
            for (int j = 0; j < n; ++j){
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

void naiveAddMultiply(
    int n, 
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
){
    for(int i = 0; i < n; ++i){
        for(int k = 0; k < n; ++k){
            float a_ik = A[i * lda + k];
            
            #pragma omp simd
            for (int j = 0; j < n; ++j){
                C[i * ldc + j] += a_ik * B[k * ldb + j];
            }
        }
    }
}

void recursiveMatMul(
    int n, 
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
){
    if (n <= THRESHOLD || n % 2 != 0) {
        naiveAddMultiply(n, A, lda, B, ldb, C, ldc);
        return;
    }

    int m = n / 2;

    const float* A11 = A;
    const float* A12 = A + m;
    const float* A21 = A + m * lda;
    const float* A22 = A + m * lda + m;

    const float* B11 = B;
    const float* B12 = B + m;
    const float* B21 = B + m * ldb;
    const float* B22 = B + m * ldb + m;

    float* C11 = C;
    float* C12 = C + m;
    float* C21 = C + m * ldc;
    float* C22 = C + m * ldc + m;
    
    #pragma omp taskgroup
    {

        // C11 += A11 * B11
        // C12 += A11 * B12
        // C21 += A21 * B11
        // C22 += A21 * B12
        
        #pragma omp task
        recursiveMatMul(m, A11, lda, B11, ldb, C11, ldc);
        
        #pragma omp task
        recursiveMatMul(m, A11, lda, B12, ldb, C12, ldc);

        #pragma omp task
        recursiveMatMul(m, A21, lda, B11, ldb, C21, ldc);

        #pragma omp task
        recursiveMatMul(m, A21, lda, B12, ldb, C22, ldc);
        
        #pragma omp taskwait 

 
        // C11 += A12 * B21
        // C12 += A12 * B22
        // C21 += A22 * B21
        // C22 += A22 * B22
        #pragma omp task
        recursiveMatMul(m, A12, lda, B21, ldb, C11, ldc);

        #pragma omp task
        recursiveMatMul(m, A12, lda, B22, ldb, C12, ldc);

        #pragma omp task
        recursiveMatMul(m, A22, lda, B21, ldb, C21, ldc);

        #pragma omp task
        recursiveMatMul(m, A22, lda, B22, ldb, C22, ldc);
    }
}

void parallelDCMatMul(
    int n, 
    const std::vector<float>& A, 
    const std::vector<float>& B, 
    std::vector<float>& C
){
   
    std::cout << "Original N=" << n << std::endl;

    #pragma omp parallel
    {
        #pragma omp single
        {
            recursiveMatMul(n, A.data(), n, B.data(), n, C.data(), n);
        }
    }
}


int main(int argc, char ** argv){

    if (argc < 3){
        std::cerr << "Usage: " << " <matrix_size>" << "  <check err>  ";
        return 1;
    }
    int N = std::atoi(argv[1]);

    std::cout << "Initializing..." << std::endl;
    omp_set_num_threads(OMP_NUM_THREAD); 
    
    auto A = createRandomMatrix(N, 123);
    auto B = createRandomMatrix(N, 456);
    std::vector<float> C(N * N, 0.0);

    std::cout << "Starting Parallel Divide & Conquer..." << std::endl;
    Timer t;
    t.start();
    
    parallelDCMatMul(N, A, B, C);
    
    float time = t.elapse();
    std::cout << "Time: " << time << "s" << std::endl;

    int check = std::atoi(argv[2]);
    if (check==0){
        return 0;
    }

    std::cout << "Verifying results..." << std::endl;
    std::vector<float> CC(N * N, 0.0f);
    serialVerify(N, A.data(), B.data(), CC.data());
    
    float diff_sum = 0.0, ref_sum = 0.0;
    for (int i = 0; i < N * N; ++i) {
        float diff = C[i] - CC[i];
        diff_sum += diff * diff;
        ref_sum += CC[i] * CC[i];
    }
    float rel_error = std::sqrt(diff_sum / (ref_sum + 1e-12));
    std::cout << "Relative L2 error: " << rel_error << "\n";

    return 0;
}