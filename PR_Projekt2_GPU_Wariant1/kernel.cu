/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /**
  * Matrix multiplication: C = A * B.
  * Host code.
  *
  * This sample implements matrix multiplication as described in Chapter 3
  * of the programming guide.
  * It has been written for clarity of exposition to illustrate various CUDA
  * programming principles, not with the goal of providing the most
  * performant generic kernel for matrix multiplication.
  *
  * See also:
  * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
  * in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
  * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
  */

  // System includes
//#define WIN32
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float* C, float* A, float* B, int wA, int wB)
{

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix


    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];


    for (int a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep)
    {
    

        // TO JEST ROBIONE CO MA BYÆ ROBIONE
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];


        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;

}

template <int BLOCK_SIZE> __global__ void
matrixMulCUDA3(float* C, float* A, float* B, int wA, int wB, int streamX, int streamY)
{

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix


    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];


    for (int a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep)
    {


        // TO JEST ROBIONE CO MA BYÆ ROBIONE
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];


        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    int row = wA * BLOCK_SIZE * by + wA * ty;
    int col = BLOCK_SIZE * bx + tx;
    /*if (by == 0 && bx == 0 && tx == 0 && ty == 0)
    {
        printf("%d r%d c%d\n", streamX * wB + streamY * wA * 2 + c + wB * ty * 2 + tx, (streamX * wB + streamY * wA * 2 + c + wB * ty * 2 + tx)/wA, (streamX * wB + streamY * wA * 2 + c + wB * ty * 2 + tx) % wA);
        printf("x%d y%d %d %d\n", streamX, streamY, row + streamY*wA*wA/2, col + streamX * wA/2);
    }*/
    C[row + streamY * wA * wA / 2 + col + streamX * wA / 2] = Csub;
    //C[row + streamX + col + streamY] = Csub;

}

template <int BLOCK_SIZE> __global__ void
matrixMulCUDA2(float* C, float* A, float* B, int width)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    float C_local = 0;

    __shared__ float Ads[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bds[BLOCK_SIZE][BLOCK_SIZE];

    // okreœlenie obliczanego przez w¹tek elementu macierzy (jak w poprzednim kodzie – tu brak)
    //tx, ty to identyfikatory w¹tków w ramach bloku, Row i Col - analogicznie
    __shared__ float Ads2[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bds2[BLOCK_SIZE][BLOCK_SIZE];

    //TODO: Pobranie pierwszego bloku danych z pamiêci globalnej do A
    Ads[ty][tx] = A[Row * width + tx]; //kolejny element dla s¹siedniego w¹tku
    Bds[ty][tx] = B[Col * width + ty]; //Chyba jak by by³a transponowowa to bêdzie dzia³aæ mo¿e
    //Bds[ty][tx] = B[ty * width + Col]; // u¿ywana kolumna – jakoœæ pobrañ ?
    __syncthreads();

    // width 1024, 32
    // m zaczynac od 0 czy 1 ?
    for (int m = 0; m < width / BLOCK_SIZE; ++m) {

        //TODO: Przepisanie danych z A do pamiêci B 
        // ----------- START
        
        Ads2[ty][tx] = Ads[ty][tx];
        Bds2[ty][tx] = Bds[ty][tx];

        //Ads[ty][tx] = A[Row * width + m * BLOCK_SIZE + tx]; //kolejny element dla s¹siedniego w¹tku
        //Bds[ty][tx] = B[(m * BLOCK_SIZE + ty) * width + Col]; // u¿ywana kolumna – jakoœæ pobrañ ?
        // ----------- KONIEC -- zwolnienie A
        __syncthreads();
        //TODO: Pobranie kolejnego bloku z globalnej do A
        Ads[ty][tx] = A[Row * width + m * BLOCK_SIZE + tx]; //kolejny element dla s¹siedniego w¹tku
        Bds[ty][tx] = B[Col * width + m * BLOCK_SIZE + ty]; // u¿ywana kolumna – jakoœæ pobrañ ? Chyba jak by by³a transponowowa to bêdzie dzia³aæ mo¿e
        //Bds[ty][tx] = B[(m * BLOCK_SIZE + ty) * width + Col]; // u¿ywana kolumna – jakoœæ pobrañ ?

        //TODO: Obliczenia na B
        // ----------- START
        for (int k = 0; k < BLOCK_SIZE; ++k)
            C_local += Ads2[ty][k] * Bds2[k][tx];
        // ----------- KONIEC -- zwolnienie B
        __syncthreads();
    }

    C[Row * width + Col] = C_local;
}

void constantInit(float* data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int matrixMultiply(int argc, char** argv, int block_size, dim3& dimsA, dim3& dimsB)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A;
    cudaError_t error = cudaMallocHost((void**)&h_A, mem_size_A);
    if (error != cudaSuccess)
    {
        printf("cudaMallocHost h_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B;
    error = cudaMallocHost((void**)&h_B, mem_size_B);
    if (error != cudaSuccess)
    {
        printf("cudaMallocHost h_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // Allocate device memory
    float* d_A, * d_B, * d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float* h_C = (float*)malloc(mem_size_C);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**)&d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**)&d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void**)&d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    /*error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }*/
    
    // 3x4 4x3
    // 4x4
    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
    // dim3 grid(dimsB.x (640) / threads.x (32), dimsA.y (320) / threads.y (32));
    // dim3 grid(20, 10)

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    if (block_size == 16)
    {
        //matrixMulCUDA<16> <<< grid, threads >>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    else
    {
        //matrixMulCUDA<32> <<< grid, threads >>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    printf("done\n");

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        printf("cuda device synchronize returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
     
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Create streams
    const int nStreams = 4;
    cudaStream_t streams[nStreams];

    for (int i = 0; i < nStreams; i++)
    {
        error = cudaStreamCreate(&streams[i]);
        if (error != cudaSuccess)
        {
            printf("cuda Stream Create %d returned error code %d, line(%d)\n", i, error, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    /*for (int i = 0; i < dimsA.y; i++) {
        for (int j = 0; j < dimsA.x; j++) {
            printf("%f ", h_C[i * dimsA.x + j]);
        }
        printf("\n");
    }*/

   /* std::cout << "Result matrix C = " << std::endl;
        for( int g = 0; g < mem_size_C/sizeof(float); g++){
            std::cout << h_C std::endl;
        } 
   */

    // Record the start event
    error = cudaEventRecord(start, 0);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Execute the kernel
    int nIter = 200;
    //printf("TUTAJ\n");


    /*for (int j = 0; j < nIter; j++) // Ten troche dzia³a³ for
    {
        int newMemA = mem_size_A / nStreams;
        int newMemB = mem_size_B / nStreams;
        // TODO: NO TUTAJ TO TRZEBA NAPRAWIÆ
        
        for (int i = 0; i < nStreams; i++)
        {

            int offsetA = i * newMemA / sizeof(float);
            int offsetB = i * newMemB / sizeof(float);
            // Rozwi¹zanie ?? allokowaæ za ka¿dym razem pamiêæ device od odpowiedniego indeksu zanim przekopiujemy ?
            //error = cudaMemcpyAsync(d_A + offsetA, h_A + offsetA, newMemA, cudaMemcpyHostToDevice, streams[i]);
            error = cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, streams[i]);
            if (error != cudaSuccess)
            {
                printf("%d %d cuda memcpy A async returned error code %d %s, line(%d)\n", j, i, error, cudaGetErrorString(error), __LINE__);
                exit(EXIT_FAILURE);
            }
            //error = cudaMemcpyAsync(d_B + offsetB, h_B + offsetB, newMemB, cudaMemcpyHostToDevice , streams[i]);
            error = cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, streams[i]);
            if (error != cudaSuccess)
            {
                printf("%d %d cuda memcpy B async returned error code %d %s, line(%d)\n", j, i, error, cudaGetErrorString(error), __LINE__);
                exit(EXIT_FAILURE);
            }
 

            if (block_size == 16)
            {
                //matrixMulCUDA2<16> <<< grid, threads, 0, streams[i] >>> (d_C+offsetA, d_A+offsetA, d_B+offsetB, dimsA.x);
                matrixMulCUDA2<16> << < grid, threads, 0, streams[i] >> > (d_C, d_A, d_B, dimsA.x);
            }
            else if (block_size == 32)
            {
                //matrixMulCUDA2<32> <<< grid, threads, 0, streams[i] >>> (d_C+offsetA, d_A+offsetA, d_B+offsetB, dimsA.x);
                matrixMulCUDA2<32> << < grid, threads, 0, streams[i] >> > (d_C, d_A, d_B, dimsA.x);
            }
            else {
                matrixMulCUDA2<8> << < grid, threads, 0, streams[i] >> > (d_C, d_A, d_B, dimsA.x);
            }
        }

        if (block_size == 16)
        {
            //matrixMulCUDA<16> <<< grid, threads >>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
            //matrixMulCUDA2<16> <<< grid, threads >>> (d_C, d_A, d_B, dimsA.x);
        }
        else
        {
            //matrixMulCUDA<32> <<< grid, threads >>> (d_C, d_A, d_B, dimsA.x, dimsB.x);
            //matrixMulCUDA2<32> <<< grid, threads >>> (d_C, d_A, d_B, dimsA.x);
        }
        streamY * wA * wA / 2
        streamX * wA / 2
    }*/

    for (int j = 0; j < nIter; j++)
    {
        int newMemA = 2 * mem_size_A / nStreams;
        int newMemB = 2 * mem_size_B / nStreams;
        // TODO: NO TUTAJ TO TRZEBA NAPRAWIÆ

        for (int i = 0; i < nStreams; i++)
        {

            int streamX = i % 2;
            int streamY = i / 2;

            int offsetA = streamY * newMemA / sizeof(float);
            int offsetB = streamX * newMemB / sizeof(float);
            // Rozwi¹zanie ?? allokowaæ za ka¿dym razem pamiêæ device od odpowiedniego indeksu zanim przekopiujemy ?
            //error = cudaMemcpyAsync(d_A + offsetA, h_A + offsetA, newMemA, cudaMemcpyHostToDevice, streams[i]);
            error = cudaMemcpyAsync(d_A + offsetA, h_A + offsetA, newMemA, cudaMemcpyHostToDevice, streams[i]);
            if (error != cudaSuccess)
            {
                printf("%d %d cuda memcpy A async returned error code %d %s, line(%d)\n", j, i, error, cudaGetErrorString(error), __LINE__);
                exit(EXIT_FAILURE);
            }
            //error = cudaMemcpyAsync(d_B + offsetB, h_B + offsetB, newMemB, cudaMemcpyHostToDevice , streams[i]);
            error = cudaMemcpyAsync(d_B + offsetB, h_B + offsetB, newMemB, cudaMemcpyHostToDevice, streams[i]);
            if (error != cudaSuccess)
            {
                printf("%d %d cuda memcpy B async returned error code %d %s, line(%d)\n", j, i, error, cudaGetErrorString(error), __LINE__);
                exit(EXIT_FAILURE);
            }

            grid = dim3(dimsC.x / threads.x / (nStreams / 2), dimsC.y / threads.y / (nStreams / 2));
            //std::cout << "dim3 = " << dimsC.x / threads.x / (nStreams / 2) << " " << dimsC.y / threads.y / (nStreams / 2) << std::endl;
            if (block_size == 16)
            {
                matrixMulCUDA3<16> <<<grid, threads, 0, streams[i] >>> (d_C, d_A + offsetA, d_B + offsetB, dimsA.x, dimsB.x/2, streamX, streamY);
            }
            else if (block_size == 32)
            {
                matrixMulCUDA3<32> <<<grid, threads, 0, streams[i] >> > (d_C, d_A + offsetA, d_B + offsetB, dimsA.x, dimsB.x / 2, streamX, streamY);
                //matrixMulCUDA3<32> <<<grid, threads, 0, streams[i] >> > (d_C, d_A + offsetA, d_B + offsetB, dimsA.x, dimsB.x / 2, offR, offC);
            }
            else {
                matrixMulCUDA3<8> <<<grid, threads, 0, streams[i] >> > (d_C, d_A + offsetA, d_B + offsetB, dimsA.x, dimsB.x / 2, streamX, streamY);
            }
        }
    }

    /*cudaMemcpy(h_A, d_A, mem_size_A, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, mem_size_B, cudaMemcpyDeviceToHost);*/

    /*for (int i = 0; i < dimsA.y; i++) {
        for (int j = 0; j < dimsA.x; j++) {
            printf("%.4f ", h_B[i * dimsA.x + j]);
        }
        printf("\n");
    }*/




    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        printf("cuda device synchronize returned error code %d %s, line(%d)\n", error, cudaGetErrorString(error), __LINE__);
        exit(EXIT_FAILURE);
    }

    // Record the stop event
    error = cudaEventRecord(stop, 0);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul,
        threads.x * threads.y);

    // Copy result from device to host
    error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_C,d_C) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    /*for (int i = 0; i < dimsA.y; i++) {
        for (int j = 0; j < dimsA.x; j++) {
            printf("%.4f ", h_C[i * dimsA.x + j]);
        }
        printf("\n");
    }*/

    printf("Checking computed result for correctness: ");
    bool correct = true;

    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
    {
        if (fabs(h_C[i] - (dimsA.x * valB)) > 1e-3)
        {
            printf("Error! Matrix[%05d][r=%d,c=%d]=%.8f, ref=%.8f error term is > 1e-5\n", i, i/dimsA.x, i%dimsA.x, h_C[i], dimsA.x * valB);
            correct = false;
        }
    }

    printf("%s\n", correct ? "OK" : "FAIL");

    // Clean up memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nNote: For peak performance, please refer to the matrixMulCUBLAS example.\n");

    cudaDeviceReset();

    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char** argv)
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    /*if (checkCmdLineFlag(argc, (const char**)argv, "help") ||
        checkCmdLineFlag(argc, (const char**)argv, "?"))
    {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");

        exit(EXIT_SUCCESS);
    }*/

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    /*if (checkCmdLineFlag(argc, (const char**)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char**)argv, "device");
        cudaSetDevice(devID);
    }*/

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // Use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;

    dim3 dimsA, dimsB;

    if (argc < 5) 
    {
        dimsA = dim3(5 * 2 * block_size, 5 * 2 * block_size, 1);
        dimsB = dim3(5 * 4 * block_size, 5 * 2 * block_size, 1);
    }
    else 
    {
        dimsA.x = atoi(argv[1]);
        dimsA.y = atoi(argv[2]);
        dimsB.x = atoi(argv[3]);
        dimsB.y = atoi(argv[4]);
    }
    //dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    //dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

    /*dimsA.x = atoi(argv[1]);
    dimsA.y = atoi(argv[2]);
    dimsB.x = atoi(argv[3]);
    dimsB.y = atoi(argv[4]);*/

    // width of Matrix A
    /*if (checkCmdLineFlag(argc, (const char**)argv, "wA"))
    {
        dimsA.x = getCmdLineArgumentInt(argc, (const char**)argv, "wA");
    }

    // height of Matrix A
    if (checkCmdLineFlag(argc, (const char**)argv, "hA"))
    {
        dimsA.y = getCmdLineArgumentInt(argc, (const char**)argv, "hA");
    }

    // width of Matrix B
    if (checkCmdLineFlag(argc, (const char**)argv, "wB"))
    {
        dimsB.x = getCmdLineArgumentInt(argc, (const char**)argv, "wB");
    }

    // height of Matrix B
    if (checkCmdLineFlag(argc, (const char**)argv, "hB"))
    {
        dimsB.y = getCmdLineArgumentInt(argc, (const char**)argv, "hB");
    }*/

    if (dimsA.x != dimsB.y)
    {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
            dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    int matrix_result = matrixMultiply(argc, argv, block_size, dimsA, dimsB);

    exit(matrix_result);
}
