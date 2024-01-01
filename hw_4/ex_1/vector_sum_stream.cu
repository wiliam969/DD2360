
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#ifdef __linux__
#include <sys/time.h>
#endif

#include <stdlib.h> 
#include <cmath>

#define Double double

const int TPB = 256;
const int N_STREAMS = 4;

// When comparing the output between CPU and GPU implementation, 
// the precision of the floating-point operations might differ between different versions, 
// which can translate into srounding error differences. 
// Hence, use a margin error range when comparing both versions.

// Please implement a simple vectorAdd program that sums two vectors and stores the results into a third vector. 
// You will understand how to index 1D arrays inside a GPU kernel. 
// Please complete the following main steps in your code. You can create your own code, or, 
// use the following code template (Download Code Template Here hw2_ex1_template.cu 
// Download hw2_ex1_template.cu ) and edit code parts demarcated by the //@@ comment lines. 

__device__ Double addNum(Double x1, Double x2) {
    return x1 + x2;
}

__global__ void vecAdd(Double* in1, Double* in2, Double* out, Double len) {
    //@@ Insert code to implement vector addition here
    for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < len; i += gridDim.x * blockDim.x) {
        const Double x = addNum(in1[i], in2[i]);
        out[i] = x;
    }
}

//@@ Insert code to implement timer start

long int cpuSecond() {
    #ifdef __linux__
    struct timeval timer;
    gettimeofday(&timer, NULL);
    return timer.tv_sec * 1000000 + timer.tv_usec;
    #endif

    return 0;
}

// In this exercise, you will need to use multiple CUDA Streams in your code to improve its parallelism. You may want to consider using asynchronous memory copy if you previously used synchronous copy. Please finish the following implementation based on your code for lab3-ex1: 
// 1. Divide an input vector into multiple segments of a given size (S_seg)
// 2. Create 4 CUDA streams to copy asynchronously from host to GPU memory, perform vector addition on GPU, and copy back the results from GPU memory to host memory
// 3. Add timers to compare the performance using different segment size by varying the value of S_seg.

int main(int argc, char** argv) {

    int input_length = 0;
    int seg_size = 0; 
    long int iStart;
    long int iElaps;
    Double* hostInput1;
    Double* hostInput2;
    Double* hostOutput;
    Double* resultRef;
    Double* deviceInput1;
    Double* deviceInput2;
    Double* deviceOutput;


    //@@ Insert code below to read in inputLength from args
    if (argc != 3) {
        printf("!!\tERROR: Expected 2 arguments, got %d\n", argc - 1);
        exit(1);
    }

    printf("->\tStart of execution...\n");

    input_length = std::atoi(argv[1]);
    seg_size = std::atoi(argv[2]);

    printf("->\tInput length dim (%d)\n", input_length);
    printf("->\Segment Size length dim (%d)\n", seg_size);

    //@@ Insert code below to allocate Host memory for input and output
    hostInput1 = (Double*)malloc(input_length * sizeof(Double));
    hostInput2 = (Double*)malloc(input_length * sizeof(Double));
    hostOutput = (Double*)malloc(input_length * sizeof(Double));
    resultRef = (Double*)malloc(input_length * sizeof(Double));

    //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    for (int i = 0; i < input_length; i++) {
        hostInput1[i] = (rand() % 1000) * sin(i);
        hostInput2[i] = (rand() % 1000) * sin(i);
    }

    printf("->\tRandom input vectors created...\n");

    for (int i = 0; i < input_length; i++) {
        hostOutput[i] = hostInput1[i] + hostInput2[i];
    }

    printf("->\tReference vector created...\n");

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput1, input_length * sizeof(Double));
    cudaMalloc(&deviceInput2, input_length * sizeof(Double));
    cudaMalloc(&deviceOutput, input_length * sizeof(Double));

    iStart = cpuSecond();

    //@@ Insert code below to initialize streams

    int stream_bytes = seg_size * sizeof(Double);
    int remainder = 0; 
    
    if (input_length - seg_size * N_STREAMS != 0)
    {
        remainder = input_length - seg_size * N_STREAMS;
    }

    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) cudaStreamCreate(&streams[i]);

    //@@ Insert code to Copy memory to the GPU here

    int remaining_length = input_length;

    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * seg_size;
        int t_stream_bytes = stream_bytes;

        if (N_STREAMS - 1 == i)
        {
            t_stream_bytes = (seg_size + remainder) * sizeof(Double);
        }
        
        cudaError err = cudaMemcpyAsync(&deviceInput1[offset],
            &hostInput1[offset],
            t_stream_bytes,
            cudaMemcpyHostToDevice,
            streams[i]);

        err = cudaMemcpyAsync(&deviceInput2[offset],
            &hostInput2[offset],
            t_stream_bytes,
            cudaMemcpyHostToDevice,
            streams[i]);

        if (err != cudaSuccess) {
            printf("!!\tCUDA error copying to Device: %s\n", cudaGetErrorString(err));
            cudaFree(deviceInput1);
            cudaFree(deviceInput2);
            cudaFree(deviceOutput);
            exit(2);
        }

        remaining_length -= seg_size;
    }

    iElaps = cpuSecond() - iStart;
    printf("~~\tTime to copy memory to GPU: %ld\n", iElaps);

    printf("->\tCuda memory initialized...\n");



    iStart = cpuSecond();
    //@@ Launch the GPU Kernel here
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * seg_size;
        int t_seg_size = seg_size; 

        if (N_STREAMS - 1 == i)
        {
            t_seg_size = seg_size + remainder;
        }

        //@@ Initialize the 1D grid and block dimensions here
        dim3 dimGrid((t_seg_size + TPB - 1) / TPB, 1, 1);

        dim3 dimBlock(TPB, 1, 1);

        vecAdd <<<dimGrid, dimBlock, 0, streams[i]>>>
        (&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], t_seg_size);
    }

    cudaDeviceSynchronize();

    iElaps = cpuSecond() - iStart;

    printf("~~\tTime to execute the kernel: %ld\n", iElaps);
    printf("->\tGPU computation terminated...\n");

    //@@ Copy the GPU memory back to the CPU here

    iStart = cpuSecond();
    stream_bytes = seg_size * sizeof(Double);

    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * seg_size;

        int t_stream_bytes = stream_bytes;

        if (N_STREAMS - 1 == i)
        {
            t_stream_bytes = (seg_size + remainder) * sizeof(Double);
        }

        cudaError err = cudaMemcpyAsync(&resultRef[offset], &deviceOutput[offset], t_stream_bytes, cudaMemcpyDeviceToHost, streams[i]);
        if (err != cudaSuccess) {
            printf("!!\tCUDA error copying to Host: %s\n", cudaGetErrorString(err));
            cudaFree(deviceInput1);
            cudaFree(deviceInput2);
            cudaFree(deviceOutput);
            exit(2);
        }
    }
    iElaps = cpuSecond() - iStart;
    printf("~~\tTime to copy memory from GPU: %ld\n", iElaps);

    //@@ Insert code below to compare the output with the reference
    for (int i = 0; i < input_length; i++) {
        if (abs(resultRef[i] - hostOutput[i]) >= 1) {
            printf("!!\tError in position (%d);\n", i);
            printf("!!\tExpected %f, found %f\n\n", resultRef[i], hostOutput[i]);
        }
    }

    printf("->\tThe result provided by the GPU is correct!\n");

    // //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    // //@@ Free the CPU memory here
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

    return 0;
}
