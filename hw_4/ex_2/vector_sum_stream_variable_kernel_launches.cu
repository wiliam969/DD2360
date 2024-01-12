
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#ifdef __linux__
#include <sys/time.h>
#endif

#include <stdlib.h> 
#include <cmath>

#define Double double

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

__global__ void vecAdd(Double *in1, Double *in2, Double *out, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < len) {
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

//----------
int main(int argc, char **argv) {
int input_length;
int N_STREAMS;
long int iStart;
long int iElaps;
Double *hostInput1;
Double *hostInput2;
Double *hostOutput;
Double *resultRef;
Double *deviceInput1;
Double *deviceInput2;
Double *deviceOutput;

//@@ Insert code below to read in input_length from args
if (argc != 3) {
    printf("!!\tERROR: Expected 2 arguments, got %d\n", argc - 1);
    exit(1);
}

printf("->\tStart of execution...\n");

input_length = atoi(argv[1]);                           //ASCII to integer
N_STREAMS = atoi(argv[2]);    

printf("The input length is %d\n", input_length);
printf("Streams: %d\n", N_STREAMS);
  
const int S_seg = input_length / N_STREAMS;
const int stream_bytes = S_seg * sizeof(Double);

// Allocate Host memory for input and output
cudaHostAlloc(&hostInput1, input_length * sizeof(Double),cudaHostAllocDefault);
cudaHostAlloc(&hostInput2, input_length * sizeof(Double),cudaHostAllocDefault);
cudaHostAlloc(&hostOutput, input_length * sizeof(Double),cudaHostAllocDefault);
resultRef  = (Double*)malloc(input_length * sizeof(Double));

//@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for (int i = 0; i < input_length; i++) {
    hostInput1[i] = rand()/(Double)RAND_MAX;
    hostInput2[i] = rand()/(Double)RAND_MAX;
    resultRef[i]  = hostInput1[i] + hostInput2[i];
  }

printf("->\tRandom input vectors created...\n");

//@@ Insert code below to allocate GPU memory here
cudaMalloc(&deviceInput1, input_length * sizeof(Double));
cudaMalloc(&deviceInput2, input_length * sizeof(Double));
cudaMalloc(&deviceOutput, input_length * sizeof(Double));

printf("->\tReference vector created...\n");
//@@ Initialize the 1D grid and block dimensions here
dim3 dimBlock(1024, 1, 1);
dim3 dimGrid((S_seg + 1024 - 1) / 1024, 1, 1);

//@@ Insert code to below to Copy memory to the GPU here
iStart = cpuSecond();

// create stream
cudaStream_t stream[N_STREAMS];

for (int i = 0; i < N_STREAMS; ++i)
    cudaStreamCreate(&stream[i]); 

for (int i = 0; i < N_STREAMS; i++)  
{
    int offset = i*S_seg;
    cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], stream_bytes, cudaMemcpyHostToDevice,stream[i]);
    cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], stream_bytes, cudaMemcpyHostToDevice,stream[i]); 
    vecAdd<<<dimGrid, dimBlock,0,stream[i]>>>(&deviceInput1[offset], &deviceInput2[offset], &deviceOutput[offset], input_length);
    cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], stream_bytes, cudaMemcpyDeviceToHost,stream[i]);
}

cudaDeviceSynchronize();
iElaps = cpuSecond() - iStart;
printf("~~\tTime to execute the H2D+kernel+D2H: %ld\n", iElaps);

//@@ Insert code below to compare the output with the reference
for (int i = 0; i < input_length; i++){
    if(abs(resultRef[i] - hostOutput[i]) >= 1){
        printf("!!\tError in position (%d);\n", i);
        printf("!!\tExpected %f, found %f\n\n", resultRef[i], hostOutput[i]);
    }
}

for (int i = 0; i < N_STREAMS; ++i)
    cudaStreamDestroy( stream[i] );
//@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

//@@ Free the CPU memory here
  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(hostOutput);
  free(resultRef);

  return 0;
}