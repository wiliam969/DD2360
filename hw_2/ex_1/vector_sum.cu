#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h> 
#include <cmath>

#define Double double

const int TPB = 256;

// When comparing the output between CPU and GPU implementation, 
// the precision of the floating-point operations might differ between different versions, 
// which can translate into rounding error differences. 
// Hence, use a margin error range when comparing both versions.

// Please implement a simple vectorAdd program that sums two vectors and stores the results into a third vector. 
// You will understand how to index 1D arrays inside a GPU kernel. 
// Please complete the following main steps in your code. You can create your own code, or, 
// use the following code template (Download Code Template Here hw2_ex1_template.cu 
// Download hw2_ex1_template.cu ) and edit code parts demarcated by the //@@ comment lines. 

__device__ Double addNum(Double x1, Double x2) {
  return x1 + x2; 
}

__global__ void vecAdd(Double *in1, Double *in2, Double *out, Double len) {
  //@@ Insert code to implement vector addition here
  const int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < len) {
    const Double x = addNum(in1[i], in2[i]);
    out[i] = x;

  }
}

//@@ Insert code to implement timer start

long int cpuSecond() {
  struct timeval timer;
  gettimeofday(&timer, NULL);
  return timer.tv_sec * 1000000 + timer.tv_usec;
}

int main(int argc, char **argv) {
  
  int inputLength = 0;
  Double *hostInput1;
  Double *hostInput2;
  Double *hostOutput;
  Double *resultRef;
  Double *deviceInput1;
  Double *deviceInput2;
  Double *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  if (argc != 2){
    printf("!!\tNumber of arguments is not correct. Given %d, expected 1\n", argc - 1);
    exit(1);
  }

  printf("->\tStart of execution...\n");
  
  inputLength = std::atoi(argv[1]);

  printf("->\tInput length dim (%d)\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (Double*)malloc(inputLength * sizeof(Double));
  hostInput2 = (Double*)malloc(inputLength * sizeof(Double));
  hostOutput = (Double*)malloc(inputLength * sizeof(Double));
  resultRef  = (Double*)malloc(inputLength * sizeof(Double));

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = (rand() % 1000) * sin(i);
        hostInput2[i] = (rand() % 1000) * sin(i);
  }

  printf("->\tRandom input vectors created...\n");

  for (int i = 0; i < inputLength; i++){
    hostOutput[i] = hostInput1[i] + hostInput2[i];
  }

  printf("->\tReference vector created...\n");

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength * sizeof(Double));
  cudaMalloc(&deviceInput2, inputLength * sizeof(Double));
  cudaMalloc(&deviceOutput, inputLength * sizeof(Double));

  long int iStart = cpuSecond();

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(Double), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(Double), cudaMemcpyHostToDevice);
  long int iElaps = cpuSecond() - iStart;
 
  printf("~~\tTime to copy memory to GPU: %ld\n", iElaps);

  printf("->\tCuda memory initialized...\n");

  //@@ Initialize the 1D grid and block dimensions here
  dim3 dimGrid((inputLength + TPB - 1) / TPB, 1, 1);

  dim3 dimBlock(TPB, 1, 1);

  iStart = cpuSecond();
  //@@ Launch the GPU Kernel here
  vecAdd<<<dimGrid, dimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  iElaps = cpuSecond() - iStart;

  printf("~~\tTime to execute the kernel: %ld\n", iElaps);
  printf("->\tGPU computation terminated...\n");

  //@@ Copy the GPU memory back to the CPU here

  iStart = cpuSecond();
  cudaError err =cudaMemcpy(resultRef, deviceOutput, inputLength * sizeof(Double), cudaMemcpyDeviceToHost);
  if(err!=cudaSuccess) {
      printf("!!\tCUDA error copying to Host: %s\n", cudaGetErrorString(err));
      cudaFree(deviceInput1);
      cudaFree(deviceInput2);
      cudaFree(deviceOutput);
      exit(2);
  }

  iElaps = cpuSecond() - iStart;
  printf("~~\tTime to copy memory from GPU: %ld\n", iElaps);
 
  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < inputLength; i++){
    if(abs(resultRef[i] - hostOutput[i]) >= 1){
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
