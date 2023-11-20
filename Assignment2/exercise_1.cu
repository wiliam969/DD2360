
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h> 

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

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
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
  if (argc > 1){
    inputLength = std::atoi(argv[1]);

    printf("arg 0 %s\n", argv[0]);
    printf("arg 1 %s\n", argv[1]); 
  }

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (Double*)malloc(inputLength * sizeof(Double));
  hostInput2 = (Double*)malloc(inputLength * sizeof(Double));
  hostOutput = (Double*)malloc(inputLength * sizeof(Double));
  resultRef  = (Double*)malloc(inputLength * sizeof(Double));

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() % 10;
        hostInput2[i] = rand() % 10;
  }

  for (int i = 0; i < inputLength; i++){
    hostOutput[i] = hostInput1[i] + hostInput2[i];
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength * sizeof(Double));
  cudaMalloc(&deviceInput2, inputLength * sizeof(Double));
  cudaMalloc(&deviceOutput, inputLength * sizeof(Double));

  double iStart = cpuSecond();

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(Double), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(Double), cudaMemcpyHostToDevice);
  double iElaps = cpuSecond() - iStart;

  printf("Copy CPU TO GPU: %f", iElaps);

  //@@ Initialize the 1D grid and block dimensions here
  dim3 dimGrid((inputLength + TPB - 1) / TPB, 1, 1);

  dim3 dimBlock(TPB, 1, 1);

  iStart = cpuSecond();
  //@@ Launch the GPU Kernel here
  vecAdd<<<dimGrid, dimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  iElaps = cpuSecond() - iStart;

  printf("Kernel: %f", iElaps);
  //@@ Copy the GPU memory back to the CPU here

  iStart = cpuSecond();
  cudaMemcpy(resultRef, deviceOutput, inputLength * sizeof(Double), cudaMemcpyDeviceToHost);
  iElaps = cpuSecond() - iStart;

  printf("Copy GPU TO CPU: %f", iElaps);
  
  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < inputLength; i++){
     if(resultRef[i] == hostOutput[i]){
        continue;
     }

     printf("Missmatch occured with row: %d, HValue: %f, DValue: %f\n", i, hostOutput[i], resultRef[i]);
  }

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
