%%writefile matrix_multiplication.cu

#include <stdio.h>
#include <sys/time.h>
#include <cstdlib>
#include <stdlib.h>

#define DataType float

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here

  // Compute the index of the thread
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  const int C_size = numARows * numBColumns;

  // If the thread index is larger than the size of the result matrix,
  // then no computation is needed
  if(idx >= C_size) return;

  // Extract index of row and column of the result matrix
  const int C_row = idx / numBColumns;
  const int C_col = idx % numBColumns;

  // Local variable to store the result of the MAC operation
  DataType cell_value = 0;

  // To get the value of the cell (C_row, C_col) in the result matrix,
  // we need to perform a dot product between the row C_row of A and the
  // column C_col of B
  for(int i = 0; i < numAColumns; i++){
    int A_index = C_row * numAColumns + i;
    int B_index = i * numBColumns + C_col;
    cell_value += (A[A_index] * B[B_index]);
  }

  // Write the result
  C[idx] = cell_value;

}

//@@ Insert code to implement a timer
__host__ long int get_time(void){
  struct timeval timer;
  gettimeofday(&timer, NULL);
  return timer.tv_sec * 1000000 + timer.tv_usec;
}

int main(int argc, char **argv) {

  DataType *hostA;      // The A matrix
  DataType *hostB;      // The B matrix
  DataType *hostC;      // The output C matrix
  DataType *resultRef;  // The reference result
  DataType *deviceA;    // A matrix for the device
  DataType *deviceB;    // B matrix for the device
  DataType *deviceC;    // C matrix for the device

  int numARows;         // number of rows in the matrix A
  int numAColumns;      // number of columns in the matrix A
  int numBRows;         // number of rows in the matrix B
  int numBColumns;      // number of columns in the matrix B
  int numCRows;         // == numARows
  int numCColumns;      // == numBcolumns
  int sizeA;            // number of cells in A
  int sizeB;            // number of cells in B
  int sizeC;            // number of cells in C

  long int timer;       // timer to compute the time for different operations

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args

  // The value of argc must be 4
  if(argc != 4) {
    printf("!!\tNumber of arguments is not correct. Given %d, expected 3\n", argc - 1);
    exit(1);
  }

  printf("->\tStart of execution...\n");

  // In case the provided values via args are not valid numbers, the behaviour of atoi is undefined.
  // Since there is no other solution, we just expect these values to be correct.
  numARows      = atoi(argv[1]);
  numAColumns   = atoi(argv[2]);
  numBColumns   = atoi(argv[3]);
  numBRows      = numAColumns;
  numCRows      = numARows;
  numCColumns   = numBColumns;

  // Compute the size of each matrix
  sizeA         = numARows * numAColumns;
  sizeB         = numBRows * numBColumns;
  sizeC         = numCRows * numCColumns;

  printf("->\tInput matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  //@@ Insert code below to allocate Host memory for input and output

  hostA     = new DataType[sizeA];
  hostB     = new DataType[sizeB];
  hostC     = new DataType[sizeC];
  resultRef = new DataType[sizeC];

  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU

  // Necessary to provide a random seed for the generation of random numbers
  srand(time(0));

  // Generate random numbers between -100 and 100 for each cell of both A and B
  for(int i = 0; i < sizeA; i++) hostA[i] = (rand() % 200 - 100) * sin(i);
  for(int i = 0; i < sizeB; i++) hostB[i] = (rand() % 200 - 100) * sin(i);

  printf("->\tRandom input matrices created...\n");

  // Compute the reference result
  for(int i = 0; i < numCRows; i++){
    for(int j = 0; j < numCColumns; j++){
      int Ref_index = i * numCColumns + j;
      for(int k = 0; k < numAColumns; k++){
        int A_index = i * numAColumns + k;
        int B_index = k * numBColumns +   j;
        resultRef[Ref_index] += (hostA[A_index] * hostB[B_index]);
      }
    }
  }

  printf("->\tReference matrix created...\n");

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc((void **)&deviceA, sizeA * sizeof(DataType));
  cudaMalloc((void **)&deviceB, sizeB * sizeof(DataType));
  cudaMalloc((void **)&deviceC, sizeC * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here

  // Get current timestamp
  timer = get_time();

  cudaMemcpy(deviceA, hostA, sizeA * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeB * sizeof(DataType), cudaMemcpyHostToDevice);

  printf("~~\tTime to copy memory to GPU: %ld\n", get_time() - timer);

  printf("->\tCuda memory initialized...\n");

  //@@ Initialize the grid and block dimensions here

  int TBP = 32;
  dim3 blocks_in_grid((sizeC + TBP -1) / TBP, 1, 1);
  dim3 threads_in_block(TBP, 1, 1);

  //@@ Launch the GPU Kernel here

  // Get current timestamp
  timer = get_time();

  gemm<<<blocks_in_grid, threads_in_block>>>(
    deviceA, deviceB, deviceC,
    numARows, numAColumns,
    numBRows, numBColumns
  );

  // The call to the kernel is asynchronoys, so we need to wait until all the threads are done
  cudaDeviceSynchronize();

  printf("~~\tTime to execute the kernel: %ld\n", get_time() - timer);
  printf("->\tGPU computation terminated...\n");

  //@@ Copy the GPU memory back to the CPU here

  // Get current timestamp
  timer = get_time();

  cudaError err = cudaMemcpy(hostC, deviceC, sizeC * sizeof(DataType), cudaMemcpyDeviceToHost);
  if(err!=cudaSuccess) {
      printf("!!\tCUDA error copying to Host: %s\n", cudaGetErrorString(err));
      cudaFree(deviceA);
      cudaFree(deviceB);
      cudaFree(deviceC);
      exit(2);
  }

  printf("~~\tTime to copy memory from GPU: %ld\n", get_time() - timer);
  //@@ Insert code below to compare the output with the reference

  bool error_found = false;

  for(int i = 0; i < numCRows; i++){
    for(int j = 0; j < numCColumns; j++){
      int index = i * numCColumns + j;
      if(abs(hostC[index] - resultRef[index]) >= 1){
        printf("!!\tError in position (%d, %d);\n", i , j);
        printf("!!\tExpected %f, found %f\n\n", resultRef[index], hostC[index]);
        error_found = true;
      }
    }
  }

  if(!error_found)
    printf("->\tThe result provided by the GPU is correct!\n");

  //@@ Free the GPU memory here

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here

  delete [] hostA;
  delete [] hostB;
  delete [] hostC;
  delete [] resultRef;

  printf("End of execution...\n");

  return 0;
}
