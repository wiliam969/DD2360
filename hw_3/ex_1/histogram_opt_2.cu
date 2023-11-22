%%writefile histogram_opt_2.cu

#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel( unsigned int *input,
                                  unsigned int *bins,
                                  unsigned int num_elements,
                                  unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics

  // Define shared memory
  __shared__ unsigned int s_bins[NUM_BINS];
  // Get thread id
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  // Variable to store the value of in[idx]
  int input_value;

  // Each thread is in charge of setting to zero an area of the shared memory
  for(int i = threadIdx.x; i < num_bins; i+=blockDim.x)
    s_bins[i] = 0;

  // If idx is a valid value to access the input, we increment the corresponding
  // bin in the shared memory
  if(idx < num_elements){

    input_value = input[idx];
    atomicAdd(&s_bins[input_value], 1);

  }

  // Wait until all threads in the block are done
  __syncthreads();

  // Each thread is in chare of updating one area of the global memory
  for(int i = threadIdx.x; i < num_bins; i+=blockDim.x)
    atomicAdd(&bins[i], s_bins[i]);

  return;
}

__global__ void convert_kernel( unsigned int *bins,
                                unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127

  const int idx = threadIdx.x + blockDim.x * blockIdx.x;

  bins[idx] = (bins[idx] > 127) ? 127 : bins[idx];

  return;
}

//@@ Insert code to implement a timer

double get_time() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char **argv) {

  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;
  double        timer;
  int           n_streams;
  int           stream_size;

  //@@ Insert code below to read in inputLength from args

  if(argc != 2){
    printf("!!\tERROR: Expected 1 argument, got %d\n", argc - 1);
    exit(1);
  }

  inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);

  //@@ Insert code below to allocate Host memory for input and output

  hostInput = new unsigned int[inputLength];
  hostBins  = new unsigned int[NUM_BINS];
  resultRef = new unsigned int[NUM_BINS];

  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)

  srand(time(0));

  for(int i = 0; i < inputLength; i++){
    hostInput[i] = rand() % NUM_BINS;
  }

  //@@ Insert code below to create reference result in CPU

  for(int i = 0; i < NUM_BINS; i++) resultRef[i] = 0;
  for(int i = 0; i < inputLength; i++)
    resultRef[hostInput[i]] = min(resultRef[hostInput[i]] + 1, 127);

  //@@ Init timer

  timer = get_time();

  //@@ Insert code below to allocate GPU memory here

  cudaMalloc((void **)&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc((void **)&deviceBins,  NUM_BINS    * sizeof(unsigned int));

  //@@ Insert code below to initialize streams

  stream_size = 1024 * 100;
  n_streams = (inputLength + stream_size - 1) / stream_size;

  cudaStream_t streams[n_streams];
  for(int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

  //@@ Insert code to Copy memory to the GPU here

  int remaining_length = inputLength;

  for(int i = 0; i < n_streams; i++){
    int offset = i * stream_size;
    cudaError err = cudaMemcpyAsync(&deviceInput[offset],
                                    &hostInput[offset],
                                    min(remaining_length, stream_size) * sizeof(unsigned int),
                                    cudaMemcpyHostToDevice,
                                    streams[i]);

    if(err != cudaSuccess) {
        printf("!!\tCUDA error copying to Device: %s\n", cudaGetErrorString(err));
        cudaFree(deviceInput);
        cudaFree(deviceBins);
        exit(2);
    }

    remaining_length -= stream_size;
  }

  //@@ Initialize the grid and block dimensions here

  int TPB = 1024;

  //@@ Launch the GPU Kernel here

  for(int i = 0; i < n_streams; i++){
    int offset = i * stream_size;
    histogram_kernel<<< (stream_size + TPB - 1) / TPB, TPB, 0, streams[i]>>>(
      &deviceInput[offset],
      deviceBins,
      stream_size,
      NUM_BINS
    );
  }

  cudaDeviceSynchronize();

  //@@ Initialize the second grid and block dimensions here

  dim3 blocks_in_grid_2((NUM_BINS + TPB - 1) / TPB);
  dim3 threads_in_block_2(TPB);

  //@@ Launch the second GPU Kernel here

  convert_kernel<<<blocks_in_grid_2, threads_in_block_2>>>(
    deviceBins,
    NUM_BINS
  );
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here

  cudaError err = cudaMemcpyAsync(&hostBins[0],
                                  &deviceBins[0],
                                  NUM_BINS * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost);

  if(err != cudaSuccess) {
      printf("!!\tCUDA error copying to Host: %s\n", cudaGetErrorString(err));
      cudaFree(deviceInput);
      cudaFree(deviceBins);
      exit(2);
  }

  //@@ Insert code below to compare the output with the reference

  bool found_erorr = false;
  for(int i = 0; i < NUM_BINS; i++){
    if(hostBins[i] != resultRef[i]){
      printf("!!\tResult mismatch in position %d: expected %d, got %d\n", i, resultRef[i], hostBins[i]);
      found_erorr = true;
    }
  }

  if(!found_erorr) printf("->\tResult computed by GPU is correct!\n");

  //@@ Free the GPU memory here

  for(int i = 0; i < n_streams; i++) cudaStreamDestroy(streams[i]);
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here

  delete [] hostInput;
  delete [] hostBins;
  delete [] resultRef;

  //@@ Get execution time
  printf("~~\tExecution time is %f\n", get_time() - timer);

  printf("->\tEnd of execution");

  return 0;
}
