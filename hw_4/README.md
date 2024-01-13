# Assignment 4

## Exercise 2 - Cuda Streams

- `ex_2/vector_sum_stream_variable_kernel_launches.cu.cu` is a vec_add version with variable streams. Using this N Streams will be populated with the second argument;
- `ex_2/vector_sum_stream_variable_streams.cu` is the vec_add stream version in which the streams are capped to 4. Given the second parameter the input will be divided into segments and executed on the respective 4 streams;

To compile the version

```bash
nvcc ex_2/vector_sum_stream_variable_kernel_launches.cu -o vector_sum_stream_variable_kernel_launches
nvcc ex_2/vector_sum_stream_variable_streams.cu -o vector_sum_stream_variable_streams
```

To execute the code:

```bash
./vector_sum_stream_variable_kernel_launches input_size stream_size
./vector_sum_stream_variable_streams input_size segment_size
```

## Exercise 3 - Heat Equation with using NVIDIA libraries

The project was developed through Google Colab, and the files still have the Colab's directives to write files.

In order to compile, use

```bash
nvcc heat_eq.cu -lcublas_static -lcublasLt_static -lculibos  -lcusparse -o  execute
```

To run, use

```bash
./ex <size> <iterations>
```

