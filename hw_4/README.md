# Assignment 4

## Exercise 2 - Cuda Streams

- `ex_2/vector_sum_stream_variable_kernel_launches.cu.cu` is the vec_add stream version;

To compile the version

```bash
nvcc ex_2/vector_sum_stream_variable_kernel_launches.cu -o vector_sum_stream_variable_kernel_launches
```

To execute the code:

```bash
./vector_sum_stream_variable_kernel_launches input_size stream_size
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

