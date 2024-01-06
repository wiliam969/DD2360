# Assignment 4

## Exercise 2 - Cuda Streams

- `ex_2/vector_sum_stream.cu` is the vec_add stream version;

To compile the version

```bash
nvcc ex_2/vector_sum_stream.cu -o vector_sum_stream
```

To execute the code:

```bash
./vector_sum_stream input_size stream_size
```

## Exercise 3 - Heat Equation with using NVIDIA libraries

To compile 

```bash
make
```

To execute the code:

```bash
./bin/sputniPIC.out inputfiles/GEM_2D.inp
```
