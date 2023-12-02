# Assignment 3

## Exercise 1 - Histogram and Atomics

There are three different versions of the same algorithm. 

- `ex_1/histogram_no_opt.cu` is the version without optimizations;
- `ex_1/histogram_opt_1.cu` is the version which exploits unified memory and which has been used to write the report;
- `ex_1/histogram_opt_2.cu` is the version which exploits streams.

To compile each version

```bash
nvcc ex_1/file_name.cu -o ex
```

To execute the code:

```bash
./ex input_size
```

## Exercise 2 - Partical Simulation

To compile 

```bash
make
```

To execute the code:

```bash
./bin/sputniPIC.out inputfiles/GEM_2D.inp
```
