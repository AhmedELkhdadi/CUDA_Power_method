# CUDA_Power_method
An implementation of the power methode in C and CUDA

In grid5000 or if you have 

Compile tp.cu with the following command 

`nvcc tp.cu -o tp_power –gpu-architecture=compute_61 --gpu-code=sm_61`

then run 

`./tp_power [Size of matrixe]`
`
