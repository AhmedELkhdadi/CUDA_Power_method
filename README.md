# CUDA_Power_method
An implementation of the power methode in C and CUDA

In a grid5000 machine or if you have An nvidia GPU and Cuda installed

Compile tp.cu with the following command 

`nvcc tp.cu -o tp_power –gpu-architecture=compute_61 --gpu-code=sm_61`

then run 

`./tp_power [Size of matrix]`
