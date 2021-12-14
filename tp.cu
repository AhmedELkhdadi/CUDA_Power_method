/* indent -nfbs -i4 -nip -npsl -di0 -nut iterated_seq.c */
/* Auteur: C. Bouillaguet et P. Fortin (Univ. Lille) */ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "defs.h"

#define NB_THREADS_BLOC 256

__device__ void init_ligneGPU(REAL_T *d_A, long i, long n)
{
    for (long j = 0; j < n; j++)
    {
        d_A[i * n + j] = (((REAL_T)((i * i * PRNG_1 + j * j * PRNG_2) & PRNG_MAX)) / PRNG_MAX) / n;
    }
    for (long k = 1; k < n; k *= 2)
    {
        if (i + k < n)
        {
            d_A[i * n + i + k] = ((i - k) * PRNG_2 + i * PRNG_1) % RANGE;
        }
        if (i - k >= 0)
        {
            d_A[i * n + i - k] = ((i + k) * PRNG_2 + i * PRNG_1) % RANGE;
        }
    }
}


__global__ void init(float *d_A, float *d_X, int d_n){
        // printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
        unsigned int i = blockDim.x* blockIdx.x + threadIdx.x;
        if(i < d_n){
            // init_ligne(d_A, i, d_n);
            init_ligneGPU(d_A, i, d_n);
            d_X[i] = 1.0 / d_n;
        }
}

__global__ void prodMatVec(float *d_A, float *d_X, int d_n, float *d_Y, float *d_N, float *d_norme){
        unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
        if(i < d_n){
            // *d_norme = 0 ;
        d_Y[i] = 0;
        d_N[i] = 0;
        for(unsigned int j = 0; j < d_n; j++){
            d_Y[i] += d_A[i*d_n + j] * d_X[j];
            d_N[i] += d_Y[i] * d_Y[i];
        }
        atomicAdd(d_norme, d_N[i]);
    }
}



__global__  void normalisationEtErreur(float *d_X, float *d_Y, float *d_norme, float *d_E, float *d_erreur, int d_n){
        unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
        if(i < d_n){
            // *d_erreur  = 0 ;
        d_Y[i] = d_Y[i] / *d_norme;
        d_E[i] = (d_X[i] - d_Y[i]) * (d_X[i] - d_Y[i]);
        atomicAdd(d_erreur, d_E[i]);         
        }
}

__global__ void swichXandY(float *d_X,float *d_Y, int d_n){
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < d_n){
        d_X[i] = d_Y[i];
    }
}



int main(int argc, char **argv){
    int i, j, n;
    long long size_a, size_x;
    REAL_T  delta;
    REAL_T * error;
    double start_time, total_time;
    int n_iterations;
    FILE *output;
    REAL_T *X;
    REAL_T norm;
    // GPU
    int *d_n;
    REAL_T *d_norme, *d_erreur;
    REAL_T *d_A, *d_X, *d_Y, *d_N, *d_E;
    
    if (argc < 2) {
        printf("USAGE: %s [n]\n", argv[0]);
        exit(1);
    }
    
    
    n = atoi(argv[1]);

    // tailles blocs
    int taille_bloc_x = n;
    int taille_bloc_y = 1;
    
    size_a = (long long) n * n * sizeof(REAL_T);
    size_x = (long long) n * sizeof(REAL_T);
    printf("taille de la matrice : %.1f G\n", size_a / 1073741824.);
    
     
    long long size_X =  n * sizeof(REAL_T);
    
    X = (REAL_T *)malloc(size_X);
    error = (REAL_T *)malloc(sizeof(REAL_T));

    // /*** allocation de la matrice et des vecteurs sur GPU ***/
    // /*** initialisation de la matrice et de x ***/
    
    cudaMalloc((void **) &d_n, sizeof(int));
    cudaMalloc((void **) &d_norme, sizeof(REAL_T));
    cudaMalloc((void **) &d_erreur, sizeof(REAL_T));
    
    cudaMalloc((void **) &d_A, size_a);
    cudaMalloc((void **) &d_X, size_x);
    cudaMalloc((void **) &d_Y, size_x);
    cudaMalloc((void **) &d_N, size_x);
    cudaMalloc((void **) &d_E, size_x);
    
    /*** envoyer la taille de la matrice au GPU ***/
    cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsParBloc(NB_THREADS_BLOC, 1);
    dim3 tailleGrille(ceil(n/NB_THREADS_BLOC),1);
    
    start_time = my_gettimeofday();
    *error = INFINITY;
    n_iterations = 0;
    // // while (error > ERROR_THRESHOLD) {
    while (*error > 10e-5) {
    //     printf("-1 test\n");

        printf("iteration %4d, erreur actuelle %f\n", n_iterations, *error);
        printf("0 test\n");
        

        // initialisation A X 
        init<<<tailleGrille, threadsParBloc>>>(d_A, d_X, n);
    //     // cudaDeviceSynchronize();
        printf("1 test");

        prodMatVec<<<tailleGrille, threadsParBloc>>>(d_A, d_X, n, d_Y, d_N, d_norme);
    //     printf("2 test");
        
        cudaMemcpy(X,d_X, size_X, cudaMemcpyDeviceToHost);



        
    //     /*** y <--- y / ||y|| ***/
        normalisationEtErreur<<<tailleGrille, threadsParBloc>>>(d_X, d_Y, d_norme, d_E, d_erreur,n);
    //     printf("test");

    //     printf("1 %g\n", *error);

        cudaMemcpy(error,d_erreur, sizeof(REAL_T), cudaMemcpyDeviceToHost);
        
    //     /// copier d_y dans d_x
        // d_X = d_Y;
        swichXandY<<<tailleGrille, threadsParBloc>>>(d_X,d_Y,n);
    //     printf("2 %g\n", *error);
    //     /*** error <--- ||x - y|| ***/
        *error = sqrt(*error);
    //     printf("4 %g\n", *error);
    //     /*** x <--> y ***/

        n_iterations++;
    }
    cudaMemcpy( X,d_X, size_X, cudaMemcpyDeviceToHost);
    cudaMemcpy(&norm, d_norme , sizeof(REAL_T), cudaMemcpyDeviceToHost);


    total_time = my_gettimeofday() - start_time;
    printf("erreur finale après %4d iterations : %g (|VP| = %g)\n", n_iterations, *error, norm);
    printf("temps : %.1f s      Mflop/s : %.1f \n", total_time, (2.0 * n * n + 7.0 * n) * n_iterations / 1048576. / total_time);

    /*** stocke le vecteur propre dans un fichier ***/
    output = fopen("result.out", "w");
    if (output == NULL) {
        perror("impossible d'ouvrir result.out en écriture");
        exit(1);
    }
    fprintf(output, "%d\n", n);
    for (i = 0; i < n; i++) {
        fprintf(output, "%.17g\n", X[i]);
    }
    fclose(output);
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_A);
    cudaFree(d_N);
    cudaFree(d_E);
    cudaFree(d_erreur);
    cudaFree(d_norme);
    free(X);
    free(error);
}
