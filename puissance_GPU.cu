/* indent -nfbs -i4 -nip -npsl -di0 -nut iterated_seq.c */
/* Auteur: C. Bouillaguet et P. Fortin (Univ. Lille) */ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "defs.h"

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

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


__global__ void init(REAL_T *d_A, REAL_T *d_X, int d_n){
        unsigned int i = blockDim.x* blockIdx.x + threadIdx.x;
        if(i < d_n){
            
            init_ligneGPU(d_A, i, d_n);
            d_X[i] = 1.0 / d_n;
        }
}

__global__ void prodMatVec(REAL_T *d_A, REAL_T *d_X, int d_n, REAL_T *d_Y, REAL_T *d_N, REAL_T *d_norme){
        unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
        if(i < d_n){
        d_Y[i] = 0;
        for(unsigned int j = 0; j < d_n; j++){
            d_Y[i] += d_A[i*d_n + j] * d_X[j];
        }
        d_N[i] = d_Y[i] * d_Y[i];
        // if(n == )
        // *d_norme = 0 ; 
        atomicAdd(d_norme, d_N[i]);
    }
}


__global__ void errorAndNormTozero(REAL_T *d_norme, REAL_T *d_erreur, int d_n){
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
        if(i == 0){
            *d_erreur= 0;
            *d_norme= 0;
    }
}


__global__  void normalisationEtErreur(REAL_T *d_X, REAL_T *d_Y, REAL_T *d_norme, REAL_T *d_E, REAL_T *d_erreur, int d_n){
        unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
        
        if(i < d_n){
            d_Y[i] = d_Y[i] / sqrt(*d_norme);
            d_E[i] = (d_X[i] - d_Y[i]) * (d_X[i] - d_Y[i]);
            // *d_erreur = 0;
            
            atomicAdd(d_erreur, d_E[i]);   
        }
}

__global__ void calculateError(REAL_T *d_X, REAL_T *d_Y, REAL_T *d_E, REAL_T *d_erreur, int d_n){
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
        
        if(i < d_n){
            d_E[i] = (d_X[i] - d_Y[i]) * (d_X[i] - d_Y[i]);
            atomicAdd(d_erreur, d_E[i]);   
        }

}

__global__ void swichXandY(REAL_T *d_X,REAL_T *d_Y, int d_n){
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < d_n){
        d_X[i] = d_Y[i];
    }
}



int main(int argc, char **argv){
    int i, j, n;
    long long size_a, size_x;
    REAL_T  delta;
    REAL_T *error;
    double start_time, total_time;
    int n_iterations;
    FILE *output;
    REAL_T *X;
    REAL_T *norm;
    // GPU
    int *d_n;
    REAL_T *d_norme, *d_erreur;
    REAL_T *d_A, *d_X, *d_Y, *d_N, *d_E;
    REAL_T *tmp;
    
    REAL_T zero = 0;

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
    norm = (REAL_T *)malloc(sizeof(REAL_T));

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
    // cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsParBloc(NB_THREADS_BLOC, 1);
    dim3 tailleGrille(ceil(n/NB_THREADS_BLOC),1);
    
    start_time = my_gettimeofday();
    *error = INFINITY;
    n_iterations = 0;
    
    init<<<tailleGrille, threadsParBloc>>>(d_A, d_X, n);
    while (*error > 1e-5) {
        
        errorAndNormTozero<<<tailleGrille, threadsParBloc>>>(d_erreur, d_norme, n);
        
        prodMatVec<<<tailleGrille, threadsParBloc>>>(d_A, d_X, n, d_Y, d_N, d_norme);
        
        /*** y <--- y / ||y|| ***/
        normalisationEtErreur<<<tailleGrille, threadsParBloc>>>(d_X, d_Y, d_norme, d_E, d_erreur,n);

        /*** calcule de l'erreur ***/
        cudaMemcpy(error,d_erreur, sizeof(REAL_T), cudaMemcpyDeviceToHost);
        *error = sqrt(*error);
        
        /// copier d_y dans d_x
        /*** x <--> y ***/
        tmp = d_X;
        d_X = d_Y;
        d_Y = tmp;

    
        n_iterations++;
        printf("iteration %4d, erreur actuelle %g\n", n_iterations, *error);
    }

    cudaMemcpy( X,d_X, size_X, cudaMemcpyDeviceToHost);
    cudaMemcpy(norm, d_norme , sizeof(REAL_T), cudaMemcpyDeviceToHost);
    *norm = sqrt(*norm);

    total_time = my_gettimeofday() - start_time;
    printf("erreur finale après %4d iterations : %g (|VP| = %g)\n", n_iterations, *error, *norm);
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
