/* indent -nfbs -i4 -nip -npsl -di0 -nut checker.c */
/* Auteur: C. Bouillaguet (Univ. Lille 1), P. Fortin (UPMC) */ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#include "defs.h"

int main(int argc, char **argv){
  long i, j, k, n;
  REAL_T norm, inv_norm, error, delta;
  double start_time;
  REAL_T *A, *X, *Y;
  FILE *input;

  if (argc < 2) {
    printf("USAGE: %s [file]\n", argv[0]);
    exit(1);
  }
  input = fopen(argv[1], "r");
  if (input == NULL) {
    fprintf(stderr, "impossible d'ouvrir %s en lecture", argv[1]);
    exit(1);
  }
  fscanf(input, "%ld", &n);
  printf("lu : n = %ld\n", n);


  X = (REAL_T *)malloc(n * sizeof(REAL_T));
  Y = (REAL_T *)malloc(n * sizeof(REAL_T));
  if ((X == NULL) || (Y == NULL)) {
    perror("impossible d'allouer la mémoire");
    exit(1);
  }
  for (i = 0; i < n; i++) {
    fscanf(input, (sizeof(REAL_T) == sizeof(double) ? "%lf\n" : "%f\n"), X + i);
  }
  fclose(input);

  start_time = my_gettimeofday();
  error = 0;

#pragma omp parallel private(i,j,k,A) shared(n,X,Y)
  {
#pragma omp master
#ifdef _OPENMP
    printf("Number of threads: %d \n", omp_get_num_threads());
#endif
    A = (REAL_T *)malloc(n * sizeof(REAL_T));
    if (A == NULL) {
      perror("impossible d'allouer la mémoire");
      exit(1);
    }
#pragma omp for
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
	A[j] = (((REAL_T)((i * i * PRNG_1 + j * j * PRNG_2) & PRNG_MAX)) / PRNG_MAX) / n;
      }
      for (k = 1; k < n; k *= 2) {
	if (i + k < n) {
	  A[i + k] = ((i - k) * PRNG_2 + i * PRNG_1) % RANGE;
	}
	if (i - k >= 0) {
	  A[i - k] = ((i + k) * PRNG_2 + i * PRNG_1) % RANGE;
	}
      }

      Y[i] = 0;
      for (j = 0; j < n; j++) {
	Y[i] += A[j] * X[j];
      }
    }
    free(A);
  }

  /*** norm <--- ||y|| ***/
  norm = 0;
  for (i = 0; i < n; i++) {
    norm += Y[i] * Y[i];
  }
  norm = sqrt(norm);

  /*** y <--- y / ||y|| ***/
  inv_norm = 1.0 / norm;
  for (i = 0; i < n; i++) {
    Y[i] *= inv_norm;
  }

  /*** error <--- ||x - y|| ***/
  error = 0;
  for (i = 0; i < n; i++) {
    delta = X[i] - Y[i];
    error += delta * delta;
  }
  error = sqrt(error);

  printf("erreur : %g (|VP| = %g)\n", error, norm);
  printf("time : %.1f s\n", my_gettimeofday() - start_time);

  free(X);
  free(Y);
}
