#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mkl_lapacke.h"

double *generate_matrix(int size)
{
    int i;
    double *matrix = (double *)malloc(sizeof(double) * size * size);
    srand(1);

    for (i = 0; i < size * size; i++)
    {
        matrix[i] = rand() % 100;
    }

    return matrix;
}

void print_matrix(const char *name, double *matrix, int size)
{
    int i, j;
    printf("matrix: %s \n", matrix);

    for (i = 0; i < size; i++)
    {
            for (j = 0; j < size; j++)
            {
                printf("%f ", matrix[i * size + j]);
            }
            printf("\n");
    }
}

int check_result(double *bref, double *b, int size) {
    int i;
    for(i=0;i<size*size;i++) {
        if (abs(bref[i]-b[i]) != 0) return 0;
    }
    return 1;
}

double *init_mat(int size)
{
    double *mat = (double *)malloc(sizeof(double) * size * size);

    for (int i = 0; i < size * size; i++)
    {
        mat[i] = 0;
    }

    return mat;
}

double *product_mat(double *matA, double *matB, int size)
{
	double *matC = (double *)malloc(sizeof(double) * size * size);

	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			matC[i*size + j] = 0;

			for (int k = 0; k < size; k++)
			{
				matC[i*size + j] += matA[i*size + k]*matB[k*size + j];
			}
		}
	}
	return matC;
}

double *transpose(double *mat, int size) {

	double *trans = (double *)malloc(sizeof(double) * size * size);
    	for(int i = 0; i < size; i++)
      {
       		for(int j = 0; j < size; j++)
          {

          		trans[i*size + j] = mat[j*size + i];
		}
	}
	return trans;
}


void decompQR(double *matA, int size, double *matQ, double *matR)
{

	for(int i = 0; i < size; i++)
	{
		double var = 0;

		for(int j = 0; j < size; j++)
		{
			var += matA[j*size + i]*matA[j*size + i];
		}
		matR[i*(size + 1)] = sqrt(var);
		for(int j = 0; j < size; j++)
		{
			matQ[j*size + i] = matA[j*size + i] / matR[i*(size + 1)];
		}
		for(int j = i+1; j < size; j++)
		{
			var = 0;
			for(int k = 0; k < size; k++)
			{
				var = var + matA[k*size + j] * matQ[k*size + i];
			}
			matR[i*size + j] = var;
			for(int k = 0; k < size; k++)
			{
				matA[k*size + j] = matA[k*size + j] - matR[i*size + j] * matQ[k*size + i];
			}
		}
	}
}

double *QR(double *matA, double *matB, int size)
{
	double *matQ = init_mat(size);
	double *matR = init_mat(size);
	double *matX = init_mat(size);

  decompQR(matA, size, matQ, matR);
	double *matProd = product_mat(transpose(matQ, size), matB, size);

	for(int i = size - 1; i > -1; i--)
  {
		for(int j = size - 1; j > -1; j--)
    {
			double var = 0;
			for(int k = i + 1; k < size; k++)
      {
				var += matR[i*size + k]* matX[k*size + j];
			}
		matX[i*size + j] = (matProd[i*size + j] - var)/matR[i*(size+1)];
		}
	}
	return matX;
}

    void main(int argc, char *argv[])
    {

        int size = atoi(argv[1]);

        double *a, *aref;
        double *b, *bref;

        a = generate_matrix(size);
        aref = generate_matrix(size);
        b = generate_matrix(size);
        bref = generate_matrix(size);


        //print_matrix("A", a, size);
        //print_matrix("B", b, size);

        // Using MKL to solve the system
        MKL_INT n = size, nrhs = size, lda = size, ldb = size, info;
        MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT)*size);

        clock_t tStart = clock();
        info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
        printf("Time taken by MKL: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

        tStart = clock();
	      b = QR(a, b, size);
        printf("Time taken by my implementation: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

        if (check_result(bref,b,size)==1)
            printf("Result is ok!\n");
        else
            printf("Result is wrong!\n");

        //print_matrix("X", b, size);
        //print_matrix("Xref", bref, size);
    }
