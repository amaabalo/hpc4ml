#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mkl.h>

typedef struct NeuralNetwork NeuralNetwork;
double *matmul(double *A, double *B, double *C, int rows_a, int cols_a, int rows_b, int cols_b, int *rows_c, int *cols_c);
double *mat(int rows, int cols, double (*f)(int, int));
double f(int i, int j);
void feed_forward(NeuralNetwork *nn, double *exec_time, double *checksum);
void feed_forward_MKL(NeuralNetwork *nn, double *exec_time, double *checksum);
double get_checksum(NeuralNetwork *nn);
void print_matrix(double *A, int rows_a, int cols_a);
NeuralNetwork *NeuralNetwork_new();
void NeuralNetwork_del(NeuralNetwork *nn);

struct NeuralNetwork {
	int input_rows;
	int n_layers;
	int *layer_dims;
	double **W;
	double **Z;
	double *input;
};

double f(int i, int j) {
	return (0.4 + ((i+j) % 40 - 20) / 40.0);
}

double *mat(int rows, int cols, double (*f)(int, int)) {
	int n = rows * cols;
	double *m = (double *)malloc(n * sizeof(double));
	if (f == NULL) {
		memset((void *)m, 0, n);
		return (m);
	}
	
	int i, c, r;
	for (i = 0; i < n; i++) {
		r = i / cols;
		c = i % cols;
		m[i] = (*f)(r, c);	
	}
	return m;
}

NeuralNetwork *NeuralNetwork_new(int input_rows, int n_layers, int *layer_dims) {

	NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));

	nn->input_rows = input_rows;
	nn->n_layers = n_layers;
	nn->layer_dims = layer_dims;
	

	// Initialise the input
	nn->input = mat(input_rows, 1, f);
	

	// Initialise the weights at each layer
	nn->W = (double **)malloc(sizeof(double *) * n_layers);
	int n_cols = input_rows;
	int i;
	for (i = 0; i < n_layers; i++) {
		(nn->W)[i] = mat(layer_dims[i], n_cols, f);
		n_cols = layer_dims[i]; 
	}

	// Allocate space for zs
	nn->Z = (double **)malloc(sizeof(double *) * n_layers);
	for (i = 0; i < n_layers; i++) {
		(nn->Z)[i] = mat(layer_dims[i], 1, NULL);
	}
	return nn;
}

void NeuralNetwork_del(NeuralNetwork *nn) {
	free(nn->input);

	int i;
	for (i = 0; i < nn->n_layers; i++) {
		free((nn->W)[i]);
		free((nn->Z)[i]);
	}

	free(nn->W);
	free(nn->Z);
	free(nn); //:(
}

double get_checksum(NeuralNetwork *nn) {

	double sum = 0;
	int i;
	int l = nn->n_layers - 1;
	int lim = (nn->layer_dims)[l];
	for (i = 0; i < lim; i++) {
		sum += ((nn->Z)[l])[i];
	}
	return sum;
}

void feed_forward(NeuralNetwork *nn, double *exec_time, double *checksum){
	int l;
	int cols_a;
	int rows_c, cols_c;
	struct timespec start, end;

	clock_gettime(CLOCK_MONOTONIC, &start);		
	for (l = 0; l < nn->n_layers; l++) {
		double *x;
		if (l == 0) {
			x = nn->input;
			cols_a = nn->input_rows;
		} else {
			x = (nn->Z)[l - 1];
			cols_a = (nn->layer_dims)[l - 1];
		}

		double *Z = (nn->Z)[l];
		matmul((nn->W)[l], x, Z, (nn->layer_dims)[l], cols_a, cols_a, 1, &rows_c, &cols_c);
	
		//RELU activation
		int n = rows_c * cols_c;
		int i; 
		for (i = 0; i < n; i++) {
			if (Z[i] < 0) {
				Z[i] = 0;
			}
		}
		//(nn->Z)[l] = Z;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);	
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	*exec_time = time_usec;
	*checksum = get_checksum(nn);
}

void feed_forward_MKL(NeuralNetwork *nn, double *exec_time, double *checksum){
	int l;
	int cols_a;
	int rows_c, cols_c;
	struct timespec start, end;

	clock_gettime(CLOCK_MONOTONIC, &start);		
	for (l = 0; l < nn->n_layers; l++) {
		double *x;
		if (l == 0) {
			x = nn->input;
			cols_a = nn->input_rows;
		} else {
			x = (nn->Z)[l - 1];
			cols_a = (nn->layer_dims)[l - 1];
		}

		//double *Z = matmul((nn->W)[l], x, (nn->layer_dims)[l], cols_a, cols_a, 1, &rows_c, &cols_c);
		double *Z = (nn->Z)[l];
		cblas_dgemv(CblasRowMajor, CblasNoTrans, (nn->layer_dims)[l], cols_a, 1, (nn->W)[l], cols_a, x, 1, 0, Z, 1); 
		rows_c = (nn->layer_dims)[l];
		cols_c = 1;
	
		//RELU activation
		int n = rows_c * cols_c;
		int i; 
		for (i = 0; i < n; i++) {
			if (Z[i] < 0) {
				Z[i] = 0;
			}
		}

		//(nn->Z)[l] = Z;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);	
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	*exec_time = time_usec;
	*checksum = get_checksum(nn);
}

void print_matrix(double *A, int rows_a, int cols_a){
	
	int r;
	int c;
	
	for (r = 0; r < rows_a; r++) {
		for (c = 0; c < cols_a; c++) {
			printf("%f ", A[r * cols_a + c]);
		}
		printf("\n");
	}
}

// Compute AB, return C
double *matmul(double *A, double *B, double *C, int rows_a, int cols_a, int rows_b, int cols_b, int *rows_c, int *cols_c) {
	if (cols_a != rows_b) {
		return NULL;
	}

	if (C == NULL) {
		C = (double *)malloc(rows_a * cols_b * sizeof(double));
	}
	*rows_c = rows_a;
	*cols_c = cols_b;

	int i = 0;
	int r_a, c_b, r_b;
	double sum;
	for (r_a = 0; r_a < rows_a; r_a++) {
		for (c_b = 0; c_b < cols_b; c_b++) {
			sum = 0;
			for (r_b = 0; r_b < rows_b; r_b++){
				sum += A[r_a * cols_a + r_b] * B[r_b * cols_b + c_b];
			}
			C[i] = sum;
			i += 1;
		}
	}
	
	return C;
}

int main() {
	double C3_exec_time = 34.576209323015064; // Lowest time measured from earlier runs

	int nn_dims[] = {4000, 1000};
	NeuralNetwork *nn = NeuralNetwork_new(256 * 256, 2, nn_dims);
	double exec_time, checksum;
	feed_forward(nn, &exec_time, &checksum);

	printf("*C5*\n");
	printf("Time: %.10f secs\n", exec_time / 1000000);
	printf("Checksum: %f\n", checksum);
	printf("Speedup w.r.t. C3: %f\n", C3_exec_time * 1000000 / exec_time);
	printf("\n");

	feed_forward_MKL(nn, &exec_time, &checksum);
	printf("*C6*\n");
	printf("Time: %.10f secs\n", exec_time / 1000000);
	printf("Checksum: %f\n", checksum);
	printf("Speedup w.r.t. C3: %f\n", C3_exec_time * 1000000 / exec_time);
	printf("\n");

	NeuralNetwork_del(nn);
	return 0;
}
