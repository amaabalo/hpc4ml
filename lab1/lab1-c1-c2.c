#include <stdio.h>
#include <stdlib.h>
#include <time.h>
 
//#define ARRAY_SIZE 134217728 * 5 // 5GiB
#define ARRAY_SIZE 1000008
//#define ARRAY_SIZE 100000
#define NUM_TRIALS 50

double c1(double *array);
double c2(double *array);
void evaluate(double (*fp)(), double *array);
void printResults(char *name, int flops, float duration);


double c1(double *array) {
	FILE *fp = fopen("/dev/null", "w");
	struct timespec start;
	struct timespec end;
	
	int i;
	double sum = 0;
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < ARRAY_SIZE; i++) {
		sum += array[i] * 2;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	fprintf(fp, "%f\n", sum);
	fclose(fp);
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	printf("%f\n", time_usec/1000000);
	return (time_usec);
}


double c2(double *array) {
	struct timespec start;
	struct timespec end;
	
	int i;
	double sum = 0;
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < ARRAY_SIZE; i += 8) {
		sum += array[i] * 2;
		sum += array[i + 1] * 2;
		sum += array[i + 2] * 2;
		sum += array[i + 3] * 2;
		sum += array[i + 4] * 2;
		sum += array[i + 5] * 2;
		sum += array[i + 6] * 2;
		sum += array[i + 7] * 2;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	//FILE *fp = fopen("/dev/null", "w");
	//fprintf(fp, "%f\n", sum);
	//fclose(fp);
	printf("%f\n", sum);
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	printf("%f\n", time_usec/1000000);
	return (time_usec);
}

double c1a(double *array) {
	FILE *fp = fopen("/dev/null", "w");
	struct timespec start;
	struct timespec end;

	int j;
	double sum;
	for (j = 0; j < 10; j++) {	
	int i = 0;
	sum = 0;
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < ARRAY_SIZE; i++) {
		sum += array[i] * 2;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	printf("%f\n", time_usec/1000000);
	}
	fprintf(fp, "%f\n", sum);
	fclose(fp);
	return (sum);
}
double c2a(double *array) {
	FILE *fp = fopen("/dev/null", "w");
	struct timespec start;
	struct timespec end;

	int j;
	double sum;
	for (j = 0; j < 10; j++) {	
	int i = 0;
	sum = 0;
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < ARRAY_SIZE; i += 8) {
		sum += array[i] * 2;
		sum += array[i + 1] * 2;
		sum += array[i + 2] * 2;
		sum += array[i + 3] * 2;
		sum += array[i + 4] * 2;
		sum += array[i + 5] * 2;
		sum += array[i + 6] * 2;
		sum += array[i + 7] * 2;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	printf("%f\n", time_usec/1000000);
	}
	fprintf(fp, "%f\n", sum);
	fclose(fp);
	return (sum);
}

double c3(double *array) {
	FILE *fp = fopen("/dev/null", "w");
	struct timespec start;
	struct timespec end;
	
	int i;
	double sum = 0;
	double sum1 = 0;
	double sum2 = 0;
	double sum3 = 0;
	double sum4 = 0;
	double sum5 = 0;
	double sum6 = 0;
	double sum7 = 0;

	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < ARRAY_SIZE; i += 8) {
		sum += array[i] * 2;
		sum1 += array[i + 1] * 2;
		sum2 += array[i + 2] * 2;
		sum3 += array[i + 3] * 2;
		sum4 += array[i + 4] * 2;
		sum5 += array[i + 5] * 2;
		sum6 += array[i + 6] * 2;
		sum7 += array[i + 7] * 2;
	}
	sum = sum + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
	clock_gettime(CLOCK_MONOTONIC, &end);
	fprintf(fp, "%f\n", sum);
	fclose(fp);
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	return (time_usec);
}
void evaluate(double (*fp)(), double *array) {
	double min_time = 0;
	int i;
	for (i = 0; i < NUM_TRIALS; i++) {
		if (i == 0) {
			min_time = (*fp)(array);
			continue;
		}
		double this_time = (*fp)(array);
		if (this_time < min_time)
			min_time = this_time;
	}

	double time_sec = min_time / 1000000;
	double GBps = (ARRAY_SIZE * sizeof(double) / (1000000000 * time_sec));
	double gflops = ARRAY_SIZE * 2 / (1000000000 * time_sec);
	double floppb = 2.0 / sizeof(double);
	printf("Time: %.2f secs\n", time_sec);
	printf("Bw: %.2f GB/s\n", GBps);
	printf("FLOPS: %.2f GFLOP/s\n", gflops);
	printf("FLOP/byte: %.2f FLOP/byte\n", floppb);
	printf("%f\n", (float)(ARRAY_SIZE * sizeof(double)) / 1000000000.0);
}


double c2b(double *array) {
	FILE *fp = fopen("/dev/null", "w");
	struct timespec start;
	struct timespec end;

	int j;
	double sum;
	for (j = 0; j < 10; j++) {	
	int i = 0;
	sum = 0;
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < ARRAY_SIZE; i += 2) {
		sum += array[i] * 2;
		sum += array[i + 1] * 2;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	printf("%f\n", time_usec/1000000);
	}
	fprintf(fp, "%f\n", sum);
	fclose(fp);
	return (sum);
}
double c2c(double *array) {
	FILE *fp = fopen("/dev/null", "w");
	struct timespec start;
	struct timespec end;

	int j;
	double sum;
	for (j = 0; j < 10; j++) {	
	int i = 0;
	sum = 0;
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < ARRAY_SIZE; i += 4) {
		sum += array[i] * 2;
		sum += array[i + 1] * 2;
		sum += array[i + 2] * 2;
		sum += array[i + 3] * 2;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	printf("%f\n", time_usec/1000000);
	}
	fprintf(fp, "%f\n", sum);
	fclose(fp);
	return (sum);
}
double c2d(double *array) {
	FILE *fp = fopen("/dev/null", "w");
	struct timespec start;
	struct timespec end;

	int j;
	double sum;
	for (j = 0; j < 10; j++) {	
	int i = 0;
	sum = 0;
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < ARRAY_SIZE; i += 6) {
		sum += array[i] * 2;
		sum += array[i + 1] * 2;
		sum += array[i + 2] * 2;
		sum += array[i + 3] * 2;
		sum += array[i + 4] * 2;
		sum += array[i + 5] * 2;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	printf("%f\n", time_usec/1000000);
	}
	fprintf(fp, "%f\n", sum);
	fclose(fp);
	return (sum);
}

int main() {

	//Initialise array

	double array[ARRAY_SIZE];
	int i;
	for (i = 0; i < ARRAY_SIZE; i++) {
		array[i] = (double)i/3;
	}

	printf("*C1*\n");
	evaluate(c1, array);

	printf("\n\n");	

	printf("*C2*\n");
	evaluate(c2, array);

	printf("\n\n");	

	printf("*C3*\n");
	evaluate(c3, array);

	printf("C1***************************************************\n");

	c1a(array);
	printf("C2 2***************************************************\n");

	c2b(array);
	printf("C2 4***************************************************\n");
	c2c(array);
	printf("C2 6***************************************************\n");
	c2d(array);
	printf("C2 8***************************************************\n");
	c2a(array);
	return 0;

}
