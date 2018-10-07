#include <stdio.h>
#include <stdlib.h>
#include <time.h>
 
#define ARRAY_SIZE 134217728 * 5 // 5GiB
#define NUM_TRIALS 10

double c1(double *array);
double c2(double *array);
double c3(double *array);
double c4(double *array);
void evaluate(double (*fp)(), double *array);
void printResults(char *name, int flops, float duration);

// Calculate 2*(sum of array)
double c1(double *array) {
	struct timespec start;
	struct timespec end;
	
	int i;
	double sum = 0;
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < ARRAY_SIZE; i++) {
		sum += array[i] * 2;
	}

	// Write to file to avoid compiler optimisation
	FILE *fp = fopen("/dev/null", "w");
	clock_gettime(CLOCK_MONOTONIC, &end);
	fprintf(fp, "%f\n", sum);
	fclose(fp);
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	return (time_usec);
}

// Unroll the loop
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
	FILE *fp = fopen("/dev/null", "w");
	fprintf(fp, "%f\n", sum);
	fclose(fp);
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	return (time_usec);
}

// Test: Break the dependency chain, allow 2 loads per cycle
double c3(double *array) {
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
	FILE *fp = fopen("/dev/null", "w");
	fprintf(fp, "%f\n", sum);
	fclose(fp);
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	return (time_usec);
}

// Test: Break the dependency chain, allow 2 loads per cycle
double c4(double *array) {
	struct timespec start;
	struct timespec end;
	
	int i;
	double sum = 0;

	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < ARRAY_SIZE; i += 2) {
		sum += array[i] * 2 + array[i + 1] * 2;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	FILE *fp = fopen("/dev/null", "w");
	fprintf(fp, "%f\n", sum);
	fclose(fp);
	double time_usec=(((double)end.tv_sec *1000000 + (double)end.tv_nsec/1000) - ((double)start.tv_sec *1000000 + (double)start.tv_nsec/1000));
	return (time_usec);
}

// Run fp NUM_TRIALS times, record lowest exec time, bw, flops.
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
	printf("Time: %.2f secs\n", time_sec);
	printf("Bw: %.2f GB/s\n", GBps);
	printf("FLOPS: %.2f GFLOP/s\n", gflops);
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

	printf("\n");

	printf("*C2*\n");
	evaluate(c2, array);

	printf("\n");

	/*printf("*C3* (Test)\n");
	evaluate(c3, array);

	printf("\n");

	printf("*C4* (Test)\n");
	evaluate(c4, array);
	printf("\n");*/
	return 0;

}
