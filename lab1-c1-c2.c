#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 2000000

float c1(int *array);
void c2();
void printResults(char *name, int flops, float duration);


float c1(int *array) {

	struct timespec start;
	struct timespec end;
	
	int i;
	double sum = 0;
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (i = 0; i < ARRAY_SIZE; i++) {
		sum += array[i] * 2;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);

	float seconds = end.tv_sec - start.tv_sec;
	long nano_seconds = end.tv_nsec - start.tv_nsec;
	printf("%ld\n", nano_seconds);

	return (seconds + nano_seconds / 1000000000);
}

int main() {

	//Initialise array

	int array[ARRAY_SIZE];
	int i;
	for (i = 0; i < ARRAY_SIZE; i++) {
		array[i] = (double)i/3;
	}

	float t = c1(array);
	printf("%f\n", t);

	return 0;

}
