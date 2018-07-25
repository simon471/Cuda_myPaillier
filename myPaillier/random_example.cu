
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#define MAX 100

/* this GPU kernel function calculates a random number and stores it in the parameter */
__global__ void random(unsigned int seed, int* result) {
	/* CUDA's random number library uses curandState_t to keep track of the seed value
	we will store a random state for every thread  */
	curandState_t state;

	/* we have to initialize the state */
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

	/* curand works like rand - except that it takes a state as a parameter */
	*result = curand(&state) % MAX;
}

int main() {
	/* allocate an int on the GPU */
	int* gpu_x;
	cudaMalloc((void**)&gpu_x, sizeof(int));

	/* invoke the GPU to initialize all of the random states */
	random <<<1, 1 >> >(time(NULL), gpu_x);

	/* copy the random number back */
	int x;
	cudaMemcpy(&x, gpu_x, sizeof(int), cudaMemcpyDeviceToHost);

	printf("Random number = %d.\n", x);

	/* free the memory we allocated */
	cudaFree(gpu_x);

	return 0;
}