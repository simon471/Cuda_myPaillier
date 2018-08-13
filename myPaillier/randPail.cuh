#ifndef __RANDPAIL_CUH__
#define __RANDPAIL_CUH__

//includes
#ifdef __CUDACC__ //if running on device do this
#define ALL __device__ 
#include "arith.cuh"
#include <random>
#include <limits>
//#include <time.h> 
#include <curand.h> //used for generating random number on device code
#include <curand_kernel.h>
#else //if running on host do this
#define ALL 
#include "arith.cuh"
#include <random>
#include <limits>
#endif 

#define MAX 65536 //2^16 used for random functions

//Prototype
#ifdef __CUDACC__
ALL unsigned long long primeGen();
ALL unsigned long long getRandZ(unsigned long long);
ALL unsigned long long rng(unsigned long long);
unsigned long long rnghost(unsigned long long);
#else
ALL unsigned long long primeGen();
ALL unsigned long long getRandZ(unsigned long long);
ALL unsigned long long rng(unsigned long long);
unsigned long long rnghost(unsigned long long);
#endif

//functions total = 4

//find the right prime number for p and q
ALL unsigned long long primeGen() {
	unsigned long long rand;
	do {
		rand = rng(MAX);
	} while (!isPrime(rand));
	return rand;
}


//Get the proper rand number between z*_x
ALL unsigned long long getRandZ(unsigned long long x) {
	unsigned long long rand;
	do {
		rand = rng(x);
	} while (gcd(x, rand) != 1 && rand < x);
	return rand;
}

//generates a random number betweeen 0 to max
ALL unsigned long long rng(unsigned long long max) {
#ifndef __CUDACC__ //when running on host code
	std::random_device rd;
	std::mt19937_64 eng(rd());
	std::uniform_int_distribution<unsigned long long> distr;
	unsigned long long num = distr(eng) % max + 1;
#else //when running on device code
	curandState_t state;
	curand_init((unsigned long long)clock(), 0, 0, &state);
	unsigned long long num = curand(&state) % max;
#endif
	return num;
}

unsigned long long rnghost(unsigned long long x) {
	std::random_device rd;
	std::mt19937_64 eng(rd());
	std::uniform_int_distribution<unsigned long long> distr;
	unsigned long long num = distr(eng) % x + 1;
	return num;
}


#endif
