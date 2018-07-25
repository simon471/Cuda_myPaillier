#ifndef __MYPAIL_CUH__
#define __MYPAIL_CUH__

#ifdef __CUDACC__ //if running on device do this
#define ALL __host__ __device__ inline
#include <curand.h>
#include <curand_kernel.h>
#else //if running on host do this
#define ALL 
#include <stdlib.h> //srand(), rand()
#include <time.h> //time()
#endif 

#define MAX 32768 //2^15 used for random functions

/*Prototypes List*/
ALL unsigned primeGen();
ALL unsigned rng();
ALL bool isPrime(unsigned);

/*Define Datatypes*/
typedef struct {
	unsigned n;
	unsigned g;
}pubkey;

typedef struct {
	unsigned lamda;
	unsigned mu;
}prvkey;

/*Paillier Functions*/

//find the right prime number for p and q
ALL unsigned primeGen() {
	unsigned rand;
	do {
		rand = rng();
	} while (!isPrime(rand));
	return rand;
}


//setup public key and private key
ALL void setup(pubkey pub, prvkey prv){
	#ifndef __CUDACC__ //when running on host code
		
	#else //when running on device code
	#endif
}



/*Arithmetic Functions*/

//generates a random number betweeen 0 to 2^15
ALL unsigned rng(){
	#ifndef __CUDACC__ //when running on host code
		srand(time(NULL));
		unsigned num = rand() % MAX + 1;
	#else //when running on device code
		curandState_t state;
		curand_init(time(NULL), 0, 0, &state);
		unsigned num = curand(&state) % MAX;
	#endif
		return num;
}

//check if the number is a prime number
ALL bool isPrime(unsigned n){
	bool isPrime = true;
	for (int i = 2; i < n / 2; i++) {
		if (n%i == 0) {
			isPrime = false;
			break;
		}
	}
	return isPrime;
}

//find the gcd(greatest common divisor) of two numbers
ALL unsigned gcd(unsigned a, unsigned b) {
	for (;;)
	{
		if (a == 0) return b;
		b %= a;
		if (b == 0) return a;
		a %= b;
	}
}

//find the lcm(least common multiple) of two numbers
ALL unsigned lcm(unsigned a, unsigned b){
	return a * b / gcd(a, b);
}



//function L(x)=(x-1)/n
ALL unsigned L(unsigned x, unsigned n) {
	return (x - 1) / n;
}




#endif
