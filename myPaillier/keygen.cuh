#ifndef __KEYGEN_CUH__
#define __KEYGEN_CUH__

//includes
#ifdef __CUDACC__ //if running on device do this
#define ALL __device__ 
#include "arith.cuh"
#include "randPail.cuh"

#else //if running on host do this
#define ALL 
#include "arith.cuh"
#include "randPail.cuh"
#include <random>

#endif 

//Datatype for public key containing n and g
typedef struct {
	unsigned long long* n = (unsigned long long*)malloc(sizeof(unsigned long long));
	unsigned long long* g = (unsigned long long*)malloc(sizeof(unsigned long long));
}pubkey;

//Datatype for public key containing lamda and mu
typedef struct {
	unsigned long long* lamda = (unsigned long long*)malloc(sizeof(unsigned long long));
	unsigned long long* mu = (unsigned long long*)malloc(sizeof(unsigned long long));
}prvkey;

//Prototype
#ifdef __CUDACC__
ALL void setup(pubkey, prvkey);
#else
ALL void setup(pubkey, prvkey);
#endif

//functions total = 1

//setup public key and private key
ALL void setup(pubkey pub, prvkey prv) {
#ifndef __CUDACC__ //when running on host code
	unsigned long long p;
	unsigned long long q;
	do {
		p = primeGen(); //generating prime number p
		q = primeGen(); //generating prime number q
		*pub.n = p * q;
	} while (*pub.n < 100000000);
	*pub.g = getRandZ((*pub.n)*(*pub.n)); //rand num from a set without the factors of n^2
	*prv.lamda = lcm(p - 1, q - 1); //lamda = lcm(p-1,q-1)
	unsigned long long l = L(powermod(*pub.g, *prv.lamda, (*pub.n)*(*pub.n)), *pub.n); //L(g^lamda mod n^2)
	*prv.mu = modInverse(l, *pub.n);
#else //running on device code
	unsigned long long p;
	unsigned long long q;
	do {
		p = primeGen(); //generating prime number p
		q = primeGen(); //generating prime number q
		*pub.n = p * q;
	} while (*pub.n < 100000000);
	*pub.g = getRandZ((*pub.n)*(*pub.n)); //rand num from a set without the factors of n^2
	*prv.lamda = lcm(p - 1, q - 1); //lamda = lcm(p-1,q-1)
	unsigned long long l = L(powermod(*pub.g, *prv.lamda, (*pub.n)*(*pub.n)), *pub.n); //L(g^lamda mod n^2)
	*prv.mu = modInverse(l, *pub.n);
#endif

}


#endif
