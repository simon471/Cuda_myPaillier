#ifndef __HYPERNYMS_CUH__
#define __HYPERNYMS_CUH__

//includes
#ifdef __CUDACC__ //if running on device do this
#define ALL __device__
#include <stdio.h>
#include "keygen.cuh"
#include "arith.cuh"
#else //if running on host do this
#define ALL 
#include <iostream>
#include "keygen.cuh"
#include "arith.cuh"

#endif 

//Prototype
#ifdef __CUDACC__
ALL unsigned long long enc(pubkey, unsigned long long);
ALL unsigned long long dec(pubkey, prvkey, unsigned long long);
#else
ALL unsigned long long enc(pubkey, unsigned long long);
ALL unsigned long long dec(pubkey, prvkey, unsigned long long);
#endif

//functions total = 2

//encryption function
ALL unsigned long long enc(pubkey pub, unsigned long long m) {
	unsigned long long n2 = (*pub.n)*(*pub.n);
	unsigned long long r = getRandZ(*pub.n);

	//check m
	if (m < *pub.n) {
		unsigned long long gm = powermod(*pub.g, m, n2);
		unsigned long long rn = powermod(r, *pub.n, n2);
		unsigned long long c = mulmod(gm, rn, n2);
		return c;
	}
	else {
#ifndef __CUDACC__ //when running on host code
		std::cout << "m is larger than n" << std::endl;
#else
		printf("m is larger than n \n");
#endif
	}
}

//decryption function
ALL unsigned long long dec(pubkey pub, prvkey prv, unsigned long long c) {
	unsigned long long p1 = powermod(c, *prv.lamda, (*pub.n)*(*pub.n));
	unsigned long long l = L(p1, *pub.n);
	unsigned long long m = mulmod(l, *prv.mu, *pub.n);
	return m;
}


#endif
