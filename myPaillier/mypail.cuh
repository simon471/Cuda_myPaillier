#ifndef __MYPAIL_CUH__
#define __MYPAIL_CUH__



#ifdef __CUDACC__ //if running on device do this
#define ALL __device__ 
#include <curand.h> //used for generating random number on device code
#include <curand_kernel.h>
#include <stdio.h>

#else //if running on host do this
#define ALL 
#include <iostream>
#include <stdlib.h> //srand(), rand()
#endif 

#define MAX 65536 //2^16 used for random functions

/*Define Datatypes*/

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


/*Prototypes List*/
#ifdef __CUDACC__
ALL unsigned long long primeGen();
ALL unsigned long long getRandZ(unsigned long long);
ALL void setup(pubkey, prvkey);
ALL unsigned long long enc(pubkey, unsigned long long);
ALL unsigned long long dec(pubkey, prvkey, unsigned long long);
ALL unsigned long long rng(unsigned long long);
ALL bool isPrime(unsigned long long);
ALL unsigned long long gcd(unsigned long long, unsigned long long);
ALL unsigned long long lcm(unsigned long long, unsigned long long);
ALL unsigned long long L(unsigned long long, unsigned long long);
ALL unsigned long long power(unsigned long long, unsigned long long);
ALL unsigned long long power(unsigned long long, unsigned long long, unsigned long long);
ALL unsigned long long modInverse(unsigned long long, unsigned long long);
ALL unsigned long long mulmod(unsigned long long, unsigned long long, unsigned long long);
ALL bool gCheck(pubkey, prvkey);
#else
ALL unsigned long long primeGen();
ALL unsigned long long getRandZ(unsigned long long);
ALL void setup(pubkey, prvkey);
ALL unsigned long long enc(pubkey, unsigned long long);
ALL unsigned long long dec(pubkey, prvkey, unsigned long long);
ALL unsigned long long rng(unsigned long long);
ALL bool isPrime(unsigned long long);
ALL unsigned long long gcd(unsigned long long, unsigned long long);
ALL unsigned long long lcm(unsigned long long, unsigned long long);
ALL unsigned long long L(unsigned long long, unsigned long long);
ALL unsigned long long power(unsigned long long, unsigned long long);
ALL unsigned long long power(unsigned long long, unsigned long long, unsigned long long);
ALL unsigned long long modInverse(unsigned long long, unsigned long long);
ALL unsigned long long mulmod(unsigned long long, unsigned long long, unsigned long long);
ALL bool gCheck(pubkey, prvkey);
#endif

/*Paillier Functions*/

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
	do{
		rand = rng(x);
	} while (gcd(x, rand) != 1 && rand < x);
	return rand;
}

//setup public key and private key
ALL void setup(pubkey pub, prvkey prv){
	#ifndef __CUDACC__ //when running on host code
		srand(time(NULL));//randomize the seed
		unsigned long long p =  primeGen(); //generating prime number p
		unsigned long long q =  primeGen(); //generating prime number q
		*pub.n = p * q;
		*pub.g = getRandZ((*pub.n)*(*pub.n)); //rand num from a set without the factors of n^2
		*prv.lamda = lcm(p-1,q-1); //lamda = lcm(p-1,q-1)
		unsigned long long l = L(power(*pub.g, *prv.lamda , (*pub.n)*(*pub.n)), *pub.n); //L(g^lamda mod n^2)
		*prv.mu = modInverse(l, *pub.n) ;
	#else //running on device code
		unsigned long long p = primeGen(); //generating prime number p
		unsigned long long q = primeGen(); //generating prime number q
		*pub.n = p * q;
		*pub.g = getRandZ((*pub.n)*(*pub.n)); //rand num from a set without the factors of n^2
		*prv.lamda = lcm(p - 1, q - 1); //lamda = lcm(p-1,q-1)
		unsigned long long l = L(power(*pub.g, *prv.lamda, (*pub.n)*(*pub.n)), *pub.n); //L(g^lamda mod n^2)
		*prv.mu = modInverse(l, *pub.n);
	#endif

}

//encryption function
ALL unsigned long long enc(pubkey pub, unsigned long long m) {
	unsigned long long n2 = (*pub.n)*(*pub.n);
	unsigned long long r = getRandZ(*pub.n);

	//check m
	if (m < *pub.n) {
		unsigned long long gm = power(*pub.g, m, n2);
		unsigned long long rn = power(r, *pub.n, n2);
		unsigned long long c = mulmod(gm,rn,n2);
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
ALL unsigned long long dec(pubkey pub,prvkey prv, unsigned long long c) {
	unsigned long long p1 = power(c, *prv.lamda, (*pub.n)*(*pub.n));
	unsigned long long l = L(p1, *pub.n);
	unsigned long long m = mulmod(l, *prv.mu, *pub.n);
	return m;
}


/*Arithmetic Functions*/

//generates a random number betweeen 0 to 2^8
ALL unsigned long long rng(unsigned long long max){
	#ifndef __CUDACC__ //when running on host code
		//srand(seed);
		unsigned long long num = rand() % max + 1;
	#else //when running on device code
		curandState_t state;
		curand_init((unsigned long long)clock(), 0, 0, &state);
		unsigned long long num = curand(&state) % max;
	#endif
		return num;
}

//check if the number is a prime number
ALL bool isPrime(unsigned long long n){
	bool isPrime = true;
	if (n < 2) {
		isPrime = false;
	}
	else {
		for (int i = 2; i < n / 2; i++) {
			if (n%i == 0) {
				isPrime = false;
				break;
			}
		}
	}
	return isPrime;
}

//find the gcd(greatest common divisor) of two numbers
ALL unsigned long long gcd(unsigned long long a, unsigned long long b) {
	for (;;)
	{
		if (a == 0) return b;
		b %= a;
		if (b == 0) return a;
		a %= b;
	}
}

//find the lcm(least common multiple) of two numbers
ALL unsigned long long lcm(unsigned long long a, unsigned long long b){
	return a * b / gcd(a, b);
}

//function L(x)=(x-1)/n
ALL unsigned long long L(unsigned long long x, unsigned long long n) {
	return (x - 1) / n;
}

//power function:  return = base^exp
ALL unsigned long long power(unsigned long long base, unsigned long long exp = 0) {
	if (exp <= 0)
		return 1;
	return base * power(base, exp - 1);
}

//powermod function: return = x^y mod p
ALL unsigned long long power(unsigned long long x, unsigned long long y, unsigned long long p)
{
	unsigned long long res = 1;      // Initialize result

	x = x % p;  // Update x if it is more than or 
				// equal to p

	while (y > 0)
	{
		// If y is odd, multiply x with result
		if (y & 1)
			res = mulmod(res,x,p);

		// y must be even now
		y = y >> 1; // y = y/2
		x = mulmod(x,x,p);
	}
	return res;
}

//multiply mod function: a*b mod m, used to prevent overflow in power function
ALL unsigned long long mulmod(unsigned long long a, unsigned long long b, unsigned long long mod)
{
	unsigned long long res = 0; // Initialize result
	a = a % mod;
	while (b > 0)
	{
		// If b is odd, add 'a' to result
		if (b % 2 == 1)
			res = (res + a) % mod;

		// Multiply 'a' with 2
		a = (a * 2) % mod;

		// Divide b by 2
		b /= 2;
	}

	// Return result
	return res % mod;
}

//Using extended Euclid algorithm to calculate the modulo inverse of ax = 1 (mod m)
ALL unsigned long long modInverse(unsigned long long a, unsigned long long m){
	unsigned long long m0 = m;
	int y = 0, x = 1;

	if (m == 1)
		return 0;

	while (a > 1)
	{
		// q is quotient
		unsigned long long q = a / m;
		unsigned long long t = m;

		// m is remainder now, process same as
		// Euclid's algo
		m = a % m, a = t;
		t = y;

		// Update y and x
		y = x - q * y;
		x = t;
	}

	 //Make x positive
	if (x < 0)
		x += m0;

	return x;
}

//check if g is valid
ALL bool gCheck(pubkey pub, prvkey prv){
	if (gcd(L(power(*pub.g, *prv.lamda,((*pub.n)*(*pub.n))), *pub.n), *pub.n) == 1)
		return true;
	else
		return false;
}



#endif
