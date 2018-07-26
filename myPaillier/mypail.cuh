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
#include <time.h> //time()
#endif 

#define MAX 256 //2^8 used for random functions

/*Define Datatypes*/

//Datatype for public key containing n and g
typedef struct {
	unsigned* n = (unsigned*)malloc(sizeof(unsigned));
	unsigned* g = (unsigned*)malloc(sizeof(unsigned));
}pubkey;

//Datatype for public key containing lamda and mu
typedef struct {
	unsigned* lamda = (unsigned*)malloc(sizeof(unsigned));
	unsigned* mu = (unsigned*)malloc(sizeof(unsigned));
}prvkey;


/*Prototypes List*/
#ifdef __CUDACC__
ALL unsigned primeGen();
ALL unsigned setG(unsigned);
 void initial(pubkey*, prvkey*);
ALL void setup(pubkey*, prvkey*);
ALL unsigned rng(unsigned);
ALL bool isPrime(unsigned);
ALL unsigned gcd(unsigned, unsigned);
ALL unsigned lcm(unsigned, unsigned);
ALL unsigned L(unsigned, unsigned);
ALL unsigned power(unsigned, unsigned);
ALL unsigned power(unsigned, unsigned, unsigned);
ALL unsigned modInverse(unsigned, unsigned);
ALL bool gCheck(pubkey*, prvkey*);
#else
ALL unsigned primeGen();
ALL unsigned setG(unsigned);
void initial(pubkey*, prvkey*);
ALL void setup(pubkey*, prvkey*);
ALL unsigned rng(unsigned);
ALL bool isPrime(unsigned);
ALL unsigned gcd(unsigned, unsigned);
ALL unsigned lcm(unsigned, unsigned);
ALL unsigned L(unsigned, unsigned);
ALL unsigned power(unsigned, unsigned);
ALL unsigned power(unsigned, unsigned, unsigned);
ALL unsigned modInverse(unsigned, unsigned);
ALL bool gCheck(pubkey*, prvkey*);
#endif

/*Paillier Functions*/

//find the right prime number for p and q
ALL unsigned primeGen() {
	unsigned rand;
	do {
		rand = rng(MAX);
	} while (!isPrime(rand));
	return rand;
}

//Get the proper rand number for g
ALL unsigned setG(unsigned nsquare) {
	unsigned rand;
	do{
		rand = rng(nsquare);
	} while (gcd(nsquare, rand) != 1);
	return rand;
}

void initial(pubkey* pub, prvkey* prv) {

	pub->n = (unsigned)0;
	pub->g = (unsigned)0;
	prv->lamda = (unsigned)0;
	prv->mu = (unsigned)0;
}

//setup public key and private key
ALL void setup(pubkey* pub, prvkey* prv){
	#ifndef __CUDACC__ //when running on host code
		srand(time(NULL));//randomize the seed
		unsigned p = primeGen(); //generating prime number p
		std::cout << "p: " << p << std::endl;

		unsigned q = primeGen(); //generating prime number q
		std::cout << "q: " << q << std::endl;

		*pub->n = p * q;
		std::cout << "n: " << *pub->n << std::endl;

		*pub->g = setG((*pub->n)*(*pub->n)); //rand num from a set without the factors of n^2
		std::cout << "g: " << *pub->g << std::endl;

		*prv->lamda = lcm(p-1,q-1); //lamda = lcm(p-1,q-1)
		std::cout << "lamda: " << *prv->lamda << std::endl;

		unsigned l = L(power(*pub->g, *prv->lamda , (*pub->n)*(*pub->n)), *pub->n); //L(g^lamda mod n^2)
		*prv->mu = modInverse(l, *pub->n) ;
		std::cout << "mu: " << *prv->mu << std::endl;

		bool flag = gCheck(pub, prv);
		if (flag)
			std::cout << "Is g correct: yes" << std::endl;
		else
			std::cout << "Is g correct: no" << std::endl;
	#else //when running on device code
		
	//something wrong with using variables from the structure !!!!!!!!!!!
		unsigned n, g, lamda, mu = (unsigned)0;
		pub->n = &n;
		pub->g = &g;
		prv->lamda = &lamda;
		prv->mu = &mu;
		


		unsigned p = primeGen(); //generating prime number p
		printf("p: %d \n", p);

		unsigned q = primeGen(); //generating prime number q
		printf("q: %d \n", q);

		printf("p*q: %u \n", (p*q));

		*pub->n = p * q;
		printf("n: %d \n",*pub->n);

		*pub->g = setG((*pub->n)*(*pub->n)); //rand num from a set without the factors of n^2
		printf("g: %u \n", *pub->g);

		*prv->lamda = lcm(p - 1, q - 1); //lamda = lcm(p-1,q-1)
		printf("lamda: %d \n", *prv->lamda);

		unsigned l = L(power(*pub->g, *prv->lamda, (*pub->n)*(*pub->n)), *pub->n); //L(g^lamda mod n^2)
		*prv->mu = modInverse(l, *pub->n);
		printf("mu: %d \n", *prv->mu);

		bool flag = gCheck(pub, prv);
		if (flag)
			printf("Is g correct: yes \n");
		else
			printf("Is g correct: no \n");
	#endif
}


/*Arithmetic Functions*/

//generates a random number betweeen 0 to 2^8
ALL unsigned rng(unsigned max){
	#ifndef __CUDACC__ //when running on host code
		//srand(time(NULL));
		unsigned num = rand() % max + 1;
	#else //when running on device code
		curandState_t state;
		curand_init(1234567890, 0, 0, &state);
		unsigned num = curand(&state) % max;
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

//power function:  return = base^exp
ALL unsigned power(unsigned base, unsigned exp = 0) {
	if (exp <= 0)
		return 1;
	return base * power(base, exp - 1);
}

ALL unsigned power(unsigned x, unsigned y, unsigned p)
{
	unsigned res = 1;      // Initialize result

	x = x % p;  // Update x if it is more than or 
				// equal to p

	while (y > 0)
	{
		// If y is odd, multiply x with result
		if (y & 1)
			res = (res*x) % p;

		// y must be even now
		y = y >> 1; // y = y/2
		x = (x*x) % p;
	}
	return res;
}

//Using extended Euclid algorithm to calculate the modulo inverse of ax = 1 (mod m)
ALL unsigned modInverse(unsigned a, unsigned m){
	unsigned m0 = m;
	unsigned y = 0, x = 1;

	if (m == 1)
		return 0;

	while (a > 1)
	{
		// q is quotient
		unsigned q = a / m;
		unsigned t = m;

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

ALL bool gCheck(pubkey* pub, prvkey* prv){
	if (gcd(L(power(*pub->g, *prv->lamda,((*pub->n)*(*pub->n))), *pub->n), *pub->n) == 1)
		return true;
	else
		return false;
}



#endif
