#ifndef __ARITH_CUH__
#define __ARITH_CUH__

//includes
#ifdef __CUDACC__ //if running on device do this
	#define ALL __device__ 
#else //if running on host do this
	#define ALL 
#endif 


//Prototype
#ifdef __CUDACC__
ALL bool isPrime(unsigned long long n);
ALL unsigned long long gcd(unsigned long long, unsigned long long);
ALL unsigned long long lcm(unsigned long long, unsigned long long);
ALL unsigned long long L(unsigned long long, unsigned long long);
ALL unsigned long long power(unsigned long long, unsigned long long);
ALL unsigned long long powermod(unsigned long long, unsigned long long, unsigned long long);
ALL unsigned long long mulmod(unsigned long long, unsigned long long, unsigned long long);
ALL unsigned long long modInverse(unsigned long long, unsigned long long);
#else
ALL bool isPrime(unsigned long long n);
ALL unsigned long long gcd(unsigned long long, unsigned long long);
ALL unsigned long long lcm(unsigned long long, unsigned long long);
ALL unsigned long long L(unsigned long long, unsigned long long);
ALL unsigned long long power(unsigned long long, unsigned long long);
ALL unsigned long long powermod(unsigned long long, unsigned long long, unsigned long long);
ALL unsigned long long mulmod(unsigned long long, unsigned long long, unsigned long long);
ALL unsigned long long modInverse(unsigned long long, unsigned long long);
#endif

//functions total = 8

//check if it is prime
ALL bool isPrime(unsigned long long n) {
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
ALL unsigned long long lcm(unsigned long long a, unsigned long long b) {
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
ALL unsigned long long powermod(unsigned long long x, unsigned long long y, unsigned long long p)
{
	unsigned long long res = 1;      // Initialize result

	x = x % p;  // Update x if it is more than or 
				// equal to p

	while (y > 0)
	{
		// If y is odd, multiply x with result
		if (y & 1)
			res = mulmod(res, x, p);

		// y must be even now
		y = y >> 1; // y = y/2
		x = mulmod(x, x, p);
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
ALL unsigned long long modInverse(unsigned long long a, unsigned long long m) {
	unsigned long long m0 = m;
	long long int y = 0, x = 1;

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



#endif
