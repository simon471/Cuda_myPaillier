#include <iostream>
#include <time.h> //time()
#include "mypail.cuh"
using namespace std;

//testing out functions
int main() {
	
	cout << "program starting..." << endl;

	cout << "Declaring variables..." << endl;
	pubkey publickey;
	prvkey privatekey;
	cout << "Variables declared..." << endl;

	cout << "Setting up key values..." << endl;
	time_t seed;
	time(&seed);

	setup(publickey, privatekey,(unsigned long)seed);
	cout << "Key generation done..." << endl;

	cout << "program ending..." << endl;

}