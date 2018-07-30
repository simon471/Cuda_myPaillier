#include <iostream>
#include <time.h> //time()
#include "mypail.cuh"
using namespace std;

//testing out functions
int main() {
	
	cout << "program starting..." << endl<<endl;

	cout << "Declaring variables..." << endl;
	pubkey publickey;
	prvkey privatekey;
	cout << "Variables declared..." << endl;

	cout << "Setting up key values..." << endl;

	setup(publickey, privatekey);
	cout << "Key generation done..." << endl<<endl;

	cout << "Starting encryption and decryption process..." << endl;
	unsigned m = 123;
	cout << "Message: "<< m << endl;
	unsigned c = enc(publickey,m);
	cout << "Cypher: "<< c << endl;
	unsigned m2 = dec(publickey, privatekey, c);
	cout << "Decrypted cypher: "<< m2 << endl;
	cout << "Encryption and decryption process done..." << endl<<endl;

	cout << UINT_MAX<<endl;


	cout << "program ending..." << endl;

}