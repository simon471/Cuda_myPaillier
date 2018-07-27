#include <iostream>
#include "mypail.cuh"
using namespace std;

//testing out functions
int main() {
	
	cout << "program starting..." << endl;

	cout << "Declaring variables..." << endl;
	pubkey publickey;
	prvkey privatekey;
	cout << "Variables declared..." << endl;

	cout << "Initializing keys..." << endl;
	initial(&publickey, &privatekey);
	cout << "Keys initialized..." << endl;

	cout << "Setting up key values..." << endl;
	setup(&publickey, &privatekey);
	cout << "Key generation done..." << endl;

	cout << "program ending..." << endl;

}