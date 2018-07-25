#include <iostream>
#include "mypail.cuh"
using namespace std;

//testing out functions
int main() {
	
	cout << "program starting..." << endl;

	pubkey publickey;
	prvkey privatekey;
	cout << "Variables declared..." << endl;

	setup(publickey, privatekey);


	cout << "program ending..." << endl;

}