//paillier library
#include "fileio.cuh"
//time record
#include <chrono>
using namespace std;

int main(){
	cout << ">> This file is going to use host code to generate public and private key to a file." << endl;
	cout << ">> Program starting..." << endl << endl;

	//Declaring variables
	cout << ">> Declaring variables..." << endl;
	//public key and private key
	pubkey pub;
	prvkey prv;
	string pubkeyFile = "pubkey.txt";
	string prvkeyFile = "prvkey.txt";
	cout << "Done..." << endl << endl;

	//setting up pub and prv key
	cout << ">> Setting up public and private key..." << endl;
	auto begin = chrono::steady_clock::now();
	setup(pub, prv);
	auto end = chrono::steady_clock::now();
	cout << "Done..." << endl << endl;

	//create file
	cout << ">> Creating public and private key file..." << endl;
	createFile(pubkeyFile);
	createFile(prvkeyFile);
	cout << "Done..." << endl << endl;

	//write key to file
	cout << ">> Writing key into file..." << endl;
	keyWriteFile(pub, pubkeyFile);
	keyWriteFile(prv, prvkeyFile);
	cout << "Done..." << endl << endl;

	//printing key values
	cout << "Printing key values: " << endl;
	cout << "n: " << *pub.n << endl;
	cout << "g: " << *pub.g << endl;
	cout << "lamda: " << *prv.lamda << endl;
	cout << "mu: " << *prv.mu << endl;
	cout << "Done..." << endl << endl;

	//printing time
	cout << "Printing key generation time: " << endl;
	cout << "Elapsed time in nanoseconds: " << chrono::duration_cast<chrono::nanoseconds> (end - begin).count() << " ns" << endl;
	cout << "Elapsed time in microseconds: " << chrono::duration_cast<chrono::microseconds> (end - begin).count() << " mus" << endl;
	cout << "Elapsed time in milliseconds: " << chrono::duration_cast<chrono::milliseconds> (end - begin).count() << " ms" << endl;
	cout << "Elapsed time in seconds: " << chrono::duration_cast<chrono::seconds> (end - begin).count() << " s" << endl << endl;
	cout << "Done..." << endl << endl;

	cout << "Program ending..." << endl << endl;
}