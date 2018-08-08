#include <iostream>
#include <time.h> //time()
#include "mypail.cuh"
#include <string>
#include <ctime>
#include <chrono>
using namespace std;

//testing out functions
int main() {
	
	

	cout << "program starting..." << endl<<endl;

	cout << ">> Declaring variables..." << endl;
	pubkey publickey;
	prvkey privatekey;
	int n = 100000;
	string messageFile = "message.txt";
	string cipherFile = "cipher.txt";
	string message2File = "message2.txt";
	unsigned long long* message = new unsigned long long[n];
	unsigned long long* cipher = new unsigned long long[n];
	unsigned long long* message2 = new unsigned long long[n];
	cout << "Variables declared..." << endl<<endl;

	cout << ">> Setting up key values..." << endl;

	setup(publickey, privatekey);
	cout << "Key generation done..." << endl<<endl;

	cout << ">> generating messages file..." << endl;
	genRandFile(publickey,messageFile, n); //generate random number in a file
	cout << "messages generated..." << endl<<endl;

	cout << ">> copying message from file to array..." << endl;
	readFile(message, messageFile, n); //copying message from file to array
	cout << "messages copied..." << endl << endl;
	
	cout << ">> Starting encryption and decryption process..." << endl<<endl;

	//timer
	auto begin = chrono::steady_clock::now();

	//cout << ">> Encrypting message array..." << endl;
	for (int i = 0; i < n; i++) {
		*(cipher + i) = enc(publickey, *(message + i));
	}

	auto end = chrono::steady_clock::now();


	cout << "Message encrypted..." << endl << endl;

	cout << ">> copying cipher array to file..." << endl;
	createFile(cipherFile);
	writeFile(cipher, cipherFile, n); 
	cout << "cipher copied..." << endl << endl;


	cout << ">> Decrypting cipher..." << endl;
	for (int i = 0; i < n; i++) {
		*(message2 + i) = dec(publickey,privatekey, *(cipher + i));
	}
	cout << "cipher decrypted..." << endl << endl;

	cout << ">> copying decrypted message array to file..." << endl;
	createFile(message2File);
	writeFile(message2, message2File, n);
	cout << "decrypted message copied..." << endl << endl;

	/*
	//testing the encryption and decryption process
	unsigned long long m = 123;
	cout << "Message: "<< m << endl;
	unsigned long long c = enc(publickey,m);
	cout << "Cypher: "<< c << endl;
	unsigned long long m2 = dec(publickey, privatekey, c);
	cout << "Decrypted cypher: "<< m2 << endl;
	*/
	cout << ">> Encryption and decryption process done..." << endl<<endl;

	cout << ">> Freeing memory of the arrays..." << endl;
	delete[] message;
	delete[] cipher;
	delete[] message2;
	cout << "memory freed..." << endl << endl;

	//Printing time results
	cout << "Elapsed time in nanoseconds: " << chrono::duration_cast<chrono::nanoseconds> (end - begin).count() << " ns" << endl;
	cout << "Elapsed time in microseconds: " << chrono::duration_cast<chrono::microseconds> (end - begin).count() << " mus" << endl;
	cout << "Elapsed time in milliseconds: " << chrono::duration_cast<chrono::milliseconds> (end - begin).count() << " ms" << endl;
	cout << "Elapsed time in seconds: " << chrono::duration_cast<chrono::seconds> (end - begin).count() << " s" << endl;

	cout << "program ending..." << endl<<endl;

}