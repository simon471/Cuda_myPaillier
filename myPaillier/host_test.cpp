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

	//cout << ">> Declaring variables..." << endl;
	//pubkey publickey;
	//prvkey privatekey;
	//int n = 100;
	//string messageFile = "message.txt";
	//string cipherFile = "cipher.txt";
	//string message2File = "message2.txt";
	//unsigned long long* message = new unsigned long long[n];
	//unsigned long long* cipher = new unsigned long long[n];
	//unsigned long long* message2 = new unsigned long long[n];
	//cout << "Variables declared..." << endl<<endl;

	//cout << ">> Setting up key values..." << endl;

	//setup(publickey, privatekey);
	//cout << "Key generation done..." << endl<<endl;

	//cout << ">> generating messages file..." << endl;
	//genRandFile(publickey,messageFile, n); //generate random number in a file
	//cout << "messages generated..." << endl<<endl;

	//cout << ">> copying message from file to array..." << endl;
	//readFile(message, messageFile, n); //copying message from file to array
	//cout << "messages copied..." << endl << endl;
	//
	//cout << ">> Starting encryption and decryption process..." << endl<<endl;

	////timer
	//auto begin = chrono::steady_clock::now();

	////cout << ">> Encrypting message array..." << endl;
	//for (int i = 0; i < n; i++) {
	//	*(cipher + i) = enc(publickey, *(message + i));
	//}

	//auto end = chrono::steady_clock::now();


	//cout << "Message encrypted..." << endl << endl;

	//cout << ">> copying cipher array to file..." << endl;
	//createFile(cipherFile);
	//writeFile(cipher, cipherFile, n); 
	//cout << "cipher copied..." << endl << endl;


	//cout << ">> Decrypting cipher..." << endl;
	//for (int i = 0; i < n; i++) {
	//	*(message2 + i) = dec(publickey,privatekey, *(cipher + i));
	//}
	//cout << "cipher decrypted..." << endl << endl;

	//cout << ">> copying decrypted message array to file..." << endl;
	//createFile(message2File);
	//writeFile(message2, message2File, n);
	//cout << "decrypted message copied..." << endl << endl;

	///*
	////testing the encryption and decryption process
	//unsigned long long m = 123;
	//cout << "Message: "<< m << endl;
	//unsigned long long c = enc(publickey,m);
	//cout << "Cypher: "<< c << endl;
	//unsigned long long m2 = dec(publickey, privatekey, c);
	//cout << "Decrypted cypher: "<< m2 << endl;
	//*/
	//cout << ">> Encryption and decryption process done..." << endl<<endl;

	//cout << ">> Freeing memory of the arrays..." << endl;
	//delete[] message;
	//delete[] cipher;
	//delete[] message2;
	//cout << "memory freed..." << endl << endl;

	////Printing time results
	//cout << "Elapsed time in nanoseconds: " << chrono::duration_cast<chrono::nanoseconds> (end - begin).count() << " ns" << endl;
	//cout << "Elapsed time in microseconds: " << chrono::duration_cast<chrono::microseconds> (end - begin).count() << " mus" << endl;
	//cout << "Elapsed time in milliseconds: " << chrono::duration_cast<chrono::milliseconds> (end - begin).count() << " ms" << endl;
	//cout << "Elapsed time in seconds: " << chrono::duration_cast<chrono::seconds> (end - begin).count() << " s" << endl;

	//cout << "n: " << *publickey.n << endl;
	//int count = 0;
	//for (int i = 0; i < 10000; i++) {
	//	if (rnghost(*publickey.n) >*publickey.n/2) {
	//		count++;
	//	}
	//}
	//std::cout << "count: " << count << std::endl;


	printf("Starting host process...\n\n");

	cout << ">> Declaring variables..." << endl;
	int n = 10000;
	pubkey pub;
	prvkey prv;
	unsigned long long* m1 = new unsigned long long[n];
	unsigned long long* m2 = new unsigned long long[n];
	unsigned long long* c1 = new unsigned long long[n];
	unsigned long long* c2 = new unsigned long long[n];
	unsigned long long* cres = new unsigned long long[n];
	unsigned long long* res = new unsigned long long[n];
	std::string message1File = "message1.txt";
	std::string message2File = "message2.txt";
	std::string cipher1File = "host_cipher1.txt";
	std::string cipher2File = "host_cipher2.txt";
	std::string cresultFile = "host_cresult.txt";
	std::string resultFile = "host_result.txt";
	std::string pubkeyFile = "pubkey.txt";
	std::string prvkeyFile = "prvkey.txt";
	std::cout << "Variables declared..." << std::endl << std::endl;


	cout << ">> Setting up key values..." << endl;
	keyReadFile(pub, pubkeyFile);
	keyReadFile(prv, prvkeyFile);

	cout << "n: " << *pub.n << endl;
	cout << "g: " << *pub.g << endl;
	cout << "lamda: " << *prv.lamda << endl;
	cout << "mu: " << *prv.mu << endl;
	cout << "Key generation done..." << endl << endl;

	std::cout << ">> copying message from file to array..." << std::endl;
	readFile(m1, message1File, n); //copying message from file to array
	readFile(m2, message2File, n); //copying message from file to array
	std::cout << "messages copied..." << std::endl << std::endl;

	std::cout << ">> Starting encryptionprocess..." << std::endl << std::endl;

	//timer
	auto begin = std::chrono::steady_clock::now();

	//cout << ">> Encrypting message array..." << endl;
	for (int i = 0; i < n; i++) {
		*(c1 + i) = enc(pub, *(m1 + i));
		*(c2 + i) = enc(pub, *(m2 + i));
	}

	auto end = std::chrono::steady_clock::now();


	std::cout << "Message encrypted..." << std::endl << std::endl;

	std::cout << ">> copying cipher array to file..." << std::endl;
	createFile(cipher1File);
	writeFile(c1, cipher1File, n);
	createFile(cipher2File);
	writeFile(c2, cipher2File, n);
	std::cout << "cipher copied..." << std::endl << std::endl;

	std::cout << ">> Multipling cipher..." << std::endl;
	auto begin1 = std::chrono::steady_clock::now();
	unsigned long long n2 = (*pub.n)*(*pub.n);
	for (int i = 0; i < n; i++) {
		*(cres + i) = mulmod(*(c1 + i), *(c2 + i), n2);
	}
	auto end1 = std::chrono::steady_clock::now();
	std::cout << "cipher multiplied..." << std::endl << std::endl;

	std::cout << ">> Decrypting cipher..." << std::endl;
	auto begin2 = std::chrono::steady_clock::now();
	for (int i = 0; i < n; i++) {
		*(res + i) = dec(pub, prv, *(cres + i));
	}
	auto end2 = std::chrono::steady_clock::now();
	std::cout << "cipher decrypted..." << std::endl << std::endl;

	std::cout << ">> copying decrypted message array to file..." << std::endl;
	createFile(resultFile);
	writeFile(res, resultFile, n);
	std::cout << "decrypted message copied..." << std::endl << std::endl;

	std::cout << ">> Encryption and decryption process done..." << std::endl << std::endl;

	std::cout << ">> Freeing memory of the arrays..." << std::endl;
	delete[] m1;
	delete[] m2;
	delete[] c1;
	delete[] c2;
	delete[] cres;
	delete[] res;
	std::cout << "memory freed..." << std::endl << std::endl;

	//Printing time results
	std::cout << "Encryption time: " << std::endl;
	std::cout << "Elapsed time in nanoseconds: " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << " ns" << std::endl;
	std::cout << "Elapsed time in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() << " mus" << std::endl;
	std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << " ms" << std::endl;
	std::cout << "Elapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count() << " s" << std::endl << std::endl;

	std::cout << "Multiplication time: " << std::endl;
	std::cout << "Elapsed time in nanoseconds: " << std::chrono::duration_cast<std::chrono::nanoseconds> (end1 - begin1).count() << " ns" << std::endl;
	std::cout << "Elapsed time in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds> (end1 - begin1).count() << " mus" << std::endl;
	std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds> (end1 - begin1).count() << " ms" << std::endl;
	std::cout << "Elapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds> (end1 - begin1).count() << " s" << std::endl << std::endl;

	std::cout << "Decryption time: " << std::endl;
	std::cout << "Elapsed time in nanoseconds: " << std::chrono::duration_cast<std::chrono::nanoseconds> (end2 - begin2).count() << " ns" << std::endl;
	std::cout << "Elapsed time in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds> (end2 - begin2).count() << " mus" << std::endl;
	std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds> (end2 - begin2).count() << " ms" << std::endl;
	std::cout << "Elapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds> (end2 - begin2).count() << " s" << std::endl << std::endl;

	cout << "program ending..." << endl<<endl;

}