//paillier library
#include "fileio.cuh"
#include "hypernyms.cuh"
//time record
#include <chrono>

using namespace std;

int main() {

	cout << ">> This file is the host test run of the encryption, decryption and homo prop." << endl;
	cout << ">> Program starting..." << endl << endl;

	//declare variables
	cout << ">> Declaring variables..." << endl;
	int n = 100000;
	pubkey pub;
	prvkey prv;
	unsigned long long* m1 = new unsigned long long[n];
	unsigned long long* m2 = new unsigned long long[n];
	unsigned long long* c1 = new unsigned long long[n];
	unsigned long long* c2 = new unsigned long long[n];
	unsigned long long* cres = new unsigned long long[n];
	unsigned long long* res = new unsigned long long[n];
	string message1File = "message1.txt";
	string message2File = "message2.txt";
	string pubkeyFile = "pubkey.txt";
	string prvkeyFile = "prvkey.txt";
	string cipher1File = "host_cipher1.txt";
	string cipher2File = "host_cipher2.txt";
	string cresultFile = "host_cresult.txt";
	string resultFile = "host_result.txt";
	
	cout << "Done..." << endl << endl;

	//read key
	cout << ">> Reading key values..." << endl;
	keyReadFile(pub, pubkeyFile);
	keyReadFile(prv, prvkeyFile);
	cout << "Done..." << endl << endl;

	//read message
	cout << ">> Reading messages..." << endl;
	readFile(m1, message1File, n);
	readFile(m2, message2File, n);
	cout << "Done..." << endl << endl;

	//encrypt message
	cout << ">> Encrypting messages..." << endl;
	auto begin = chrono::steady_clock::now();
	for (int i = 0; i < n; i++) {
		*(c1 + i) = enc(pub, *(m1 + i));
		*(c2 + i) = enc(pub, *(m2 + i));
	}
	auto end = chrono::steady_clock::now();
	cout << "Done..." << endl << endl;

	//write cipher to file
	cout << ">> Create and write message cipher to file..." << endl;
	createFile(cipher1File);
	writeFile(c1, cipher1File, n);
	createFile(cipher2File);
	writeFile(c2, cipher2File, n);
	cout << "Done..." << endl << endl;

	//multiply encrypted code
	cout << ">> Multiplying ciphers..." << endl;
	auto begin1 = chrono::steady_clock::now();
	unsigned long long n2 = (*pub.n)*(*pub.n);
	for (int i = 0; i < n; i++) {
		*(cres + i) = mulmod(*(c1 + i), *(c2 + i), n2);
	}
	auto end1 = chrono::steady_clock::now();
	cout << "Done..." << endl << endl;

	//write new cipher to file
	cout << ">> Create and write multiplied cipher to file..." << endl;
	createFile(cresultFile);
	writeFile(cres, cresultFile, n);
	cout << "Done..." << endl << endl;

	//decrypt message
	cout << ">> Decrypting multiplied cipher..." << endl;
	auto begin2 = chrono::steady_clock::now();
	for (int i = 0; i < n; i++) {
		*(res + i) = dec(pub, prv, *(cres + i));
	}
	auto end2 = chrono::steady_clock::now();
	cout << "Done..." << endl << endl;

	//write result to file
	cout << ">> Create and write result to file..." << endl;
	createFile(resultFile);
	writeFile(res, resultFile, n);
	cout << "Done..." << endl << endl;

	//free memory
	cout << ">> Freeing memories..." << endl;
	delete[] m1;
	delete[] m2;
	delete[] c1;
	delete[] c2;
	delete[] cres;
	delete[] res;
	cout << "Done..." << endl << endl;

	//Printing time results
	cout << "Encryption time: " << endl;
	cout << "Elapsed time in nanoseconds: " << chrono::duration_cast<chrono::nanoseconds> (end - begin).count() << " ns" << endl;
	cout << "Elapsed time in microseconds: " << chrono::duration_cast<chrono::microseconds> (end - begin).count() << " mus" << endl;
	cout << "Elapsed time in milliseconds: " << chrono::duration_cast<chrono::milliseconds> (end - begin).count() << " ms" << endl;
	cout << "Elapsed time in seconds: " << chrono::duration_cast<chrono::seconds> (end - begin).count() << " s" << endl << endl;

	cout << "Multiplication time: " << endl;
	cout << "Elapsed time in nanoseconds: " << chrono::duration_cast<chrono::nanoseconds> (end1 - begin1).count() << " ns" << endl;
	cout << "Elapsed time in microseconds: " << chrono::duration_cast<chrono::microseconds> (end1 - begin1).count() << " mus" << endl;
	cout << "Elapsed time in milliseconds: " << chrono::duration_cast<chrono::milliseconds> (end1 - begin1).count() << " ms" << endl;
	cout << "Elapsed time in seconds: " << chrono::duration_cast<chrono::seconds> (end1 - begin1).count() << " s" << endl << endl;

	cout << "Decryption time: " << endl;
	cout << "Elapsed time in nanoseconds: " << chrono::duration_cast<chrono::nanoseconds> (end2 - begin2).count() << " ns" << endl;
	cout << "Elapsed time in microseconds: " << chrono::duration_cast<chrono::microseconds> (end2 - begin2).count() << " mus" << endl;
	cout << "Elapsed time in milliseconds: " << chrono::duration_cast<chrono::milliseconds> (end2 - begin2).count() << " ms" << endl;
	cout << "Elapsed time in seconds: " << chrono::duration_cast<chrono::seconds> (end2 - begin2).count() << " s" << endl << endl;

	cout << "Program ending..." << endl << endl;
}