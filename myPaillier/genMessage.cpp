//paillier library
#include "fileio.cuh"
using namespace std;

int main() {
	cout << ">> This file is going to generate two message files." << endl;
	cout << ">> Program starting..." << endl << endl;

	cout << ">> Declaring variables..." << endl;
	int n = 100000;
	pubkey pub;
	string message1File = "message1.txt";
	string message2File = "message2.txt";
	string pubkeyFile = "pubkey.txt";
	cout << "Done..." << endl << endl;

	//read keys from file
	cout << ">> Reading public key..." << endl;
	keyReadFile(pub, pubkeyFile);
	cout << "Done..." << endl << endl;

	//generate message file
	cout << ">> Generating message..." << endl;
	genRandFile(pub, message1File, n);
	genRandFile(pub, message2File, n);
	cout << "Done..." << endl << endl;

	cout << "Program ending..." << endl << endl;
}