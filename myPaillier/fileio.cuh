#ifndef __FILEIO_CUH__
#define __FILEIO_CUH__

//includes
#include <iostream>
#include "keygen.cuh"
#include <fstream>
#include <string>
using namespace std;

//Prototype
void genRandFile(pubkey, string, int);
void createFile(string);
void writeFile(unsigned long long*, string, int);
void readFile(unsigned long long*, string, int);
void keyWriteFile(pubkey, string);
void keyWriteFile(prvkey, string);
void keyReadFile(pubkey, string);
void keyReadFile(prvkey, string);

//functions total = 8

//create a file with nth random numbers
void genRandFile(pubkey pub, string filename, int n) {
	ofstream file(filename);
	//srand(time(NULL));
	if (file.is_open()) {
		for (int i = 0; i < n; i++) {
			file << rnghost(10000000) << "\n";
		}
		file.close();
	}
}

//create a file
void createFile(string filename) {
	ofstream file(filename);
	file.close();
}

//write cipher array into file
void writeFile(unsigned long long* carry, string filename, int n) {
	ofstream file(filename);
	if (file.is_open()) {
		for (int i = 0; i < n; i++) {
			file << *(carry + i) << "\n";
		}
		file.close();
	}
	else cout << "Unable to open file" << endl;
}

//read file into array
void readFile(unsigned long long* marray, string filename, int n) {
	ifstream file(filename);
	unsigned long long value;
	for (int i = 0; file >> value; i++) // While file can be read in
	{
		*(marray + i) = value;
	}
	file.close();

}

void keyWriteFile(pubkey pub, string filename) {
	ofstream file(filename);
	if (file.is_open()) {
		file << *pub.n << "\n";
		file << *pub.g << "\n";
		file.close();
	}
}

void keyWriteFile(prvkey prv, string filename) {
	ofstream file(filename);
	if (file.is_open()) {
		file << *prv.lamda << "\n";
		file << *prv.mu << "\n";
		file.close();
	}
}

void keyReadFile(pubkey pub, string filename) {
	ifstream file(filename);
	file >> *pub.n;
	file >> *pub.g;
	file.close();

}

void keyReadFile(prvkey prv, string filename) {
	ifstream file(filename);
	file >> *prv.lamda;
	file >> *prv.mu;
	file.close();

}


#endif
