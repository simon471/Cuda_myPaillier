//paillier library
#include "fileio.cuh"
#include "hypernyms.cuh"
//time record
#include <chrono>
//cuda
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

__global__ void encryption(pubkey pub, unsigned long long* m, unsigned long long* c, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) *(c + i) = enc(pub, *(m + i));

}

__global__ void decryption(pubkey pub, prvkey prv, unsigned long long* c, unsigned long long* m2, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) *(m2 + i) = dec(pub, prv, *(c + i));
}

__global__ void mul(pubkey pub, unsigned long long* c1, unsigned long long* c2, unsigned long long* cres, int n) {
	unsigned long long n2 = (*pub.n)*(*pub.n);
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) *(cres + i) = mulmod(*(c1 + i), *(c2 + i), n2);
}

int main() {
	printf(">> This file is the device test run of the encryption, decryption and homo prop.\n");
	printf(">> Program starting... \n\n");
	printf(">> Declaring variables...\n");
	//define variable
	size_t size = sizeof(unsigned long long);
	std::string pubkeyFile = "pubkey.txt";
	std::string prvkeyFile = "prvkey.txt";
	std::string message1File = "message1.txt";
	std::string message2File = "message2.txt";
	std::string cipher1File = "device_cipher1.txt";
	std::string cipher2File = "device_cipher2.txt";
	std::string cresultFile = "device_cresult.txt";
	std::string resultFile = "device_result.txt";
	int n = 100000;
	int block = (n + 31) / 32;
	int thread = 32;
	//define variables for host
	pubkey pub;
	prvkey prv;
	unsigned long long* m1 = (unsigned long long*)malloc(n*size);
	unsigned long long* m2 = (unsigned long long*)malloc(n*size);
	unsigned long long* c1 = (unsigned long long*)malloc(n*size);
	unsigned long long* c2 = (unsigned long long*)malloc(n*size);
	unsigned long long* cres = (unsigned long long*)malloc(n*size);
	unsigned long long* res = (unsigned long long*)malloc(n*size);
	//define variables for device
	pubkey pub_d;
	prvkey prv_d;
	unsigned long long* dm1;
	unsigned long long* dm2;
	unsigned long long* dc1;
	unsigned long long* dc2;
	unsigned long long* dcres;
	unsigned long long* dres;
	printf("Done...\n\n");

	//allocate and intialize memory to device variable
	printf(">> Allocating and intializing memory for device variable...\n");
	cudaMalloc(&pub_d.n, size);
	cudaMalloc(&pub_d.g, size);
	cudaMalloc(&prv_d.lamda, size);
	cudaMalloc(&prv_d.mu, size);
	cudaMalloc(&dm1, n*size);
	cudaMalloc(&dm2, n*size);
	cudaMalloc(&dc1, n*size);
	cudaMalloc(&dc2, n*size);
	cudaMalloc(&dcres, n*size);
	cudaMalloc(&dres, n*size);
	cudaMemset(&dm1, 0, n*size);
	cudaMemset(&dm2, 0, n*size);
	cudaMemset(&dc1, 0, n*size);
	cudaMemset(&dc2, 0, n*size);
	cudaMemset(&dcres, 0, n*size);
	cudaMemset(&dres, 0, n*size);
	printf("Done...\n\n");

	//create file
	printf(">> Creating files...\n");
	createFile(cipher1File);
	createFile(cipher2File);
	createFile(cresultFile);
	createFile(resultFile);
	printf("Done...\n\n");

	//read key
	printf(">> Reading public and private key to host...\n");
	keyReadFile(pub, pubkeyFile);
	keyReadFile(prv, prvkeyFile);
	printf("Done...\n\n");

	//write key to device key
	printf(">> Copying host key to device key...\n");
	cudaMemcpy(pub_d.n, pub.n,size, cudaMemcpyHostToDevice);
	cudaMemcpy(pub_d.g, pub.g,size, cudaMemcpyHostToDevice);
	cudaMemcpy(prv_d.lamda, prv.lamda,size, cudaMemcpyHostToDevice);
	cudaMemcpy(prv_d.mu, prv.mu, size, cudaMemcpyHostToDevice);
	printf("Done...\n\n");

	//read message
	printf(">> Reading message into host array...\n");
	readFile(m1, message1File, n);
	readFile(m2, message2File, n);
	printf("Done...\n\n");

	//copy host message to device message
	printf(">> Copying message to devcie variable...\n");
	auto begin = std::chrono::steady_clock::now();
	cudaMemcpy(dm1, m1, n*size, cudaMemcpyHostToDevice);
	cudaMemcpy(dm2, m2, n*size, cudaMemcpyHostToDevice);
	auto end = std::chrono::steady_clock::now();
	printf("Done...\n\n");

	//starting encryption process
	printf(">> Starting to encrypt the messages...\n");
	encryption <<<block, thread >>> (pub_d, dm1, dc1, n);  //using encryption function
	encryption <<<block, thread >>> (pub_d, dm2, dc2, n);  //using encryption function
	printf("Done...\n\n");

	//copy cipher texts to host
	printf(">> Copying device cipher to host...\n");
	auto begin2 = std::chrono::steady_clock::now();
	cudaMemcpy(c1, dc1, n*size, cudaMemcpyDeviceToHost); //copy cypher message to host
	cudaMemcpy(c2, dc2, n*size, cudaMemcpyDeviceToHost); //copy cypher message to host
	auto end2 = std::chrono::steady_clock::now();
	printf("Done...\n\n");

	//copy cipher texts to files
	printf(">> Writing ciphers into files...\n");
	writeFile(c1, cipher1File, n);
	writeFile(c2, cipher2File, n);
	printf("Done...\n\n");

	//testing homo prop
	printf(">> Multiplying ciphers...\n");
	mul <<<block, thread >>> (pub_d, dc1, dc2, dcres, n);
	printf("Done...\n\n");

	//copy cipher result to host
	printf(">> Copying cipher result to host...\n");
	cudaMemcpy(cres, dcres, n*size, cudaMemcpyDeviceToHost); //copy cypher message to host
	printf("Done...\n\n");

	//copy cipher result to file
	printf(">> Writing cipher result into file...\n");
	writeFile(cres, cresultFile, n);	//copy cipher text to file
	printf("Done...\n\n");

	//starting to decrypt cipher text
	printf(">> Starting to decrypt the cipher result...\n");
	decryption <<<block, thread >>>(pub_d, prv_d, dcres, dres, n);  //using decryption function
	printf("Done...\n\n");

	//copy decrpted message to host
	printf(">> Copying decrypted result to host...\n");
	auto begin1 = std::chrono::steady_clock::now();
	cudaMemcpy(res, dres, n*size, cudaMemcpyDeviceToHost);  
	auto end1 = std::chrono::steady_clock::now();
	printf("Done...\n\n");

	//copy decrypted message to file
	printf(">> Writing decrypted result into file...\n");
	writeFile(res, resultFile, n);
	printf("Done...\n\n");

	//free the device memory
	printf(">> Freeing device memory...\n");
	cudaFree(pub_d.n);
	cudaFree(pub_d.g);
	cudaFree(prv_d.lamda);
	cudaFree(prv_d.mu);
	cudaFree(dm1);
	cudaFree(dm2);
	cudaFree(dc1);
	cudaFree(dc2);
	cudaFree(dcres);
	cudaFree(dres);
	printf("Done...\n\n");

	//free the host memory
	printf(">> Freeing host memory...\n");
	delete[] m1;
	delete[] m2;
	delete[] c1;
	delete[] c2;
	delete[] cres;
	delete[] res;
	printf("Done...\n\n");

	//Printing time results
	printf("Message from host to device: \n");
	std::cout << "Elapsed time in nanoseconds: " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << " ns" << std::endl;
	std::cout << "Elapsed time in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() << " mus" << std::endl;
	std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << " ms" << std::endl;
	std::cout << "Elapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count() << " s" << std::endl << std::endl;

	//Printing time results
	printf("Cipher from device to host: \n");
	std::cout << "Elapsed time in nanoseconds: " << std::chrono::duration_cast<std::chrono::nanoseconds> (end2 - begin2).count() << " ns" << std::endl;
	std::cout << "Elapsed time in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds> (end2 - begin2).count() << " mus" << std::endl;
	std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds> (end2 - begin2).count() << " ms" << std::endl;
	std::cout << "Elapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds> (end2 - begin2).count() << " s" << std::endl << std::endl;

	//Printing time results
	printf("Decrypted message form device to host: \n");
	std::cout << "Elapsed time in nanoseconds: " << std::chrono::duration_cast<std::chrono::nanoseconds> (end1 - begin1).count() << " ns" << std::endl;
	std::cout << "Elapsed time in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds> (end1 - begin1).count() << " mus" << std::endl;
	std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds> (end1 - begin1).count() << " ms" << std::endl;
	std::cout << "Elapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds> (end1 - begin1).count() << " s" << std::endl << std::endl;

	printf("Program ending...\n\n");
}