#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h" //__global__
#include "stdio.h"
#include "mypail.cuh"
#include <time.h> //time()
#include <string>
#include <ctime>
#include <chrono>



__global__ void keygen(pubkey pub, prvkey prv) {
	printf("calculating device key memory (pub_d, prv_d)\n");

	//setting up device variable (generating public key and device key)
	setup(pub, prv);
	
	printf("\n");
}

__global__ void encryption(pubkey pub, unsigned long long* m, unsigned long long* c, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) *(c + i) = enc(pub, *(m + i));

}

__global__ void decryption(pubkey pub, prvkey prv, unsigned long long* c, unsigned long long* m2, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) *(m2 + i) = dec(pub, prv, *(c + i));
}

__global__ void mul(pubkey pub,unsigned long long* c1, unsigned long long* c2, unsigned long long* cres, int n) {
	unsigned long long n2 = (*pub.n)*(*pub.n);
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) *(cres + i) = mulmod(*(c1 + i), *(c2 + i), n2);
}

void printArr(unsigned long long* c, int n) {
	std::cout << "Printing array... :" << std::endl;
	for (int i = 0; i < n; i++) {
		std::cout << *(c + i) << std::endl;
	}
}

int main() {
	printf("\nprogram starting... \n\n");

	//public key and private key for host
	printf("creating host and device variables...\n\n");
	pubkey pub;
	prvkey prv;
	//public key and private key for device
	pubkey pub_d;
	prvkey prv_d;
	//defining size
	size_t size = sizeof(unsigned long long);  
	//allocating device memory for device variables
	printf("allocating device memory for device variables...\n\n");
	cudaMalloc(&pub_d.n,size);
	cudaMalloc(&pub_d.g, size);
	cudaMalloc(&prv_d.lamda, size);
	cudaMalloc(&prv_d.mu, size);
	//running function on device
	printf("running device function...\n\n");
	keygen <<<1, 1 >>> (pub_d,prv_d);
	//waiting device to finish its job
	printf("waiting device to finish...\n\n");
	cudaDeviceSynchronize();
	//copy device memory to host memory
	printf("copying device key memory to host\n\n");
	cudaMemcpy(pub.n, pub_d.n, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(pub.g, pub_d.g, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(prv.lamda, prv_d.lamda, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(prv.mu,prv_d.mu, size, cudaMemcpyDeviceToHost);
	
	//printing the host result
	printf("printing host key memory (pub, prv): \n\n");

	printf("n: %llu\n", *pub.n);
	printf("g: %llu\n", *pub.g);
	printf("lamda: %llu\n", *prv.lamda);
	printf("mu: %llu\n\n", *prv.mu);

	//--------------------Encrytion and Decryption--------------------//
	printf("creating host and device variables for enc and dec...\n\n");
	std::string message1File = "message1.txt";
	std::string message2File = "message2.txt";
	std::string cipher1File = "cipher1.txt";
	std::string cipher2File = "cipher2.txt";
	std::string cresultFile = "cresult.txt";
	std::string resultFile = "result.txt";
	//create file
	createFile(cipher1File);
	createFile(cipher2File);
	createFile(cresultFile);
	createFile(resultFile);
	int n = 10;
	int block = (n + 31) /32;
	int thread = 32;
	unsigned long long* m1 = (unsigned long long*)malloc(n*size);
	unsigned long long* m2 = (unsigned long long*)malloc(n*size);
	unsigned long long* c1 = (unsigned long long*)malloc(n*size);
	unsigned long long* c2 = (unsigned long long*)malloc(n*size);
	unsigned long long* cres = (unsigned long long*)malloc(n*size);
	unsigned long long* res = (unsigned long long*)malloc(n*size);

	//same variables for device
	unsigned long long* dm1;
	unsigned long long* dm2;
	unsigned long long* dc1; 
	unsigned long long* dc2;
	unsigned long long* dcres;
	unsigned long long* dres;
	//allocate memory to device variable
	printf("allocating device memory...\n\n");
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
	//generating message in host message array
	printf("generating message in file...\n\n");
	srand(time(NULL));
	genRandFile(pub, message1File, n);
	genRandFile(pub, message2File, n);
	//read the message into host array
	printf("reading message into host array...\n\n");
	readFile(m1, message1File, n);
	readFile(m2, message2File, n);
	//printArr(m, n);
	//copy host message to device message
	printf("copying message to devcie variable...\n\n");
	//auto begin = std::chrono::steady_clock::now();
	cudaMemcpy(dm1,m1, n*size, cudaMemcpyHostToDevice);
	cudaMemcpy(dm2,m2, n*size, cudaMemcpyHostToDevice);
	//auto end = std::chrono::steady_clock::now();

	//starting encryption process
	printf("starting to encrypt the message... \n\n");
	encryption <<<block, thread >>> (pub_d, dm1, dc1,n);  //using encryption function
	encryption << <block, thread >> > (pub_d, dm2, dc2, n);  //using encryption function
	
	//auto begin2 = std::chrono::steady_clock::now();
	cudaMemcpy(c1, dc1, n*size, cudaMemcpyDeviceToHost); //copy cypher message to host
	cudaMemcpy(c2, dc2, n*size, cudaMemcpyDeviceToHost); //copy cypher message to host
	//auto end2 = std::chrono::steady_clock::now();
	//copy cipher text to file
	writeFile(c1, cipher1File, n);
	writeFile(c2, cipher2File, n);

	//testing homo prop
	printf("testing homo prop... \n\n");
	mul << <block, thread >> > (pub_d, dc1, dc2, dcres, n);
	cudaMemcpy(cres, dcres, n*size, cudaMemcpyDeviceToHost); //copy cypher message to host
	writeFile(cres, cresultFile, n);	//copy cipher text to file

	//starting to decrypt cipher text
	printf("starting to decrypt the cypher result... \n");
	decryption << <block, thread >> >(pub_d, prv_d, dcres, dres,n);  //using decryption function
	//time
	//auto begin1 = std::chrono::steady_clock::now();
	cudaMemcpy(res, dres, n*size, cudaMemcpyDeviceToHost);  //copy decrpted message to host
	//auto end1 = std::chrono::steady_clock::now();
	//copy decrypted message to file
	createFile(resultFile);
	writeFile(res, resultFile, n);
	//free the device memory
	printf("Freeing device memory\n\n");
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
	printf("Freeing host memory\n\n");
	delete[] m1;
	delete[] m2;
	delete[] c1;
	delete[] c2;
	delete[] cres;
	delete[] res;
	/*
	//Printing time results
	printf("Message from host to device: \n\n");
	std::cout << "Elapsed time in nanoseconds: " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << " ns" << std::endl;
	std::cout << "Elapsed time in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds> (end - begin).count() << " mus" << std::endl;
	std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << " ms" << std::endl;
	std::cout << "Elapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count() << " s" << std::endl;

	//Printing time results
	printf("Cipher from device to host: \n\n");
	std::cout << "Elapsed time in nanoseconds: " << std::chrono::duration_cast<std::chrono::nanoseconds> (end2 - begin2).count() << " ns" << std::endl;
	std::cout << "Elapsed time in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds> (end2 - begin2).count() << " mus" << std::endl;
	std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds> (end2 - begin2).count() << " ms" << std::endl;
	std::cout << "Elapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds> (end2 - begin2).count() << " s" << std::endl;

	//Printing time results
	printf("Decrypted message form device to host: \n\n");
	std::cout << "Elapsed time in nanoseconds: " << std::chrono::duration_cast<std::chrono::nanoseconds> (end1 - begin1).count() << " ns" << std::endl;
	std::cout << "Elapsed time in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds> (end1 - begin1).count() << " mus" << std::endl;
	std::cout << "Elapsed time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds> (end1 - begin1).count() << " ms" << std::endl;
	std::cout << "Elapsed time in seconds: " << std::chrono::duration_cast<std::chrono::seconds> (end1 - begin1).count() << " s" << std::endl;
	*/
	printf("program ending... \n\n");
}