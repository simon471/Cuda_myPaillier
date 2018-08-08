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
	std::string messageFile = "message.txt";
	std::string cipherFile = "cipher.txt";
	std::string message2File = "message2.txt";
	int n = 100000;
	int block = (n + 31) /32;
	int thread = 32;
	unsigned long long* m = (unsigned long long*)malloc(n*size);
	unsigned long long* c = (unsigned long long*)malloc(n*size);
	unsigned long long* m2 = (unsigned long long*)malloc(n*size);

	//same variables for device
	unsigned long long* dm;
	unsigned long long* dc; 
	unsigned long long* dm2;
	//allocate memory to device variable
	printf("allocating device memory...\n\n");
	cudaMalloc(&dm, n*size);
	cudaMalloc(&dc, n*size);
	cudaMalloc(&dm2, n*size);
	cudaMemset(&dm, 0, n*size);
	cudaMemset(&dc, 0, n*size);
	cudaMemset(&dm2, 0, n*size);
	//generating message in host message array
	printf("generating message in file...\n\n");
	genRandFile(pub, messageFile, n);
	//read the message into host array
	printf("reading message into host array...\n\n");
	readFile(m, messageFile, n);
	//printArr(m, n);
	//copy host message to device message
	printf("copying message to devcie variable...\n\n");
	auto begin = std::chrono::steady_clock::now();
	cudaMemcpy(dm,m, n*size, cudaMemcpyHostToDevice);
	auto end = std::chrono::steady_clock::now();

	//starting encryption process
	printf("starting to encrypt the message... \n\n");
	encryption <<<block, thread >>> (pub_d, dm, dc,n);  //using encryption function
	
	auto begin2 = std::chrono::steady_clock::now();
	cudaMemcpy(c, dc, n*size, cudaMemcpyDeviceToHost); //copy cypher message to host
	auto end2 = std::chrono::steady_clock::now();
	//copy cipher text to file
	createFile(cipherFile);
	writeFile(c, cipherFile, n);
	//starting to decrypt cipher text
	printf("starting to decrypt the cypher... \n");
	decryption << <block, thread >> >(pub_d, prv_d, dc, dm2,n);  //using decryption function
	//time
	auto begin1 = std::chrono::steady_clock::now();
	cudaMemcpy(m2, dm2, n*size, cudaMemcpyDeviceToHost);  //copy decrpted message to host
	auto end1 = std::chrono::steady_clock::now();
	//copy decrypted message to file
	createFile(message2File);
	writeFile(m2, message2File, n);
	//free the device memory
	printf("Freeing device memory\n\n");
	cudaFree(pub_d.n);
	cudaFree(pub_d.g);
	cudaFree(prv_d.lamda);
	cudaFree(prv_d.mu);
	cudaFree(dm);
	cudaFree(dc);
	cudaFree(dm2);
	printf("Freeing host memory\n\n");
	delete[] m;
	delete[] c;
	delete[] m2;

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

	printf("program ending... \n\n");
}