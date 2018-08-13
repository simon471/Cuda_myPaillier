//paillier library
#include "fileio.cuh"
//cuda
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

__global__ void keygen(pubkey pub, prvkey prv) {
	
	//setting up device variable (generating public key and device key)
	setup(pub, prv);

}

int main() {
	printf(">> This file is going to use device code to generate public and private key to a file. \n\n");
	printf(">> Program starting... \n\n");

	//Declaring variables
	printf(">> Creating host and device variables...\n");
	//public key and private key for host
	pubkey pub;
	prvkey prv;
	//public key and private key for device
	pubkey pub_d;
	prvkey prv_d;
	size_t size = sizeof(unsigned long long);
	std::string pubkeyFile = "pubkey.txt";
	std::string prvkeyFile = "prvkey.txt";
	printf("Done...\n\n");

	//Allocate device memory
	printf(">> Allocating device memory for device variables...\n");
	cudaMalloc(&pub_d.n, size);
	cudaMalloc(&pub_d.g, size);
	cudaMalloc(&prv_d.lamda, size);
	cudaMalloc(&prv_d.mu, size);
	printf("Done...\n\n");

	printf(">> Running device function...\n");
	keygen << <1, 1 >> > (pub_d, prv_d);
	//waiting device to finish its job
	printf(">> Waiting device to finish...\n");
	cudaDeviceSynchronize();
	printf("Done...\n\n");

	//copy device memory to host memory
	printf(">> Copying device key memory to host key...\n");
	cudaMemcpy(pub.n, pub_d.n, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(pub.g, pub_d.g, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(prv.lamda, prv_d.lamda, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(prv.mu, prv_d.mu, size, cudaMemcpyDeviceToHost);
	printf("Done...\n\n");

	//create file
	printf(">> Creating public and private key files...\n");
	createFile(pubkeyFile);
	createFile(prvkeyFile);
	printf("Done...\n\n");

	//write key to file
	printf(">> Writing host key to file...\n");
	keyWriteFile(pub, pubkeyFile);
	keyWriteFile(prv, prvkeyFile);
	printf("Done...\n\n");

	//printing the host result
	printf(">> Printing host key memory (pub, prv): \n\n");
	printf("n: %llu\n", *pub.n);
	printf("g: %llu\n", *pub.g);
	printf("lamda: %llu\n", *prv.lamda);
	printf("mu: %llu\n\n", *prv.mu);
	printf("Done...\n\n");

	printf("Program ending...\n\n");
}