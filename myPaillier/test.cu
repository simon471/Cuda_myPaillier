#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h" //__global__
#include "stdio.h"
#include "mypail.cuh"
#include <time.h> //time()
#include <string>



__global__ void keygen(pubkey pub, prvkey prv) {
	printf("calculating device key memory (pub_d, prv_d)\n");

	//setting up device variable (generating public key and device key)
	setup(pub, prv);
	
	printf("\n");
}

__global__ void encryption(pubkey pub, unsigned long long* m, unsigned long long* c, int n) {
	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < n; i += stride) {
		*(c + i) = enc(pub, *(m + i));
	}
	printf("device encryption done\n\n");
}

__global__ void decryption(pubkey pub, prvkey prv, unsigned long long* c, unsigned long long* m2, int n) {
	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < n; i += stride) {
		*(m2 + i) = dec(pub, prv, *(c + i));
	}
	printf("device decryption done\n\n");
	
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
	int n = 30000;
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
	
	cudaMemcpy(dm,m, n*size, cudaMemcpyHostToDevice);
	//starting encryption process
	printf("starting to encrypt the message... \n\n");
	encryption <<<(n+255)/256, 256 >>> (pub_d, dm, dc,n);  //using encryption function
	
	cudaMemcpy(c, dc, n*size, cudaMemcpyDeviceToHost); //copy cypher message to host
	//copy cipher text to file
	createFile(cipherFile);
	writeFile(c, cipherFile, n);
	//starting to decrypt cipher text
	printf("starting to decrypt the cypher... \n");
	decryption << <(n + 255) / 256, 256 >> >(pub_d, prv_d, dc, dm2,n);  //using decryption function
	
	cudaMemcpy(m2, dm2, n*size, cudaMemcpyDeviceToHost);  //copy decrpted message to host
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

	printf("program ending... \n\n");
}