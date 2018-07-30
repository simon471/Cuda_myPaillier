#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h" //__global__
#include "stdio.h"
#include "mypail.cuh"
#include <time.h> //time()



__global__ void test(pubkey pub, prvkey prv) {
	printf("calculating device key memory (pub_d, prv_d)\n");

	//setting up device variable (generating public key and device key)
	setup(pub, prv);
	
	printf("\n");
}

__global__ void encryption(pubkey pub, unsigned long long* m, unsigned long long* c) {
	*c = enc(pub, *m);
	printf("device cypher message: %llu\n", *c);
}

__global__ void decryption(pubkey pub, prvkey prv, unsigned long long* c, unsigned long long* m2) {
	*m2 = dec(pub, prv, *c);
	printf("device decrypted message: %llu\n", *m2);
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
	test <<<1, 1 >>> (pub_d,prv_d);
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
	//create message for encryption
	unsigned long long m = 123;
	//create cypher
	unsigned long long c;
	//create message for decryption
	unsigned long long m2;

	//same variables for device
	unsigned long long* dm;
	unsigned long long* dc; 
	unsigned long long* dm2;
	//allocate memory to device variable
	printf("allocating device memory...\n\n");
	cudaMalloc(&dm, size);
	cudaMalloc(&dc, size);
	cudaMalloc(&dm2, size);
	//copy host message to device message
	printf("copying message to devcie variable...\n\n");
	cudaMemcpy(dm, &m, size, cudaMemcpyHostToDevice);

	printf("message for encrytion: %llu\n\n", m);

	printf("starting to encrypt the message... \n\n");
	encryption << <1, 1 >> > (pub_d, dm, dc);  //using encryption function
	cudaMemcpy(&c, dc, size, cudaMemcpyDeviceToHost); //copy cypher message to host
	printf("host cypher message: %llu\n\n",c);

	printf("starting to decrypt the cypher... \n");
	decryption << <1, 1 >> > (pub_d, prv_d, dc, dm2);  //using decryption function
	cudaMemcpy(&m2, dm2, size, cudaMemcpyDeviceToHost);  //copy decrpted message to host
	printf("host decrypted message: %llu\n\n",m2);

	//free the device memory
	printf("Freeing device memory\n\n");
	cudaFree(pub_d.n);
	cudaFree(pub_d.g);
	cudaFree(prv_d.lamda);
	cudaFree(prv_d.mu);
	cudaFree(dm);
	cudaFree(dc);
	cudaFree(dm2);


	printf("program ending... \n\n");
}