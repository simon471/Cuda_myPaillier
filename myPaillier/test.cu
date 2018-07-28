#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h" //__global__
#include "stdio.h"
#include "mypail.cuh"
#include <time.h> //time()



__global__ void test(pubkey pub, prvkey prv) {
	printf("printing device memory (pub_d, prv_d): \n\n");

	//setting up device variable (generating public key and device key)
	setup(pub, prv);
	
	printf("\n");
}

int main() {
	printf("\nprogram starting... \n\n");

	//public key and private key for host
	pubkey pub;
	prvkey prv;
	//public key and private key for device
	pubkey pub_d;
	prvkey prv_d;
	//defining size
	size_t size = sizeof(unsigned);  
	//allocating device memory for device variables
	cudaMalloc(&pub_d.n,size);
	cudaMalloc(&pub_d.g, size);
	cudaMalloc(&prv_d.lamda, size);
	cudaMalloc(&prv_d.mu, size);
	//running function on device
	test <<<1, 1 >>> (pub_d,prv_d);
	//waiting device to finish its job
	cudaDeviceSynchronize();
	//copy device memory to host memory
	cudaMemcpy(pub.n, pub_d.n, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(pub.g, pub_d.g, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(prv.lamda, prv_d.lamda, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(prv.mu,prv_d.mu, size, cudaMemcpyDeviceToHost);
	//free the device memory
	cudaFree(pub_d.n);
	cudaFree(pub_d.g);
	cudaFree(prv_d.lamda);
	cudaFree(prv_d.mu);
	//printing the host result
	printf("printing host memory (pub, prv): \n\n");

	printf("n: %d\n", *pub.n);
	printf("g: %d\n", *pub.g);
	printf("lamda: %d\n", *prv.lamda);
	printf("mu: %d\n", *prv.mu);


	printf("\nprogram ending... \n\n");
}