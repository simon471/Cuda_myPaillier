#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "mypail.cuh"

pubkey pub_d;
prvkey prv_d;

__global__ void test(pubkey* pub, prvkey* prv) {
	printf("printing device memory: \n");
	
	setup(pub, prv);
	printf("\n");
}

int main() {
	printf("program starting... \n");

	pubkey pub;
	prvkey prv;
	
	initial(&pub, &prv);
	initial(&pub_d, &prv_d);

	unsigned x;
	unsigned* px = &x;
	size_t size = sizeof(unsigned);

	cudaMalloc(&pub_d.n,size);
	cudaMalloc(&pub_d.g, size);
	cudaMalloc(&prv_d.lamda, size);
	cudaMalloc(&prv_d.mu, size);
	/*
	unsigned dn, dg, dlamda, dmu = 0;
	
	cudaMemset(pub_d.n, dn, size);
	cudaMemset(pub_d.g, dg, size);
	cudaMemset(prv_d.lamda, dlamda, size);
	cudaMemset(prv_d.mu, dmu, size);
	*/
	test <<<1, 1 >>> (&pub_d, &prv_d);

	cudaDeviceSynchronize();

	cudaMemcpy(px, pub_d.n, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(pub.n, pub_d.n, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(pub.g, pub_d.g, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(prv.lamda, prv_d.lamda, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(prv.mu,prv_d.mu, size, cudaMemcpyDeviceToHost);

	//cudaMemcpy(prv, prv_d, sizeof(prvkey), cudaMemcpyDeviceToHost);

	

	cudaFree(pub_d.n);
	cudaFree(pub_d.g);
	cudaFree(prv_d.lamda);
	cudaFree(prv_d.mu);
	
	printf("printing host memory: \n");

	printf("n: %d\n", *pub.n);
	printf("g: %d\n", *pub.g);
	printf("lamda: %d\n", *prv.lamda);
	printf("mu: %d\n", *prv.mu);

	printf("x: %d\n", *px);


	printf("program ending... \n");
}