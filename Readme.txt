Paillier Cryptosystem
	- Visual Studio 2017 v15.6
	- CUDA 9.1
	
Terms:
	host = running on CPU
	device = running on GPU
	
How to use .cu file:
	Go to terminal
	>cd "file dir"
	>nvcc xxx.cu -o yyy
	>yyy or >nvprof yyy

>> 20th Aug 2018:

Update: 

1. add host_genkey for key generation on host
2. changed original genkey to device_genkey

[Result]
		        CPU		GPU
key generation	2-10ms	4-11ms



>> 13th Aug 2018:

Update: seperated and organized all functions

1. run genKey first to generate the public and private key
2. run genMessage to generate Messages (it is at the second because it need public key to work, message have to be smaller than n)
3. run host_run or device_run in favor

genKey.cu
	device code that generate public and private key

genMessage.cpp
	host code that generate two message array

host_run.cpp
	host code that take the key and message file and test the enc,dec and homo prop
	
device_run.cu
	host code that take the key and message file and test the enc,dec and homo prop
	
arith.cuh
	contains arithmetic functions
		isPrime
		gcd
		lcm
		L
		power
		powermod
		mulmod
		modInverse
	
fileio.cuh
	contains file read and write functions
		genRandFile
		createFile
		writeFile
		readFile
		keyWriteFile
		keyReadFiel
		
hypernyms.cuh
	contains encryption and decryption functions
		enc
		dec
		
keygen.cuh
	contains key structur and setup function
		struct pubkey and prvkey
		setup
		
randPail.cuh
	contains rng functions
		primeGen
		getRandZ
		rng
		rnghost
		

[Test record]
n(# of messages): 100000
block: (n-31)/32
thread: 32
		  cpu	  gpu
enc		 29 s	 42 ms
dec		  8 s	 26 ms
mul		190 ms	617 us




	
10th Aug 2018 and earlier:
	
[Early stage developing] One file included all:	
	mypail.cuh - have all the functions
	host_test.cpp - host test code
	device_test.cu - device test code
	
ref_random_example.cuh
	A reference of how to use random function on CUDA
	Source: http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html
	
	
Note:

How to use CUDA and CLASS?
https://stackoverflow.com/questions/6978643/cuda-and-classes

