#include <iostream>

void printCudaDevicesInfo();

int main()
{
	std::cout << "Determining CUDA-capable GPU parameters\n\n";

	printCudaDevicesInfo();
	return 0;
}