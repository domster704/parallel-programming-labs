#include <iostream>

void printCudaDevicesInfo();

int main()
{
	std::cout << "Determining CUDA-capable GPU parameters\n\n";

	printCudaDevicesInfo();
	system("pause");
	return 0;
}