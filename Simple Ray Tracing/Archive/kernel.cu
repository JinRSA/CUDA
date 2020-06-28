#include <iostream>
#include "Input.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define HANDLE_ERROR(cudaError) HandleError(cudaError, __FILE__, __LINE__)

static void HandleError(const cudaError_t err, const char* const in_file, const int line)
{
	if (err != cudaSuccess)
	{
		printf("%s,\nin file %s,\nin line: %d.\n", cudaGetErrorString(err), in_file, line);
		exit(EXIT_FAILURE);
	}
}

int main()
{
	char demo, executor;
	unsigned short spheresCount, lightsCount, width, height;
	char fileName[64];
	input(demo, spheresCount, lightsCount, width, height, fileName, executor);
	Sphere* spheres = (Sphere*)malloc(spheresCount * sizeof(Sphere));
	Light* lightsource = (Light*)malloc(lightsCount * sizeof(Light));
	if (demo == 'n')
	{
		fillRandom(spheres, spheresCount, lightsource, lightsCount);
	}
	else
	{
		fillDemo(spheres, spheresCount, lightsource, lightsCount);
	}
	switch (executor)
	{
	case CPU:
		exportToJPG(width, height, spheres, spheresCount, lightsource, lightsCount, fileName);
		break;
	case GPU:
		{
			unsigned short* dev_Width, *dev_Height;
			unsigned char* dev_B, *dev_G, *dev_R;
			Sphere* dev_Spheres;
			Light* dev_Lights;
			HANDLE_ERROR(cudaMalloc((void**)&dev_Width, sizeof(unsigned short)));
			HANDLE_ERROR(cudaMalloc((void**)&dev_Height, sizeof(unsigned short)));
			HANDLE_ERROR(cudaMalloc((void**)&dev_B, width * height * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMalloc((void**)&dev_G, width * height * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMalloc((void**)&dev_R, width * height * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMalloc((void**)&dev_Spheres, spheresCount * sizeof(Sphere)));
			HANDLE_ERROR(cudaMalloc((void**)&dev_Lights, lightsCount * sizeof(Light)));
			HANDLE_ERROR(cudaMemcpy(dev_Width, &width, sizeof(unsigned short), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dev_Height, &height, sizeof(unsigned short), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dev_Spheres, spheres, spheresCount * sizeof(Sphere), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dev_Lights, lightsource, lightsCount * sizeof(Light), cudaMemcpyHostToDevice));
			size_t stackSize;
			HANDLE_ERROR(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
			std::cout << "Stack size = " << stackSize << std::endl;
			HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitStackSize, stackSize * 8));
			HANDLE_ERROR(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
			std::cout << "New stack size = " << stackSize << std::endl;
			clock_t Start = clock();
			dev_exportToJPG<<<dim3(width, height), 1>>>(dev_Width, dev_Height, dev_B, dev_G, dev_R, dev_Spheres, spheresCount, dev_Lights, lightsCount);
			cudaDeviceSynchronize();
			clock_t End = clock();
			std::cout << "GPU execution: " << ((double)End - Start) / CLOCKS_PER_SEC << std::endl;
			unsigned char* B = (unsigned char*)malloc(width * height * sizeof(unsigned char));
			unsigned char* G = (unsigned char*)malloc(width * height * sizeof(unsigned char));
			unsigned char* R = (unsigned char*)malloc(width * height * sizeof(unsigned char));
			HANDLE_ERROR(cudaMemcpy(B, dev_B, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(G, dev_G, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaMemcpy(R, dev_R, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
			BMP image;
			image.SetSize(width, height);
			std::cout << "Depth:\t" << image.TellBitDepth() << std::endl;
			image.SetBitDepth(24);
			std::cout << "New depth:\t" << image.TellBitDepth() << std::endl;
#pragma omp parallel for
			for (int i = 0, k = 0; i < height; ++i)
			{
				for (int j = 0; j < width; ++j, ++k)
				{
					image(j, i)->Blue = B[k];
					image(j, i)->Green = G[k];
					image(j, i)->Red = R[k];
					image(j, i)->Alpha = 0;
				}
			}
			cudaFree(dev_Width);
			cudaFree(dev_Height);
			cudaFree(dev_B);
			cudaFree(dev_G);
			cudaFree(dev_R);
			cudaFree(dev_Spheres);
			cudaFree(dev_Lights);
			image.WriteToFile(fileName);
			free(B);
			free(G);
			free(R);
		}
		break;
	case All:
#define TEST_COUNT 10
		exportToJPG(width, height, spheres, spheresCount, lightsource, lightsCount, nullptr, TEST_COUNT);
		{
			unsigned short* dev_Width, *dev_Height;
			unsigned char* dev_B, *dev_G, *dev_R;
			Sphere* dev_Spheres;
			Light* dev_Lights;
			HANDLE_ERROR(cudaMalloc((void**)&dev_Width, sizeof(unsigned short)));
			HANDLE_ERROR(cudaMalloc((void**)&dev_Height, sizeof(unsigned short)));
			HANDLE_ERROR(cudaMalloc((void**)&dev_B, width * height * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMalloc((void**)&dev_G, width * height * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMalloc((void**)&dev_R, width * height * sizeof(unsigned char)));
			HANDLE_ERROR(cudaMalloc((void**)&dev_Spheres, spheresCount * sizeof(Sphere)));
			HANDLE_ERROR(cudaMalloc((void**)&dev_Lights, lightsCount * sizeof(Light)));
			HANDLE_ERROR(cudaMemcpy(dev_Width, &width, sizeof(unsigned short), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dev_Height, &height, sizeof(unsigned short), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dev_Spheres, spheres, spheresCount * sizeof(Sphere), cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy(dev_Lights, lightsource, lightsCount * sizeof(Light), cudaMemcpyHostToDevice));
			size_t stackSize;
			HANDLE_ERROR(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
			std::cout << "Stack size = " << stackSize << std::endl;
			HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitStackSize, stackSize * 8));
			HANDLE_ERROR(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
			std::cout << "New stack size = " << stackSize << std::endl;
			for (int i = 0; i < TEST_COUNT; ++i)
			{
				clock_t Start = clock();
				dev_exportToJPG<<<dim3(width, height), 1>>>(dev_Width, dev_Height, dev_B, dev_G, dev_R, dev_Spheres, spheresCount, dev_Lights, lightsCount);
				cudaDeviceSynchronize();
				clock_t End = clock();
				std::cout << "GPU execution: " << ((double)End - Start) / CLOCKS_PER_SEC << std::endl;
			}
			cudaFree(dev_Width);
			cudaFree(dev_Height);
			cudaFree(dev_B);
			cudaFree(dev_G);
			cudaFree(dev_R);
			cudaFree(dev_Spheres);
			cudaFree(dev_Lights);
		}
		break;
	default:
		break;
	}
	free(spheres);
	free(lightsource);
	const auto q = 2.6e-05;
	return EXIT_SUCCESS;
}