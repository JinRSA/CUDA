#include <iostream>
//#include "Input.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define HANDLE_ERROR(cudaError) HandleError(cudaError, __FILE__, __LINE__)

#include "Geometry.h"
#include "Material.h"
#include "Light.h"
#include "EasyBMP.h"
#include <time.h>
#include <algorithm>
#define RAND(x, y) (x - y) * (float)rand() / RAND_MAX + y
typedef unsigned int uint;
#define M_PI       3.14159265358979323846	// pi
#define M_PI_2     1.57079632679489661923	// pi / 2
#define FLT_MAX 3.402823466e+38F
#define MAX_DEPTH 10

class alignas(32) Sphere
{
public:
	Material material;
	alignas(32) Vec3f center;
	float radius;
	Sphere() : center(0.f, 0.f, 0.f), radius(1) {};
	Sphere(const decltype(center)& center, const float& radius, const Material& material) : center(center), radius(radius), material(material) {};
	__host__ __device__ bool RayIntersect(const Vec3f& pos, const Vec3f& dir, float& dist/*t0*/) const noexcept;
};
__host__ __device__ bool Sphere::RayIntersect(const Vec3f& pos, const Vec3f& dir, float& dist) const noexcept
{
	Vec3f L = center - pos;
	float tca = L * dir;
	float dSqr = L * L - tca * tca;
	float rSqr = radius * radius;
	if (dSqr > rSqr)
		return false;
	float thc = sqrt(rSqr - dSqr);
	dist = tca - thc;
	float t1 = tca + thc;
	if (dist < 0)
		dist = t1;
	if (dist < 0)
		return false;
	return true;
}

template<class T>
__host__ __device__ const T& max(const T& a, const T& b)
{
	return a < b ? b : a;
}

__host__ __device__ Vec3f reflect(const Vec3f& I, const Vec3f& N) noexcept
{
	return I - N * 2.f * (I * N);
}

__host__ __device__ bool sceneIntersect(const Vec3f& pos, const Vec3f& dir, const Sphere* const spheres, const uint spheresCount, Vec3f& hit, Vec3f& N, Material& material)
{
	float spheresDist = FLT_MAX;
	auto iTmp = UINT_MAX;
	for (auto i = decltype(spheresCount){0}; i < spheresCount; ++i)
	{
		float distI;
		if (spheres[i].RayIntersect(pos, dir, distI) && distI < spheresDist)
		{
			spheresDist = distI;
			iTmp = i;
		}
	}
	if (iTmp != UINT_MAX)
	{
		hit = pos + dir * spheresDist;
		N = (hit - spheres[iTmp].center).normalize();
		material = spheres[iTmp].material;
	}
	return spheresDist < 1000;
}

__host__ __device__ ColorRGB castRay(const Vec3f& pos, const Vec3f& dir, const Sphere* const __restrict__ spheres, const uint spheresCount, const Light* __restrict__ const lights, const uint lightsCount, uint depth = 0)
{
	Vec3f point, N;	// N - surface normal.
	Material material;
	if (depth > MAX_DEPTH || !sceneIntersect(pos, dir, spheres, spheresCount, point, N, material))
	{
		return /*backgroundColor*/ColorRGB(0.05f, 0.04f, 0.08f);
	}
	const Vec3f reflectDir = reflect(dir, N)/*.normalize()*/;
	const Vec3f reflectOrig = reflectDir * N < 0 ? point - N * 9.96e-4f : point + N * 9.96e-4f;
	const Vec3f reflectColor = castRay(reflectOrig, reflectDir, spheres, spheresCount, lights, lightsCount, depth + 1);
	float diffuseLightIntensivity = 0.f, specularLightIntensivity = 0.f;
	for (auto i = decltype(lightsCount){0}; i < lightsCount; ++i)
	{
		const Vec3f lightDir = (lights[i].position - point).normalize();
		const float lightDist = (lights[i].position - point).norm();
		const Vec3f shadowOrig = lightDir * N < 0 ? point - N * 9.96e-4f : point + N * 9.96e-4f;
		Vec3f shadowPoint, shadowN;
		Material dummyMat;
		if (sceneIntersect(shadowOrig, lightDir, spheres, spheresCount, shadowPoint, shadowN, dummyMat) && (shadowPoint - shadowOrig).norm() < lightDist)
		{
			continue;
		}
		diffuseLightIntensivity += lights[i].intensity * max(0.f, lightDir * N);
		specularLightIntensivity += pow(max(0.f, reflect(lightDir, N) * dir), material.specularExp) * lights[i].intensity;
	}
	return /*backgroundColor*/Vec3f(0.05f, 0.04f, 0.08f) * 0.4f +
		material.diffuseColor * diffuseLightIntensivity * material.reflectivity[0] +
		Vec3f(1.0f, 1.0f, 1.0f) * specularLightIntensivity * material.reflectivity[1] +
		reflectColor * material.reflectivity[2];
	//return material.diffuseColor * (diffuseLightIntensivity + specularLightIntensivity);
	//return material.diffuseColor * (diffuseLightIntensivity + specularLightIntensivity);
}

__global__ void dev_exportToJPG(const unsigned short* const __restrict__ width, const unsigned short* const __restrict__ height, unsigned char* __restrict__ B, unsigned char* __restrict__ G, unsigned char* __restrict__ R, const Sphere* const __restrict__ spheres, const uint spheresCount, const Light* const __restrict__ lights, const uint lightsCount)
{
	const auto j = blockIdx.x * blockDim.x + threadIdx.x;
	const auto i = blockIdx.y * blockDim.y + threadIdx.y;
	if (j < *width && i < *height)
	{
		const float fov = M_PI_2;//75.f;
		const float tang = tan(fov / 2.f / 2.f);
		const float rotX = 0.f, rotY = 0.f;
		const float x = (2 * (j + rotX) / (float)*width - 1.f) * tang * *width / (float)*height;
		const float y = -(2 * (i + rotY) / (float)*height - 1.f) * tang;
		const Vec3f dir = Vec3f(x, y, -1).normalize();
		ColorRGB col = castRay({ 0.f, 0.f, 0.f }, dir, spheres, spheresCount, lights, lightsCount);
		const float maxBGR = max(col.BLUE, max(col.GREEN, col.RED));
		if (maxBGR > 1.f)
		{
			col = col * (1.f / maxBGR);
		}
		//col = max(col.BLUE, max(col.GREEN, col.RED)) > 1.f ? col * (1.f / maxBGR) : col;
		B[j + i * *width] = col.BLUE * 255;
		G[j + i * *width] = col.GREEN * 255;
		R[j + i * *width] = col.RED * 255;
	}
}

const ColorRGB backgroundColor = { 0.05f, 0.04f, 0.08f };

void exportToJPG(const unsigned short width, const unsigned short height, const Sphere* const spheres, const uint spheresCount, const Light* const lights, const uint lightsCount, const char* const fileName = nullptr, const int testCount = 1)
{
	BMP image;
	image.SetSize(width, height);
	std::cout << "Depth:\t" << image.TellBitDepth() << std::endl;
	image.SetBitDepth(24);
	std::cout << "New depth:\t" << image.TellBitDepth() << std::endl;
	for (int e = 0; e < testCount; ++e)
	{
		clock_t Start = clock();
		const float fov = M_PI_2;//75.f;
		const float tang = tan(fov / 2.f / 2.f);
		const float rotX = 0.f, rotY = 0.f;
#pragma omp parallel for
		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				const float x = (2 * (j + rotX) / (float)width - 1.f) * tang * width / (float)height;
				const float y = -(2 * (i + rotY) / (float)height - 1.f) * tang;
				const Vec3f dir = Vec3f(x, y, -1).normalize();
				ColorRGB col = castRay({ 0.f, 0.f, 0.f }, dir, spheres, spheresCount, lights, lightsCount);
				const float maxBGR = std::max(col.BLUE, std::max(col.GREEN, col.RED));
				if (maxBGR > 1.f)
				{
					col = col * (1.f / maxBGR);
				}
				image(j, i)->Blue = col.BLUE * 255;
				image(j, i)->Green = col.GREEN * 255;
				image(j, i)->Red = col.RED * 255;
				image(j, i)->Alpha = 0;
			}
		}
		clock_t End = clock();
		std::cout << "CPU execution: " << ((double)End - Start) / CLOCKS_PER_SEC << std::endl;
	}
	if (fileName != nullptr)
	{
		image.WriteToFile(fileName);
	}
}

void fillRandom(Sphere* const spheres, const unsigned short spheresCount, Light* const lights, const unsigned short lightsCount)
{
	srand(clock());
	for (int i = 0; i < spheresCount; ++i)
	{
		spheres[i].center.x = RAND(-24.f, 24.f);
		spheres[i].center.y = RAND(-13.5f, 13.5f);
		spheres[i].center.z = RAND(-12.8f, -48.f);
		spheres[i].radius = RAND(1.64f, 5.6f);
		spheres[i].material = Material(ColorRGB{ RAND(0.f, 1.f), RAND(0.f, 1.f), RAND(0.f, 1.f) },
			Vec3f(RAND(0.f, 1.f), RAND(0.f, 1.f), RAND(0.f, 1.f)), RAND(2048.f, 6144.f));
	}
	for (int i = 0; i < lightsCount; ++i)
	{
		lights[i].intensity = RAND(0.248f, 1.64f);
		lights[i].position.x = RAND(-8.8f, 8.8f);
		lights[i].position.y = RAND(4.95f, 4.95f);
		lights[i].position.z = RAND(-10.f, 10.f);
	}
}

void fillDemo(Sphere* const spheres, const unsigned short spheresCount, Light* const lights, const unsigned short lightsCount)
{
	Material mat[3] = {
		{ColorRGB{ .4f, .4f, .3f }, Vec3f(1.5f, 1.f, 0.24f), 4096.f},
		{ColorRGB{ .3f, .1f, .1f }, Vec3f(1.5f, 1.f, 0.24f), 4096.f},
		{ColorRGB{ 1.f, 1.f, 1.f }, Vec3f(0.f, 1.f, 1.f), 4096.f} };
	spheres[0] = Sphere(Vec3f(1.96, -0.56, -16), 2.88f, mat[2]);
	float circleRadius = 16.f;
	constexpr float piSqr = M_PI * 2;
	int iSphere = 1;
	for (float i = 0.f; iSphere < spheresCount; i += M_PI_2 / /*3*/2, ++iSphere)
	{
		if (i > piSqr * M_PI_2)
		{
			i = 0.f;
		}
		spheres[iSphere] = Sphere(Vec3f(circleRadius * cos(i) + 0.f, circleRadius * sin(i) + 0.f, 16 + -2.4 * iSphere/*j*/), 2.4f * ((i + 0.08f) / 3.2f), mat[iSphere % 3]);
	}
	circleRadius = 5.6f;
	int iLight = 1;
	for (float i = 0.f; iLight < lightsCount; i += M_PI_2 / /*3*/2, ++iLight)
	{
		if (i > piSqr * M_PI_2)
		{
			i = 0.f;
		}
		lights[iLight] = Light(Vec3f(circleRadius * cos(i) + 0.f, circleRadius * sin(i) + 0.f, 3.2 + -6.4 * iLight/*j*/), 1.64f * ((i + 0.08f) / 4.8f));
	}
}

enum Execution
{
	CPU = '0', GPU = '1', All = '2'
};

void input(char& demo, unsigned short& spheresCount, unsigned short& lightsCount, unsigned short& width, unsigned short& height, char* fileName, char& executor)
{
	std::cout << "Show demo scene? [y/n]: ";
	do
	{
		std::cin >> demo;
		if (std::cin.fail())
		{
			std::cin.clear();
			std::cin.ignore();
		}
	} while (demo != 'y' && demo != 'n');
	std::cout << "Enter number of spheres [1:100]: ";
	do
	{
		std::cin >> spheresCount;
		if (std::cin.fail())
		{
			std::cin.clear();
			std::cin.ignore();
		}
	} while (spheresCount < 1 || spheresCount > 100);
	std::cout << "Enter number of lighting points [1:10]: ";
	do
	{
		std::cin >> lightsCount;
		if (std::cin.fail())
		{
			std::cin.clear();
			std::cin.ignore();
		}
	} while (lightsCount < 1 || lightsCount > 10);
	std::cout << "Enter width of the image [160:7680]: ";
	do
	{
		std::cin >> width;
		if (std::cin.fail())
		{
			std::cin.clear();
			std::cin.ignore();
		}
	} while (width < 160 || width > 7680);
	std::cout << "Enter height of the image [90:4320]: ";
	do
	{
		std::cin >> height;
		if (std::cin.fail())
		{
			std::cin.clear();
			std::cin.ignore();
		}
	} while (height < 90 || height > 4320);
	std::cout << "Execute on: [CPU -> 0; GPU -> 1, Get test -> 2]: ";
	do
	{
		std::cin >> executor;
		if (std::cin.fail())
		{
			std::cin.clear();
			std::cin.ignore();
		}
	} while (executor != CPU && executor != GPU && executor != All);
	std::cout << "Enter file name (with extension) [1-64 symbols]: ";
	std::cin >> fileName;
	std::cout << "Demo on: " << demo << std::endl;
	std::cout << "Spheres: " << spheresCount << std::endl;
	std::cout << "Lights: " << lightsCount << std::endl;
	std::cout << "Resolution: " << width << 'x' << height << std::endl;
	std::cout << "Executor: " << executor << std::endl;
	std::cout << "File: " << fileName << std::endl;
}


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
			HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitStackSize, stackSize * 4));
			HANDLE_ERROR(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
			std::cout << "New stack size = " << stackSize << std::endl;
			cudaFuncSetCacheConfig(dev_exportToJPG, cudaFuncCachePreferL1);
			//cudaFuncSetCacheConfig(dev_exportToJPG, cudaFuncCachePreferShared);
			//int val;
			//cudaDeviceGetAttribute(&val, cudaDeviceAttr::cudaDevAttrMaxBlockDimX, 0);
			//cudaDeviceGetAttribute(&val, cudaDeviceAttr::cudaDevAttrMaxBlockDimY, 0);
			const int maxBlockDimX = 4, maxBlockDimY = 16;
			const auto gridDimXY = dim3((width + maxBlockDimX - 1) / maxBlockDimX, (height + maxBlockDimY - 1) / maxBlockDimY);
			const auto blockDimXY = dim3(maxBlockDimX, maxBlockDimY);
			clock_t Start = clock();
			dev_exportToJPG<<<gridDimXY, blockDimXY>>>(dev_Width, dev_Height, dev_B, dev_G, dev_R, dev_Spheres, spheresCount, dev_Lights, lightsCount);
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
			HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitStackSize, stackSize * 4));
			HANDLE_ERROR(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
			std::cout << "New stack size = " << stackSize << std::endl;
			cudaFuncSetCacheConfig(dev_exportToJPG, cudaFuncCachePreferL1);
			const int maxBlockDimX = 4, maxBlockDimY = 16;
			const auto gridDimXY = dim3((width + maxBlockDimX - 1) / maxBlockDimX, (height + maxBlockDimY - 1) / maxBlockDimY);
			const auto blockDimXY = dim3(maxBlockDimX, maxBlockDimY);
			for (int i = 0; i < TEST_COUNT; ++i)
			{
				clock_t Start = clock();
				dev_exportToJPG<<<gridDimXY, blockDimXY>>>(dev_Width, dev_Height, dev_B, dev_G, dev_R, dev_Spheres, spheresCount, dev_Lights, lightsCount);
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
	return EXIT_SUCCESS;
}
