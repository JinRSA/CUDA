#include "Input.h"

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
