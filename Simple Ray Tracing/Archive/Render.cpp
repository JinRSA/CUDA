#include "Render.h"
#include <time.h>
#include <omp.h>

const ColorRGB backgroundColor = { 0.05f, 0.04f, 0.08f };

void exportToJPG(const unsigned short width, const unsigned short height, const Sphere* const spheres, const uint spheresCount, const Light* const lights, const uint lightsCount, const char* const fileName, const int testCount)
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
//
//void exportToPPV()
//{
//	using namespace std;
//	vector<Vec3f> frameBuffer(width * height);
//	for (uint i = 0; i < height; ++i)
//	{
//		for (uint j = 0; j < width; ++j)
//		{
//			frameBuffer[i * width + j] = Vec3f(i / (float)height, j / (float)width, 0);
//		}
//	}
//	ofstream oFS;
//	oFS.open("Output image.ppm");
//	oFS << "P6\n" << width << " " << height << "\n255\n";
//	for (uint i = 0; i < width * height; ++i)
//	{
//		for (uint j = 0; j < 3; ++j)
//		{
//			oFS << char(255 * max(0.f, min(1.f, frameBuffer[i][j])));
//		}
//	}
//	oFS.close();
//}