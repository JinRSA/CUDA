#include "Render.h"
#include <omp.h>

static const uint width = 3840 / 2, height = 2160 / 2;
const ColorRGB backgroundColor = { 0.05f, 0.04f, 0.08f };

Vec3f reflect(const Vec3f& I, const Vec3f& N) noexcept
{
	return I - N * 2.f * (I * N);
}

bool sceneIntersect(const Vec3f& pos, const Vec3f& dir, const std::vector<Sphere>& spheres, Vec3f& hit, Vec3f& N, Material& material)
{
	float spheresDist = std::numeric_limits<float>::max();
	auto spheresCount = spheres.size();
	auto iTmp = std::numeric_limits<decltype(spheresCount)>::max();
	for (decltype(spheresCount) i = 0; i < spheresCount; ++i)
	{
		float distI;
		if (spheres[i].RayIntersect(pos, dir, distI) && distI < spheresDist)
		{
			spheresDist = distI;
			iTmp = i;
		}
	}
	if (iTmp != std::numeric_limits<decltype(iTmp)>::max())
	{
		hit = pos + dir * spheresDist;
		N = (hit - spheres[iTmp].center).normalize();
		material = spheres[iTmp].material;
	}
	return spheresDist < 1000;
}

ColorRGB castRay(const Vec3f& pos, const Vec3f& dir, const std::vector<Sphere>& spheres, const std::vector<Light>& lights, uint depth)
{
	Vec3f point, N;	// N - surface normal.
	Material material;
	if (depth > 24 || !sceneIntersect(pos, dir, spheres, point, N, material))
	{
		return backgroundColor;
	}
	const Vec3f reflectDir = reflect(dir, N)/*.normalize()*/;
	const Vec3f reflectOrig = reflectDir * N < 0 ? point - N * 9.96e-4f : point + N * 9.96e-4f;
	const Vec3f reflectColor = castRay(reflectOrig, reflectDir, spheres, lights, depth + 1);
	float diffuseLightIntensivity = 0.f, specularLightIntensivity = 0.f;
	const auto lightsCount = lights.size();
	for (auto i = decltype(lightsCount){ 0 }; i < lightsCount; ++i)
	{
		const Vec3f lightDir = (lights[i].position - point).normalize();
		const float lightDist = (lights[i].position - point).norm();
		const Vec3f shadowOrig = lightDir * N < 0 ? point - N * 9.96e-4f : point + N * 9.96e-4f;
		Vec3f shadowPoint, shadowN;
		Material dummyMat;
		if (sceneIntersect(shadowOrig, lightDir, spheres, shadowPoint, shadowN, dummyMat) && (shadowPoint - shadowOrig).norm() < lightDist)
		{
			continue;
		}
		diffuseLightIntensivity += lights[i].intensity * std::max(0.f, lightDir * N);
		specularLightIntensivity += pow(std::max(0.f, reflect(lightDir, N) * dir), material.specularExp) * lights[i].intensity;
	}
	return backgroundColor * 0.4f +
		material.diffuseColor * diffuseLightIntensivity * material.reflectivity[0] +
		Vec3f(1.0f, 1.0f, 1.0f) * specularLightIntensivity * material.reflectivity[1] +
		reflectColor * material.reflectivity[2];
	//return material.diffuseColor * (diffuseLightIntensivity + specularLightIntensivity);
}
//#include <time.h>
void exportToJPG(const std::vector<Sphere>& spheres, const std::vector<Light>& lights)
{
	BMP image;
	image.SetSize(width, height);
	std::cout << "Depth:\t" << image.TellBitDepth() << std::endl;
	image.SetBitDepth(24);
	std::cout << "New depth:\t" << image.TellBitDepth() << std::endl;
	//clock_t Start = clock();
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
			ColorRGB col = castRay({ 0.f, 0.f, 0.f }, dir, spheres, lights);
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
	/*clock_t End = clock();
	std::cout << ((double)End - Start) / CLOCKS_PER_SEC << std::endl;
	system("pause");*/
	image.WriteToFile("Output image.jpg");
}

void exportToPPV()
{
	using namespace std;
	vector<Vec3f> frameBuffer(width * height);
	for (uint i = 0; i < height; ++i)
	{
		for (uint j = 0; j < width; ++j)
		{
			frameBuffer[i * width + j] = Vec3f(i / (float)height, j / (float)width, 0);
		}
	}
	ofstream oFS;
	oFS.open("Output image.ppm");
	oFS << "P6\n" << width << " " << height << "\n255\n";
	for (uint i = 0; i < width * height; ++i)
	{
		for (uint j = 0; j < 3; ++j)
		{
			oFS << char(255 * max(0.f, min(1.f, frameBuffer[i][j])));
		}
	}
	oFS.close();
}