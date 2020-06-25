#include <iostream>
#include <vector>
#include "Render.h"
#include "Sphere.h"
#include "Light.h"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

int main()
{
	using namespace std;
	Material ivory(ColorRGB{ .4f, .4f, .3f }, Vec3f(1.5f, 1.f, 0.24f), 4096.f);
	Material redRubber(ColorRGB{ .3f, .1f, .1f }, Vec3f(1.5f, 1.f, 0.24f), 4096.f);
	Material mirror1(ColorRGB{ 1.f, 1.f, 1.f }, Vec3f(0.f, 1.f, 1.f), 4096.f);
	Material mirror2(ColorRGB{ 1.f, 1.f, 1.f }, Vec3f(0.f, 1.f, 1.f), 4096.f);
	//vector<Material> materials;
	vector<Sphere> spheres;
	vector<Light> lightsource;
	//materials.emplace_back();
	spheres.emplace_back(Sphere(Vec3f(-3, 0, -16), 2, ivory));
	spheres.emplace_back(Sphere(Vec3f(-1.0, -1.5, -12), 2, redRubber));
	spheres.emplace_back(Sphere(Vec3f(1.5, -0.5, -18), 3, mirror1/*redRubber*/));
	spheres.emplace_back(Sphere(Vec3f(7, 5, -18), 4, mirror2/*ivory*/));
	lightsource.emplace_back(Light(Vec3f(-3.2f, -1.f, -10.f), 0.4f));
	lightsource.emplace_back(Light(Vec3f(3.2f, 1.f, 10.f), 1.24f));

	const float circleRadius = 16.f;
	constexpr float piSqr = M_PI * 2;
	for (float j = 0.f; j < 8; j += 1.6f)
	{
		for (float i = 0.f; i < piSqr; i += M_PI_2 / /*3*/2)
		{
			spheres.emplace_back(Sphere(Vec3f(circleRadius * cosf(i) + 0.f, circleRadius * sinf(i) + 0.f, -8 * j), 2.4f * (i / 2), (int)i % 2 == 0 ? ivory : redRubber));
		}
	}
	//for (float i = 0.f; i < piSqr * 4.8f; i += M_PI_2 / 3/*2*/)
	//{
	//	spheres.emplace_back(Sphere(Vec3f(circleRadius * cosf(i) + 0.56f * i, circleRadius * sinf(i) + 0.48f * i, -8.8f + 2.48f * (-i / 2.f)),
	//		3.2f * (/*i*/2 / 2), (int)i % 2 == 0 ? ivory : redRubber));
	//}
	exportToJPG(spheres, lightsource);
	return EXIT_SUCCESS;
}