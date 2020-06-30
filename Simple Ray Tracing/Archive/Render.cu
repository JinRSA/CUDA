#include "Render.h"

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

__host__ __device__ ColorRGB castRay(const Vec3f& pos, const Vec3f& dir, const Sphere* const spheres, const uint spheresCount, const Light* const lights, const uint lightsCount, uint depth)
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

__global__ void dev_exportToJPG(const unsigned short* const width, const unsigned short* const height, unsigned char* B, unsigned char* G, unsigned char* R, const Sphere* const spheres, const uint spheresCount, const Light* const lights, const uint lightsCount)
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
