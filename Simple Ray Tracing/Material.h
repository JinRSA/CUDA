#ifndef MATERIAL_H
#define MATERIAL_H
#include "Geometry.h"

typedef Vec3f ColorRGB;
#define RED x
#define GREEN y
#define BLUE z

struct Material
{
	ColorRGB diffuseColor;
	Vec3f reflectivity;	// Albedo.
	float specularExp;
	__host__ __device__ Material() : diffuseColor(), reflectivity(1.f, 0.f, 0.f), specularExp() {}
	__host__ __device__ Material(const ColorRGB& color, const Vec3f& reflect = Vec3f(1.5f, 0.9f, 0.f), const float& specExp = 500.f) : diffuseColor(color), reflectivity(reflect), specularExp(specExp) {}
};
#endif // !MATERIAL_H