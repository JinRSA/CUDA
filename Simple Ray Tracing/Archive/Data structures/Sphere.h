#ifndef SPHERE_H
#define SPHERE_H
#include "Geometry.h"
#include "Material.h"

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
#endif // !SPHERE_H
