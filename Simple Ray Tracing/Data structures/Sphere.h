#ifndef SPHERE_H
#define SPHERE_H
#include "Geometry.h"
#include "Material.h"

class Sphere
{
public:
	Material material;
	Vec3f center;
	float radius;
	Sphere() : center(0.f, 0.f, 0.f), radius(1) {};
	Sphere(const decltype(center)& center, const float& radius, const Material& material) : center(center), radius(radius), material(material) {};
	bool RayIntersect(const Vec3f& pos, const Vec3f& dir, float& dist/*t0*/) const noexcept;
};
#endif // !SPHERE_H