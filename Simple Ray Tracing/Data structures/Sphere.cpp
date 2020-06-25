#include "Sphere.h"

bool Sphere::RayIntersect(const Vec3f& pos, const Vec3f& dir, float& dist) const noexcept
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