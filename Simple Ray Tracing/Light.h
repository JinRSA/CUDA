#ifndef LIGHT_H
#define LIGHT_H
#include "Geometry.h"

class alignas(16) Light
{
public:
	Vec3f position;
	float intensity;
	Light() : position({ 0.f, 0.f, 0.f }), intensity(1.f) {}
	Light(const Vec3f& pos, const float& intensity) : position(pos), intensity(intensity) {}
};
#endif // !LIGHT_H
