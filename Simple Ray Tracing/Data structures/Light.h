#ifndef LIGHT_H
#define LIGHT_H
#include "Geometry.h"

class Light
{
public:
	Vec3f position;
	float intensity;
	Light(const Vec3f& pos, const float& intensity) : position(pos), intensity(intensity) { }
};
#endif // !LIGHT_H