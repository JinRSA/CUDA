#ifndef RENDER_H
#define RENDER_H
#include "Geometry.h"
#include <fstream>
#define M_PI       3.14159265358979323846	// pi
#define M_PI_2     1.57079632679489661923	// pi / 2
#include <algorithm>
#include "EasyBMP.h"
#include "Sphere.h"
#include "Light.h"

typedef unsigned int uint;

// I - angle of incidence (normalized);
// N - surface normals (normalized).
Vec3f reflect(const Vec3f& I, const Vec3f& N) noexcept;
bool sceneIntersect(const Vec3f& pos, const Vec3f& dir, const std::vector<Sphere>& spheres, Vec3f& hit, Vec3f& N, Material& material);
ColorRGB castRay(const Vec3f& pos, const Vec3f& dir, const std::vector<Sphere>& spheres, const std::vector<Light>& lights, uint depth = 0);
void exportToJPG(const std::vector<Sphere>& spheres, const std::vector<Light>& lights);
void exportToPPV();
#endif // !RENDER_H