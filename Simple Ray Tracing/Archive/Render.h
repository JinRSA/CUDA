#ifndef RENDER_H
#define RENDER_H
#include "Geometry.h"
#include <fstream>
#define M_PI       3.14159265358979323846	// pi
#define M_PI_2     1.57079632679489661923	// pi / 2
#define FLT_MAX 3.402823466e+38F
#define MAX_DEPTH 10
#include <algorithm>
#include <vector>
#include "EasyBMP.h"
#include "Sphere.h"
#include "Light.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned int uint;

// I - angle of incidence (normalized);
// N - surface normals (normalized).
__host__ __device__ Vec3f reflect(const Vec3f& I, const Vec3f& N) noexcept;
__host__ __device__ bool sceneIntersect(const Vec3f& pos, const Vec3f& dir, const Sphere* const spheres, const uint spheresCount, Vec3f& hit, Vec3f& N, Material& material);
__host__ __device__ ColorRGB castRay(const Vec3f& pos, const Vec3f& dir, const Sphere* const spheres, const uint spheresCount, const Light* const lights, const uint lightsCount, uint depth = 0);
__global__ void dev_exportToJPG(const unsigned short* const width, const unsigned short* const height, unsigned char* B, unsigned char* G, unsigned char* R, const Sphere* const spheres, const uint spheresCount, const Light* const lights, const uint lightsCount);
void exportToJPG(const unsigned short width, const unsigned short height, const Sphere* const spheres, const uint spheresCount, const Light* const lights, const uint lightsCount, const char* const fileName = nullptr, const int testCount = 1);
//void exportToPPV();
#endif // !RENDER_H