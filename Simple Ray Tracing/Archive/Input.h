#ifndef INPUT_H
#define INPUT_H
#include "Render.h"
#include <time.h>
#define RAND(x, y) (x - y) * (float)rand() / RAND_MAX + y

enum Execution
{
	CPU = '0', GPU = '1', All = '2'
};
void fillRandom(Sphere* const spheres, const unsigned short spheresCount, Light* const lights, const unsigned short lightsCount);
void fillDemo(Sphere* const spheres, const unsigned short spheresCount, Light* const lights, const unsigned short lightsCount);
void input(char &demo, unsigned short& spheresCount, unsigned short& lightsCount, unsigned short& width, unsigned short& height, char* fileName, char& executor);
#endif // !INPUT_H