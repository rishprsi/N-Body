#ifndef GLUtil

#include <stdio.h>
#include <math.h>
#define PI 3.14159265

#ifdef WIN32
#include <glfw/glfw3.h>
#endif

void drawCircle(float x, float y, float r, float points);
void draw(double time);
void start();

#endif
