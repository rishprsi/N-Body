#include "GLUtil.h"


void drawCircle(float x, float y, float r, float points) {
    glBegin(GL_TRIANGLE_FAN);
    glColor3f(1, 1, 1);

    glVertex3f(x, y, 0.0f);

    for (int i = 0; i <= points; i++) {
        float angle = (2 * PI / points) * i;
        float nX = x + r * cos(angle);
        float nY = y + r * sin(angle);
        glVertex3f(nX, nY, 0.0f);
    }

    glEnd();
}


// Run this once before infinite loop starts
void start() {

}

// Run this in an infinite loop in main somewhere
void draw(double time) {
    for (int i = 0; i < 100; i++) {
        drawCircle(cos(time) / (i % 10) + (i / 100.0f), sin(time), 1 / 100.0f, 20);
    }    
}

