#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include "GLUtil.h"

const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

cv::VideoWriter video("nbody.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));

void storeFrame(double time) {
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    cv::Scalar color = cv::Scalar(255, 255, 255);
    int radius = 5;
    cv::circle(image, cv::Point(time * 100,time * 100), radius, color, -1);
    video.write(image);
}


int main(int argc, char* argv[]) {

#ifdef WIN32
    // initialize GLFW
    if (!glfwInit())
        return -1;
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "N-Body Simulation", NULL, NULL); // create window
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window); // set context to current window

    glfwSetTime(0);
    double time = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {  // main loop 
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT); // clear the screen
        double deltaTime = glfwGetTime() - time;
        printf("%f ",deltaTime);
        storeFrame(time);
        draw(time);
        glfwSwapBuffers(window); // swap front and back buffers
        glfwPollEvents();        // poll for events
        time = glfwGetTime();
    }
    glfwTerminate(); // terminate GLFW
    video.release();
#endif
    return 0;
}