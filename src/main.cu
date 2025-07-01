#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "raytracer.cuh"

void checkCudaError(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error during %s: %s\n", operation, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Camera* camera = (Camera*) glfwGetWindowUserPointer(window);
    // movement
    if (key == GLFW_KEY_W) camera->mv.F = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_S) camera->mv.B = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_A) camera->mv.L = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_D) camera->mv.R = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_E) camera->mv.U = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_Q) camera->mv.D = action == GLFW_PRESS || action == GLFW_REPEAT;
    // direction
    if (key == GLFW_KEY_LEFT)  camera->mv.RL = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_RIGHT) camera->mv.RR = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_UP)    camera->mv.RU = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_DOWN)  camera->mv.RD = action == GLFW_PRESS || action == GLFW_REPEAT;
}

int main() {
    const int width = 800;
    const int height = 600;
    unsigned char* imageData = (unsigned char*)malloc(width * height * 3 * sizeof(unsigned char));

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(width, height, "CUDA Ray-Tracing", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    Camera hostCamera;
    hostCamera.position = { .fields = { 5.0, 0.0, 0.0 } };
    hostCamera.direction = { .fields = { -1.0, 0.0, 0.0 } };
    hostCamera.up = { .fields = { 0.0, 0.0, 1.0 } };
    hostCamera.mv = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    hostCamera.moveSpeed = 5.0;
    hostCamera.rotaSpeed = 0.05;
    hostCamera.fov = 55 * PI / 180;

    glfwSetWindowUserPointer(window, &hostCamera);
    glfwSetKeyCallback(window, keyCallback);

    // CUDA
    cudaError_t err;

    Camera* devCamera;
    err = cudaMalloc((void**)&devCamera, sizeof(Camera));
    checkCudaError(err, "cudaMalloc");

    unsigned char* devImageData;
    err = cudaMalloc((void**)&devImageData, width * height * 3 * sizeof(unsigned char));
    checkCudaError(err, "cudaMalloc");

    // KERNELS
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // LOOP
    double previousTime = glfwGetTime();
    double previousSecondTime = previousTime;
    double currentTime, deltaTime;
    int frameCount = 0;

    while (!glfwWindowShouldClose(window)) {
        currentTime = glfwGetTime();
        deltaTime = currentTime - previousTime;
        previousTime = currentTime;
        frameCount++;
        if (currentTime - previousSecondTime >= 1.0) {
            printf("FPS: %d\n", frameCount);
            printf("POSITION: %f, %f, %f\n", hostCamera.position.raw[0], hostCamera.position.raw[1], hostCamera.position.raw[2]);
            printf("DIRECTION: %f, %f, %f\n", hostCamera.direction.raw[0], hostCamera.direction.raw[1], hostCamera.direction.raw[2]);
            frameCount = 0;
            previousSecondTime = currentTime;
        }
        
        cam_updateMv(&hostCamera, deltaTime);
        err = cudaMemcpy(devCamera, &hostCamera, sizeof(Camera), cudaMemcpyHostToDevice);
        checkCudaError(err, "cudaMemcpy");

        rayTraceKernel<<<gridSize, blockSize>>>(devImageData, width, height, devCamera);

        err = cudaGetLastError();
        checkCudaError(err, "kernel launch");

        err = cudaMemcpy(imageData, devImageData, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        checkCudaError(err, "cudaMemcpy");

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, imageData);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    err = cudaFree(devImageData);
    checkCudaError(err, "cudaFree");
    err = cudaFree(devCamera);
    checkCudaError(err, "cudaFree");
    free(imageData);

    glfwTerminate();
    return 0;
}
