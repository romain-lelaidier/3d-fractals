#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "raytracer.cuh"
#include "fractal.cuh"

void checkCudaError(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error during %s: %s\n", operation, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Camera* camera = (Camera*) glfwGetWindowUserPointer(window);
    bool c = action == GLFW_PRESS || action == GLFW_REPEAT;
    // movement
    if (key == GLFW_KEY_W) camera->mv.F = c;
    if (key == GLFW_KEY_S) camera->mv.B = c;
    if (key == GLFW_KEY_A) camera->mv.L = c;
    if (key == GLFW_KEY_D) camera->mv.R = c;
    if (key == GLFW_KEY_E) camera->mv.U = c;
    if (key == GLFW_KEY_Q) camera->mv.D = c;
    // direction
    if (key == GLFW_KEY_LEFT)  camera->mv.RL = c;
    if (key == GLFW_KEY_RIGHT) camera->mv.RR = c;
    if (key == GLFW_KEY_UP)    camera->mv.RU = c;
    if (key == GLFW_KEY_DOWN)  camera->mv.RD = c;
    // parameters
    if (key == GLFW_KEY_1) camera->mv.P[1] = c;
    if (key == GLFW_KEY_2) camera->mv.P[2] = c;
    if (key == GLFW_KEY_3) camera->mv.P[3] = c;
    if (key == GLFW_KEY_4) camera->mv.P[4] = c;
    if (key == GLFW_KEY_5) camera->mv.P[5] = c;
    if (key == GLFW_KEY_6) camera->mv.P[6] = c;
    if (key == GLFW_KEY_7) camera->mv.P[7] = c;
    if (key == GLFW_KEY_8) camera->mv.P[8] = c;
    if (key == GLFW_KEY_9) camera->mv.P[9] = c;
}

void fractal_updateMv(Camera &camera, Julia &julia, float dt) {
    const float speed = 1.0;
    julia.muGenerator.x += boolsToFloat(camera.mv.P[1], camera.mv.P[2]) * speed * dt;
    julia.muGenerator.y += boolsToFloat(camera.mv.P[3], camera.mv.P[4]) * speed * dt;
    julia.muGenerator.z += boolsToFloat(camera.mv.P[5], camera.mv.P[6]) * speed * dt;
    julia.muGenerator.w += boolsToFloat(camera.mv.P[7], camera.mv.P[8]) * speed * dt;
    julia.mu.x = cos(julia.muGenerator.x);
    julia.mu.y = cos(julia.muGenerator.y);
    julia.mu.z = cos(julia.muGenerator.z);
    julia.mu.w = cos(julia.muGenerator.w);
}

int main() {
    const int width = 600;
    const int height = 400;
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

    Camera camera;
    camera.position = { 0.0, 3.0, 0.0 };
    camera.direction = { 0.0, -1.0, 0.0 };
    camera.up = { 0.0, 0.0, 1.0 };
    camera.mv = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    camera.moveSpeed = 5.0;
    camera.rotaSpeed = 0.05;
    camera.fov = 55 * PI / 180;

    Julia julia = {
        make_float4(acos(0.1), acos(0.5), acos(0.6), acos(-0.2)),
        make_float4(0, 0, 0, 0),
        0.001, 0.001, 10,
    };

    glfwSetWindowUserPointer(window, &camera);
    glfwSetKeyCallback(window, keyCallback);

    // CUDA
    cudaError_t err;

    unsigned char* devImageData;
    err = cudaMalloc((void**)&devImageData, width * height * 3 * sizeof(unsigned char));
    checkCudaError(err, "cudaMalloc");

    // KERNELS
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // LOOP
    float previousTime = glfwGetTime();
    float previousSecondTime = previousTime;
    float currentTime, deltaTime;
    int frameCount = 0;

    while (!glfwWindowShouldClose(window)) {
        currentTime = glfwGetTime();
        deltaTime = currentTime - previousTime;
        previousTime = currentTime;
        frameCount++;
        if (currentTime - previousSecondTime >= 1.0) {
            printf("FPS: %d\n", frameCount);
            printf("mu: %f, %f, %f, %f\n", julia.mu.x, julia.mu.y, julia.mu.z, julia.mu.w);
            // printf("POSITION: %f, %f, %f\n",  camera.position.x,  camera.position.y,  camera.position.z);
            // printf("DIRECTION: %f, %f, %f\n", camera.direction.x, camera.direction.y, camera.direction.z);
            frameCount = 0;
            previousSecondTime = currentTime;
        }
        
        cam_updateMv(camera, deltaTime);
        fractal_updateMv(camera, julia, deltaTime);

        rayTraceKernel<<<gridSize, blockSize>>>(
            devImageData,
            width, height,
            camera, julia
        );

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
    free(imageData);

    glfwTerminate();
    return 0;
}
