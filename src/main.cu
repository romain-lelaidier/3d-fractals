#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "raytracer.cuh"

struct mov10 {
    bool front, back;
    bool left, right;
    bool up, down;
    bool turnR, turnL;
    bool turnU, turnD;
};

void checkCudaError(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error during %s: %s\n", operation, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    mov10* movement = (mov10*)glfwGetWindowUserPointer(window);
    // movement
    if (key == GLFW_KEY_W) movement->front = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_S) movement->back  = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_A) movement->left  = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_D) movement->right = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_E) movement->up    = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_Q) movement->down  = action == GLFW_PRESS || action == GLFW_REPEAT;
    // direction
    if (key == GLFW_KEY_LEFT)  movement->turnL = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_RIGHT) movement->turnR = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_UP)    movement->turnU = action == GLFW_PRESS || action == GLFW_REPEAT;
    if (key == GLFW_KEY_DOWN)  movement->turnD = action == GLFW_PRESS || action == GLFW_REPEAT;
}

void updateMovement(Camera* camera, mov10* movement, double dt) {
    vec3 m = { .fields = { 0.0, 0.0, 0.0 } };
    if (movement->back)  m.fields.x += 1.0;
    if (movement->front) m.fields.x -= 1.0;
    if (movement->right) m.fields.y += 1.0;
    if (movement->left)  m.fields.y -= 1.0;
    if (movement->up)    m.fields.z += 1.0;
    if (movement->down)  m.fields.z -= 1.0;
    vec3_normalize(&m);

    double speed = dt * 3.0;    // camera speed
    double fx = cos(camera->direction.fields.h);
    double fy = sin(camera->direction.fields.h);
    camera->position.fields.x += (fy * m.fields.y - fx * m.fields.x) * speed;
    camera->position.fields.y += (- fy * m.fields.x - fx * m.fields.y) * speed;
    camera->position.fields.z += m.fields.z * speed;

    double rspeed = dt * 0.3;   // camera rotationary speed
    if (movement->turnL) camera->direction.fields.h += rspeed;
    if (movement->turnR) camera->direction.fields.h -= rspeed;
    if (movement->turnU) camera->direction.fields.v += rspeed;
    if (movement->turnD) camera->direction.fields.v -= rspeed;
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
    hostCamera.position = { .fields = { 10.0, 0.0, 0.0 } };
    hostCamera.direction = { .fields = { PI, 0.0 } };
    hostCamera.wideness = 55 * PI / 180;

    mov10 movement = { false, false, false, false };
    glfwSetWindowUserPointer(window, &movement);
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
            frameCount = 0;
            previousSecondTime = currentTime;
        }

        updateMovement(&hostCamera, &movement, deltaTime);
        
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
