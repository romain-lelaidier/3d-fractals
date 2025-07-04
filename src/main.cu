#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "raytracer.cuh"
#include "scene.cuh"



// ----------- checkCudaError() -----------
//
// Checks if an error happened with CUDA.
// If so, prints it and closes the program
void checkCudaError(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error during %s: %s\n", operation, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}



// ----------- keyCallback() -----------
//
// Handles a keyboard event
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    KeyboardMover* kbm = (KeyboardMover*) glfwGetWindowUserPointer(window);
    bool c = action == GLFW_PRESS || action == GLFW_REPEAT;
    // movement
    if (key == GLFW_KEY_W) kbm->F = c;
    if (key == GLFW_KEY_S) kbm->B = c;
    if (key == GLFW_KEY_A) kbm->L = c;
    if (key == GLFW_KEY_D) kbm->R = c;
    if (key == GLFW_KEY_E) kbm->U = c;
    if (key == GLFW_KEY_Q) kbm->D = c;
    // direction
    if (key == GLFW_KEY_LEFT)  kbm->RL = c;
    if (key == GLFW_KEY_RIGHT) kbm->RR = c;
    if (key == GLFW_KEY_UP)    kbm->RU = c;
    if (key == GLFW_KEY_DOWN)  kbm->RD = c;
    // parameters
    if (key == GLFW_KEY_1) kbm->P[1] = c;
    if (key == GLFW_KEY_2) kbm->P[2] = c;
    if (key == GLFW_KEY_3) kbm->P[3] = c;
    if (key == GLFW_KEY_4) kbm->P[4] = c;
    if (key == GLFW_KEY_5) kbm->P[5] = c;
    if (key == GLFW_KEY_6) kbm->P[6] = c;
    if (key == GLFW_KEY_7) kbm->P[7] = c;
    if (key == GLFW_KEY_8) kbm->P[8] = c;
    if (key == GLFW_KEY_9) kbm->P[9] = c;
}



// ----------- fractalUpdateMv() -----------
//
// Updates the fractal depending on the parameters of mv
// For the Julia set : moves the parameter mu in the quaternion space
void fractalUpdateMv(KeyboardMover &kbm, Julia &julia, float dt) {
    const float speed = 1.0;
    julia.muGenerator.x += boolsToFloat(kbm.P[1], kbm.P[2]) * speed * dt;
    julia.muGenerator.y += boolsToFloat(kbm.P[3], kbm.P[4]) * speed * dt;
    julia.muGenerator.z += boolsToFloat(kbm.P[5], kbm.P[6]) * speed * dt;
    julia.muGenerator.w += boolsToFloat(kbm.P[7], kbm.P[8]) * speed * dt;
    julia.mu.x = cos(julia.muGenerator.x);
    julia.mu.y = cos(julia.muGenerator.y);
    julia.mu.z = cos(julia.muGenerator.z);
    julia.mu.w = cos(julia.muGenerator.w);
}



// ----------- main() -----------
//
//
int main() {

    // ----------- window initialization -----------
    
    // window dimensions
    const int width = 400;
    const int height = 300;

    // initialize the GLFW library
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    // opens a window
    GLFWwindow* window = glfwCreateWindow(width, height, "Hackathon - 3D fractals", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);



    // ----------- scene -----------

    // camera
    Camera camera;
    camera.position = { 0.0, 3.0, 0.0 };
    camera.direction = { 0.0, -1.0, 0.0 };
    camera.up = { 0.0, 0.0, 1.0 };
    camera.moveSpeed = 5.0;
    camera.rotaSpeed = 0.05;
    camera.fov = 55 * PI / 180;
    
    // movement handler
    KeyboardMover kbm = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    glfwSetWindowUserPointer(window, &kbm);
    glfwSetKeyCallback(window, keyCallback);

    // scene
    Scene scene;
    scene.nobjs = 3;
    scene.objs = (Object*) malloc(scene.nobjs * sizeof(Object));

    scene.objs[0].type = JULIA;
    scene.objs[0].julia = {
        // make_float4(acos(0), acos(0), acos(0), acos(0)),
        make_float4(acos(0.1), acos(0.5), acos(0.6), acos(-0.2)),
        make_float4(0, 0, 0, 0),
        0.001, 0.001, 10,
    };
    Julia* mjulia = &scene.objs[0].julia;

    scene.objs[1].type = SPHERE;
    scene.objs[1].sphere = { make_float3(2, 0, 0), 0.5, make_float3(0.5, 0, 0) };

    scene.objs[2].type = SPHERE;
    scene.objs[2].sphere = { make_float3(2, 1, 2), 0.5, make_float3(0, 0, 0.5) };



    // ----------- cuda initialization -----------

    cudaError_t err;

    // image colors, updated at each frame
    unsigned char* imageData = (unsigned char*)malloc(width * height * 3 * sizeof(unsigned char));

    unsigned char* devImageData;
    err = cudaMalloc((void**)&devImageData, width * height * 3 * sizeof(unsigned char));
    checkCudaError(err, "cudaMalloc");

    // kernels (threads dimensions)
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);



    // ----------- loop -----------

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
            // one second has elapsed
            printf("FPS: %d  -  ", frameCount);
            printf("mu: %.3f, %.3f, %.3f, %.3f\n", mjulia->mu.x, mjulia->mu.y, mjulia->mu.z, mjulia->mu.w);
            frameCount = 0;
            previousSecondTime = currentTime;
        }
        
        // update camera and fractal parameters
        cameraUpdateMv(kbm, camera, deltaTime);
        fractalUpdateMv(kbm, *mjulia, deltaTime);

        // raytracing the image
        rayTraceKernel<<<gridSize, blockSize>>>(
            devImageData,
            width, height,
            camera, scene
        );

        // check for rendering success
        err = cudaGetLastError();
        checkCudaError(err, "kernel launch");

        // display the resulting image
        err = cudaMemcpy(imageData, devImageData, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        checkCudaError(err, "cudaMemcpy");

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, imageData);

        // GLFW important actions
        glfwSwapBuffers(window);
        glfwPollEvents();
    }


    // ----------- closing -----------
    
    // free the allocated memory
    err = cudaFree(devImageData);
    checkCudaError(err, "cudaFree");
    free(imageData);

    // close the window
    glfwTerminate();

    return 0;
}
