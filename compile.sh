nvcc -o main \
    src/main.cu src/raytracer.cu src/camera.cu \
    src/geometry.cu src/fractal.cu src/math.cu \
    -Isrc -lglfw3 -lGL -lcudadevrt \
    --relocatable-device-code true