cmake_minimum_required(VERSION 3.25)
project(BVH)

add_subdirectory(deps/glad/cmake)
glad_add_library(glad STATIC LANGUAGE c LOADER API gl:core=4.6)

add_subdirectory(deps/glm)
add_subdirectory(deps/bvh)
add_subdirectory(deps/glfw)

set(CMAKE_CXX_STANDARD 23)

target_compile_definitions(glm INTERFACE
    GLM_FORCE_AVX2
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_RADIANS
    GLM_FORCE_RIGHT_HANDED)

add_executable(BVH main.cpp)

target_link_libraries(BVH PRIVATE bvh glm glfw glad)
target_include_directories(BVH PRIVATE deps/cgltf)
target_compile_options(BVH PUBLIC
    -loop-vectorize
    -march=native
    -mmmx
    -msse
    -msse2
    -msse3
    -mssse3
    -msse4
    -msse4a
    -msse4.1
    -msse4.2
    -mavx
    -mavx2
    -msha
    -maes)