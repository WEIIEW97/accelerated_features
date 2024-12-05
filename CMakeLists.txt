cmake_minimum_required(VERSION 3.10)

project(xfeat)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Torch_DIR "/home/william/Codes/libtorch/share/cmake/Torch")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(TORCH_INCLUDE_DIRS
    ${TORCH_INSTALL_PREFIX}/include
    ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)

set(XFEAT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)

# file(GLOB_RECURSE torch_sources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/libtorch/*.cpp)
file(GLOB_RECURSE xfeat_torch_sources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/torchimpl/*.cpp)
file(GLOB_RECURSE xfeat_onnx_sources ${CMAKE_CURRENT_SOURCE_DIR}/csrc/onnximpl/*.cpp)
file(GLOB xfeat_shared_sources {CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)

add_executable(${PROJECT_NAME} 
    ${xfeat_torch_sources}
    ${xfeat_onnx_sources}
    ${xfeat_shared_sources}
    main.cpp
)

# add_library(${PROJECT_NAME} 
#     SHARED
#     ${onnx_sources}
#     ${shared_sources}
#     ${mex_sources}
# )


target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${XFEAT_INCLUDE_DIRS}
)


target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE 
    ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS}
)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)