project(Examples)

include_directories(
  "${CMAKE_SOURCE_DIR}/core2d/include"
  "${CMAKE_SOURCE_DIR}/core3d/include"
  "${CMAKE_SOURCE_DIR}"
)

CUDA_ADD_EXECUTABLE(Example1 example1.cu)
TARGET_LINK_LIBRARIES(Example1 LEVELSET_CORE)
CUDA_ADD_EXECUTABLE(Example2 example2.cu)
TARGET_LINK_LIBRARIES(Example2 LEVELSET_CORE)
