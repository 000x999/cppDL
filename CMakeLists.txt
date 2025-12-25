cmake_minimum_required(VERSION 3.20)
project(cppDL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(USE_AVX256   "Enable AVX2/FMA etc." ON)
option(DEBUG_CPPDL  "Enable DEBUG macro" OFF)
option(USE_VULKAN   "Enable Vulkan support" OFF)
option(USE_OPENGL   "Enable OpenGL support" OFF)

find_package(OpenMP)

file(GLOB_RECURSE CPPDL_IMPL_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/core/impl/neural_core_impl/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/core/impl/neural_memory_impl/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/core/impl/tokenizer_core_impl/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/core/impl/tensor_core_impl/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/core/impl/attention_core_impl/*.cpp
)

file(GLOB_RECURSE CPPDL_TEST_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/test_bed/src/*.cpp
)

set(CRUSHBLAS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/core/include/CRUSHBLAS_MODULE)
file(GLOB_RECURSE CRUSHBLAS_SOURCES CONFIGURE_DEPENDS
  ${CRUSHBLAS_ROOT}/core/impl/*.cpp
  ${CRUSHBLAS_ROOT}/core/impl/*.cc
  ${CRUSHBLAS_ROOT}/core/impl/*.cxx
)

add_executable(cppDL
  ${CPPDL_TEST_SOURCES}
  ${CPPDL_IMPL_SOURCES}
  ${CRUSHBLAS_SOURCES}
)

target_include_directories(cppDL PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/core
  ${CMAKE_CURRENT_SOURCE_DIR}/core/include
  ${CRUSHBLAS_ROOT}/core
  ${CRUSHBLAS_ROOT}/core/include
  ${CRUSHBLAS_ROOT}/core/impl
  ${CRUSHBLAS_ROOT}/core/BLAS
)

target_compile_definitions(cppDL PRIVATE
  USE_AVX256=$<BOOL:${USE_AVX256}>
  USE_VULKAN=$<BOOL:${USE_VULKAN}>
  USE_OPENGL=$<BOOL:${USE_OPENGL}>
  DEBUG=$<BOOL:${DEBUG_CPPDL}>
)

if(MSVC)
  target_compile_options(cppDL PRIVATE /W4 /O2)
  if(USE_AVX256)
    target_compile_options(cppDL PRIVATE /arch:AVX2)
  endif()
else()
  target_compile_options(cppDL PRIVATE
    -Wall -Wextra -Wpedantic
    -O3 -march=native
  )
  if(USE_AVX256)
    target_compile_options(cppDL PRIVATE -mavx2 -mfma -mavx)
  endif()
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(cppDL PRIVATE OpenMP::OpenMP_CXX)
endif()

if(WIN32)
  add_custom_target(run
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/run.bat"
    DEPENDS cppDL
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building and running cppDL via run.bat"
  )
endif()
