cmake_minimum_required(VERSION 3.20)
project(cppDL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(USE_AVX256  "Enable AVX2/FMA" ON)
option(USE_AVX512  "Enable AVX-512" OFF)
option(DEBUG_CPPDL "Enable DEBUG macro" OFF)
option(USE_VULKAN  "Enable Vulkan support" OFF)
option(USE_OPENGL  "Enable OpenGL support" OFF)

find_package(OpenMP)

set(CRUSHBLAS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/core/include/CRUSHBLAS_MODULE)

file(GLOB_RECURSE CRUSHBLAS_SOURCES CONFIGURE_DEPENDS
  ${CRUSHBLAS_ROOT}/core/impl/*.cpp
  ${CRUSHBLAS_ROOT}/core/impl/*.cc
  ${CRUSHBLAS_ROOT}/core/impl/*.cxx
)

file(GLOB_RECURSE CPPDL_IMPL_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/core/impl/neural_core_impl/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/core/impl/neural_memory_impl/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/core/impl/tokenizer_core_impl/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/core/impl/tensor_core_impl/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/core/impl/attention_core_impl/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/core/impl/model_core_impl/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/core/impl/safetensor_core_impl/*.cpp
)

file(GLOB_RECURSE CPPDL_TEST_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/test_bed/src/*.cpp
)

add_library(crushblas_core STATIC
  ${CRUSHBLAS_SOURCES}
)

target_include_directories(crushblas_core PUBLIC
  ${CRUSHBLAS_ROOT}/core
  ${CRUSHBLAS_ROOT}/core/include
  ${CRUSHBLAS_ROOT}/core/BLAS
)

add_library(crushblas::crushblas ALIAS crushblas_core)

add_library(cppdl_core STATIC
  ${CPPDL_IMPL_SOURCES}
)

target_include_directories(cppdl_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/core
  ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)

target_link_libraries(cppdl_core PUBLIC
  crushblas::crushblas
)

add_library(cppdl::cppdl ALIAS cppdl_core)

foreach(target crushblas_core cppdl_core)
  target_compile_definitions(${target} PUBLIC
    USE_AVX256=$<BOOL:${USE_AVX256}>
    USE_VULKAN=$<BOOL:${USE_VULKAN}>
    USE_OPENGL=$<BOOL:${USE_OPENGL}>
    $<$<BOOL:${DEBUG_CPPDL}>:DEBUG>
  )

  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /O2)
    if(USE_AVX256)
      target_compile_options(${target} PRIVATE /arch:AVX2)
    endif()
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic
      -O3 -march=native
    )

    if(USE_AVX256)
      target_compile_options(${target} PRIVATE
        -mavx2 -mfma -mavx
      )
    endif()

    if(USE_AVX512)
      target_compile_options(${target} PRIVATE
        -mavx512f -mavx512vl -mavx512bw -mavx512dq
      )
    endif()
  endif()

  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
  endif()
endforeach()

add_executable(cppDL
  ${CPPDL_TEST_SOURCES}
)

target_link_libraries(cppDL PRIVATE
  cppdl::cppdl
)

if(WIN32)
  add_custom_target(run
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/run.bat"
    DEPENDS cppDL
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building and running cppDL via run.bat"
  )
endif()
