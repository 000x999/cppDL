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

option(USE_BLAS     "Placeholder for BLAS flag; currently unused." OFF)

find_package(OpenMP)

set(CPPDL_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CPPDL_CORE_DIR ${CPPDL_ROOT}/core)
set(CPPDL_INCLUDE_DIR ${CPPDL_CORE_DIR}/include)

set(CRUSHBLAS_ROOT ${CPPDL_INCLUDE_DIR}/CRUSHBLAS_MODULE)

file(GLOB_RECURSE CPPDL_IMPL_SOURCES CONFIGURE_DEPENDS
  ${CPPDL_CORE_DIR}/impl/neural_core_impl/*.cpp
  ${CPPDL_CORE_DIR}/impl/neural_memory_impl/*.cpp
  ${CPPDL_CORE_DIR}/impl/tokenizer_core_impl/*.cpp
  ${CPPDL_CORE_DIR}/impl/tensor_core_impl/*.cpp
  ${CPPDL_CORE_DIR}/impl/attention_core_impl/*.cpp
  ${CPPDL_CORE_DIR}/impl/model_core_impl/*.cpp
  ${CPPDL_CORE_DIR}/impl/safetensor_core_impl/*.cpp
)

file(GLOB_RECURSE CPPDL_TEST_SOURCES CONFIGURE_DEPENDS
  ${CPPDL_ROOT}/test_bed/src/*.cpp
)

file(GLOB_RECURSE CRUSHBLAS_SOURCES CONFIGURE_DEPENDS
  ${CRUSHBLAS_ROOT}/core/impl/*.cpp
  ${CRUSHBLAS_ROOT}/core/impl/*.cc
  ${CRUSHBLAS_ROOT}/core/impl/*.cxx
)

function(cppdl_apply_original_flags target_name visibility)

  target_compile_definitions(${target_name} ${visibility}
    USE_AVX256=$<BOOL:${USE_AVX256}>
    USE_VULKAN=$<BOOL:${USE_VULKAN}>
    USE_OPENGL=$<BOOL:${USE_OPENGL}>
    DEBUG=$<BOOL:${DEBUG_CPPDL}>
  )

  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /O2)

    if(USE_AVX256)
      target_compile_options(${target_name} PRIVATE /arch:AVX2)
    endif()

  else()
    target_compile_options(${target_name} PRIVATE
      -Wall -Wextra -Wpedantic
      -O3 -march=native
    )

    if(USE_AVX256)
      target_compile_options(${target_name} PRIVATE
        -mavx2 -mfma -mavx
        -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512cd
        -mavx512vbmi -mavx512vbmi2 -mavx512vnni -mavx512bitalg
        -mavx512vpopcntdq -mavx512ifma -mavx512bf16
      )
    endif()
  endif()

  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target_name} ${visibility} OpenMP::OpenMP_CXX)
  endif()

endfunction()

add_library(crushblas_core STATIC
  ${CRUSHBLAS_SOURCES}
)

target_include_directories(crushblas_core PUBLIC
  ${CRUSHBLAS_ROOT}/core
  ${CRUSHBLAS_ROOT}/core/include
  ${CRUSHBLAS_ROOT}/core/impl
  ${CRUSHBLAS_ROOT}/core/BLAS
)

cppdl_apply_original_flags(crushblas_core PUBLIC)

add_library(crushblas::crushblas ALIAS crushblas_core)

add_library(cppdl_core STATIC
  ${CPPDL_IMPL_SOURCES}
)

target_include_directories(cppdl_core PUBLIC
  ${CPPDL_ROOT}
  ${CPPDL_CORE_DIR}
  ${CPPDL_INCLUDE_DIR}
)

target_link_libraries(cppdl_core PUBLIC
  crushblas::crushblas
)

cppdl_apply_original_flags(cppdl_core PUBLIC)

add_library(cppdl::cppdl ALIAS cppdl_core)

add_executable(cppDL
  ${CPPDL_TEST_SOURCES}
  ${CPPDL_IMPL_SOURCES}
  ${CRUSHBLAS_SOURCES}
)

target_include_directories(cppDL PRIVATE
  ${CPPDL_ROOT}
  ${CPPDL_CORE_DIR}
  ${CPPDL_INCLUDE_DIR}
  ${CRUSHBLAS_ROOT}/core
  ${CRUSHBLAS_ROOT}/core/include
  ${CRUSHBLAS_ROOT}/core/impl
  ${CRUSHBLAS_ROOT}/core/BLAS
)

cppdl_apply_original_flags(cppDL PRIVATE)

if(WIN32)
  add_custom_target(run
    COMMAND "${CPPDL_ROOT}/run.bat"
    DEPENDS cppDL
    WORKING_DIRECTORY "${CPPDL_ROOT}"
    COMMENT "Building and running cppDL via run.bat"
  )
endif()
