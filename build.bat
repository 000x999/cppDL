@echo off
setlocal

echo ========= BUILDING CPPDL =========

rem Arg1: configure preset name 
rem Arg2: build preset name    
rem Arg3: AVX flag          
rem Arg4: BLAS flag 

set CONFIG_PRESET=%1
set BUILD_PRESET=%2
set AVX_FLAG=%3
set BLAS_FLAG=%4

rem 
if "%CONFIG_PRESET%"=="" set CONFIG_PRESET=mingw-debug
if "%BUILD_PRESET%"=="" set BUILD_PRESET=mingw-debug-build
if "%AVX_FLAG%"=="" set AVX_FLAG=ON
if "%BLAS_FLAG%"=="" set BLAS_FLAG=ON



echo [build] Using CONFIG_PRESET=%CONFIG_PRESET%
echo [build] Using BUILD_PRESET=%BUILD_PRESET%
echo [build] Using USE_AVX256=%AVX_FLAG%
echo [build] Using USE_BLAS=%BLAS_FLAG%

echo [build] Configuring with CMake...
cmake --preset %CONFIG_PRESET% -DUSE_AVX256=%AVX_FLAG% -DUSE_BLAS=%BLAS_FLAG%
if errorlevel 1 goto :fail

echo [build] Building...
cmake --build --preset %BUILD_PRESET%
if errorlevel 1 goto :fail

echo ========= BUILD COMPLETE =========
goto :end

:fail
echo ========= BUILD FAILED =========

:end
endlocal

