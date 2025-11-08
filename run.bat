@echo off
setlocal

REM 
if not exist "build\cppDL.exe" (
    echo [run_cppDL] Error: build\cppDL.exe not found.
    echo Build it first with: cmake --build --preset mingw-debug-build
    exit /b 1
)

echo [run_cppDL] Running cppDL.exe...
"build\cppDL.exe"

endlocal

