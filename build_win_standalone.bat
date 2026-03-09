@echo off
:: build_win_standalone.bat
:: Configures and builds mandeye_convert.exe using CMake + MSVC
:: Run this from the repo root (d:\@GitHubMSI\mandeye_to_bag)
::
:: Requirements:
::   - Visual Studio 2019 or 2022 (with C++ workload)
::   - CMake 3.15+ (either standalone or the one bundled with VS)
::   - Internet access on first run (Eigen3 auto-download if not installed)
::     OR Eigen3 installed via vcpkg / system package
::
:: Output: build_standalone\bin\Release\mandeye_convert.exe

setlocal EnableDelayedExpansion

set "BUILD_DIR=%~dp0build_standalone"
set "SRC_DIR=%~dp0src\standalone"

echo ============================================================
echo  mandeye_convert — standalone Windows build
echo ============================================================
echo  Source : %SRC_DIR%
echo  Build  : %BUILD_DIR%
echo.

:: --- Configure ---
cmake -S "%SRC_DIR%" ^
      -B "%BUILD_DIR%" ^
      -G "Visual Studio 17 2022" ^
      -A x64 ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DLASZIP_BUILD_STATIC=ON
if errorlevel 1 (
    echo.
    echo [WARN] VS 2022 not found, trying VS 2019...
    cmake -S "%SRC_DIR%" ^
          -B "%BUILD_DIR%" ^
          -G "Visual Studio 16 2019" ^
          -A x64 ^
          -DCMAKE_BUILD_TYPE=Release ^
          -DLASZIP_BUILD_STATIC=ON
    if errorlevel 1 (
        echo.
        echo ERROR: CMake configure failed.
        echo Make sure Visual Studio (2019 or 2022) and CMake are installed.
        exit /b 1
    )
)

echo.
echo --- Building (Release) ---
cmake --build "%BUILD_DIR%" --config Release --parallel
if errorlevel 1 (
    echo.
    echo ERROR: Build failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Build succeeded!
echo  EXE: %BUILD_DIR%\bin\Release\mandeye_convert.exe
echo ============================================================
echo.
echo Usage:
echo   mandeye_convert.exe ^<input_dir^> ^<output.bag^> [--pc_topic /livox/lidar]
echo.
