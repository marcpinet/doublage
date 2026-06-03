@echo off
setlocal DisableDelayedExpansion

net session >nul 2>&1
if %errorlevel% == 0 (
    goto :admin
) else (
    goto :elevate
)

:elevate
set "batchPath=%~f0"
set "batchArgs=%*"
set "vbsPath=%temp%\ElevateUAC.vbs"

setlocal EnableDelayedExpansion
echo Set UAC = CreateObject^("Shell.Application"^) > "%vbsPath%"
echo UAC.ShellExecute "!batchPath!", "!batchArgs!", "", "runas", 1 >> "%vbsPath%"
endlocal

"%vbsPath%"
del "%vbsPath%"
exit /b

:admin
setlocal EnableDelayedExpansion

set "PY=%~dp0.venv\Scripts\python.exe"

rem --- Locate the MSVC environment (needed by triton-windows, which compiles Whisper's
rem --- GPU kernels at runtime). Resolution order:
rem ---   1. VCVARS env var, if you want to point at a specific vcvars64.bat
rem ---   2. vswhere.exe (ships with every Visual Studio / Build Tools install since 2017)
rem ---   3. default install paths of VS 2022 (any edition, incl. standalone Build Tools)
rem --- Not found is NOT fatal: everything except Whisper's triton kernels works without it.
if defined VCVARS if not exist "%VCVARS%" (
    echo [run.bat] VCVARS is set but does not exist: "%VCVARS%" - ignoring it.
    set "VCVARS="
)

if not defined VCVARS (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq delims=" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find VC\Auxiliary\Build\vcvars64.bat 2^>nul`) do set "VCVARS=%%i"
    )
)

if not defined VCVARS (
    for %%e in (BuildTools Community Professional Enterprise) do (
        if not defined VCVARS if exist "%ProgramFiles%\Microsoft Visual Studio\2022\%%e\VC\Auxiliary\Build\vcvars64.bat" (
            set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\%%e\VC\Auxiliary\Build\vcvars64.bat"
        )
    )
)

if defined VCVARS (
    call "!VCVARS!" >nul 2>&1
    if errorlevel 1 (
        echo [run.bat] WARNING: MSVC environment failed to initialize ^("!VCVARS!"^).
        echo [run.bat] Whisper auto-subs GPU kernels ^(triton^) may not work; everything else is fine.
    )
) else (
    echo [run.bat] WARNING: Visual Studio / Build Tools not found ^(no vcvars64.bat^).
    echo [run.bat] Whisper auto-subs GPU kernels ^(triton^) may not work; everything else is fine.
    echo [run.bat] Install "Visual Studio Build Tools" with the C++ workload, or set VCVARS.
)

cd /d "%~dp0"
"%PY%" -m dubbing.host %*
exit /b %ERRORLEVEL%
