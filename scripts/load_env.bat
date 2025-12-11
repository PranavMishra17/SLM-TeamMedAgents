@echo off
REM ============================================================================
REM Environment Variable Loader for Vertex AI Scripts
REM Reads .env file and sets environment variables for current session
REM ============================================================================

REM Check if .env file exists
if not exist "%~dp0..\.env" (
    echo ERROR: .env file not found!
    echo.
    echo Please create .env file from template:
    echo   1. Copy .env.vertex.example to .env
    echo   2. Fill in your Vertex AI configuration
    echo.
    echo See documentation/VERTEX_AI_SETUP.md for details
    exit /b 1
)

REM Read .env file and set variables
for /f "usebackq tokens=1,2 delims==" %%a in ("%~dp0..\.env") do (
    REM Skip comments and empty lines
    echo %%a | findstr /r "^#" >nul
    if errorlevel 1 (
        if not "%%a"=="" (
            if not "%%b"=="" (
                set "%%a=%%b"
            )
        )
    )
)

REM Verify required variables are set
if "%VERTEX_AI_ENDPOINT_ID%"=="" (
    echo ERROR: VERTEX_AI_ENDPOINT_ID not set in .env file
    echo Please edit .env and set VERTEX_AI_ENDPOINT_ID
    exit /b 1
)

if "%GOOGLE_CLOUD_PROJECT%"=="" (
    echo ERROR: GOOGLE_CLOUD_PROJECT not set in .env file
    echo Please edit .env and set GOOGLE_CLOUD_PROJECT
    exit /b 1
)

REM Set defaults for optional variables
if "%GOOGLE_CLOUD_LOCATION%"=="" (
    set GOOGLE_CLOUD_LOCATION=us-central1
)

if "%GOOGLE_GENAI_USE_VERTEXAI%"=="" (
    set GOOGLE_GENAI_USE_VERTEXAI=TRUE
)

REM Success - variables loaded
echo Loaded configuration from .env:
echo   GOOGLE_CLOUD_PROJECT=%GOOGLE_CLOUD_PROJECT%
echo   GOOGLE_CLOUD_LOCATION=%GOOGLE_CLOUD_LOCATION%
echo   VERTEX_AI_ENDPOINT_ID=%VERTEX_AI_ENDPOINT_ID%
echo   GOOGLE_GENAI_USE_VERTEXAI=%GOOGLE_GENAI_USE_VERTEXAI%
echo.

exit /b 0
