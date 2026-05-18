@echo off
chcp 65001 >nul
echo ================================================
echo  INSTALADOR - Organizador de Fotos com IA Local
echo ================================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado. Instale o Python 3.10+ de python.org
    pause
    exit /b 1
)

echo [1/5] Atualizando pip...
python -m pip install --upgrade pip

echo.
echo [2/5] Instalando PyTorch com CUDA 12.1 (pode demorar)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3/5] Instalando demais bibliotecas Python...
pip install -r requirements.txt

echo.
echo [4/5] Verificando Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo Ollama nao encontrado. Baixe e instale de: https://ollama.com/download/windows
    echo Depois rode este .bat novamente.
    pause
    exit /b 1
) else (
    echo Ollama detectado.
)

echo.
echo [5/5] Baixando modelo Llava 7B (cerca de 4GB)...
ollama pull llava:7b

echo.
echo ================================================
echo  INSTALACAO CONCLUIDA!
echo ================================================
echo.
echo Proximos passos:
echo   1. Edite o arquivo config.py com seus caminhos
echo   2. Rode: python 2_analisar.py
echo.
pause
