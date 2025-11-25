#!/bin/bash

# Script para configurar el entorno virtual, instalar dependencias y ejecutar el entrenamiento
# Usage: ./run_training.sh [config_file]

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directorio del script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Archivo de configuración (por defecto: config.yaml)
CONFIG_FILE="${1:-config.yaml}"

# Nombre del entorno virtual
VENV_DIR="${SCRIPT_DIR}/venv"

# Archivo de requirements
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

echo -e "${GREEN}=== Iniciando script de entrenamiento ===${NC}"

# Verificar que Python está instalado
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 no está instalado${NC}"
    exit 1
fi

# Crear entorno virtual si no existe
if [ ! -d "${VENV_DIR}" ]; then
    echo -e "${YELLOW}Creando entorno virtual en ${VENV_DIR}...${NC}"
    python3 -m venv "${VENV_DIR}"
fi

# Activar el entorno virtual
echo -e "${GREEN}Activando entorno virtual...${NC}"
source "${VENV_DIR}/bin/activate"

# Verificar que el entorno virtual está activado
if [ -z "${VIRTUAL_ENV}" ]; then
    echo -e "${RED}Error: No se pudo activar el entorno virtual${NC}"
    exit 1
fi

echo -e "${GREEN}Entorno virtual activado: ${VIRTUAL_ENV}${NC}"

# Actualizar pip
echo -e "${YELLOW}Actualizando pip...${NC}"
pip install --upgrade pip

# Instalar requirements si el archivo existe
if [ -f "${REQUIREMENTS_FILE}" ]; then
    echo -e "${YELLOW}Instalando dependencias desde ${REQUIREMENTS_FILE}...${NC}"
    pip install -r "${REQUIREMENTS_FILE}"
else
    echo -e "${YELLOW}Advertencia: No se encontró ${REQUIREMENTS_FILE}${NC}"
fi

# Verificar que el archivo de configuración existe
if [ ! -f "${SCRIPT_DIR}/${CONFIG_FILE}" ]; then
    echo -e "${RED}Error: Archivo de configuración no encontrado: ${CONFIG_FILE}${NC}"
    echo -e "${YELLOW}Uso: ./run_training.sh [config_file]${NC}"
    deactivate
    exit 1
fi

# Verificar que el script de entrenamiento existe
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo -e "${RED}Error: Script de entrenamiento no encontrado: ${TRAIN_SCRIPT}${NC}"
    deactivate
    exit 1
fi

# Ejecutar el entrenamiento
echo -e "${GREEN}Ejecutando entrenamiento con configuración: ${CONFIG_FILE}${NC}"
python3 train.py --config "${SCRIPT_DIR}/${CONFIG_FILE}"

echo -e "${GREEN}=== Entrenamiento completado ===${NC}"

# Desactivar el entorno virtual
deactivate
