```bash
#!/bin/bash

# Exit on error
set -e

echo "Setting up environment..."

# Create logs directory
mkdir -p logs

# Install Python 3.11 if not available
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    apt update && apt install -y python3.11 python3.11-venv python3.11-dev
fi

# Create virtual environment with Python 3.11
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with Python 3.11..."
    python3.11 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install "numpy>=1.26.0" cython setuptools wheel soundfile
# Fix pkuseg by building from source without pre-generated cpp files
git clone https://github.com/lancopku/pkuseg-python.git pkuseg_tmp
cd pkuseg_tmp
find . -name "*.cpp" -delete
find . -name "*.c" -delete
pip install --no-build-isolation .
cd ..
rm -rf pkuseg_tmp
pip install -r requirements.txt

# Run training with logging
echo "Starting training..."
# Use stdbuf to unbuffer output so it appears in log immediately
stdbuf -oL python -m src.lora_es_latam 2>&1 | tee logs/train_log.txt

echo "Training completed. Log saved to logs/train_log.txt"
