#!/bin/bash
# Chatterbox ES-LATAM Training Script
# Fine-tune ResembleAI/chatterbox-multilingual with LoRA on LATAM Spanish dataset

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
CONFIG_PATH="${PROJECT_ROOT}/configs/training_config.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/output"
MERGE_ONLY=false
LORA_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --merge-only)
            MERGE_ONLY=true
            shift
            ;;
        --lora-path)
            LORA_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH       Path to training config (default: configs/training_config.yaml)"
            echo "  --output-dir PATH   Output directory (default: ./output)"
            echo "  --merge-only        Only merge existing LoRA weights"
            echo "  --lora-path PATH    Path to LoRA weights for merging"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Chatterbox ES-LATAM LoRA Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Config file: ${CONFIG_PATH}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Merge only: ${MERGE_ONLY}"
if [ -n "$LORA_PATH" ]; then
    echo "  LoRA path: ${LORA_PATH}"
fi
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install requirements if needed
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Check CUDA availability
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"
echo ""

# Set environment variables for optimal training
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Build training command
TRAIN_CMD="python src/lora_es_latam.py --config ${CONFIG_PATH} --output-dir ${OUTPUT_DIR}"

if [ "$MERGE_ONLY" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --merge-only"
    if [ -n "$LORA_PATH" ]; then
        TRAIN_CMD="$TRAIN_CMD --lora-path ${LORA_PATH}"
    else
        echo "Error: --lora-path is required when using --merge-only"
        exit 1
    fi
fi

echo "Running training..."
echo "Command: ${TRAIN_CMD}"
echo ""

# Run training
eval $TRAIN_CMD

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  LoRA weights: ${OUTPUT_DIR}/lora_weights/"
echo "  Merged model: ${OUTPUT_DIR}/merged_model/"
echo ""
echo "To deploy on Runpod, upload the merged_model directory."
