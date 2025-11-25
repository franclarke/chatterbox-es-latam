# Chatterbox ES-LATAM

Fine-tuning **ResembleAI/chatterbox-multilingual** with LoRA for LATAM Spanish TTS using the **GianDiego/latam-spanish-speech-orpheus-tts-24khz** dataset.

## Overview

This repository provides tools for training a LoRA (Low-Rank Adaptation) adapter on top of the Chatterbox multilingual TTS model to improve its performance for Latin American Spanish speech synthesis. The trained model can be deployed on Runpod or other GPU inference platforms.

## Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using PEFT
- **HuggingFace Integration**: Seamless loading of datasets and models from HuggingFace Hub
- **24kHz Audio Support**: Optimized for high-quality speech synthesis
- **Runpod Ready**: Merged models ready for deployment on Runpod
- **No Whisper Dependency**: Self-contained without ASR components

## Repository Structure

```
chatterbox-es-latam/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── configs/
│   └── training_config.yaml        # Training configuration
├── scripts/
│   └── run_train.sh                # Training shell script
└── src/
    ├── dataset_orpheus.py          # Dataset loading utilities
    └── lora_es_latam.py            # LoRA training implementation
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended: 24GB+ VRAM for full training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/franclarke/chatterbox-es-latam.git
cd chatterbox-es-latam
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the training script with default configuration:

```bash
./scripts/run_train.sh
```

### Custom Configuration

1. Edit `configs/training_config.yaml` to adjust training parameters:

```yaml
# Key parameters to customize
lora:
  r: 16                    # LoRA rank (higher = more capacity)
  alpha: 32                # LoRA scaling factor
  dropout: 0.05            # Dropout for regularization

training:
  batch_size: 4            # Reduce if OOM
  epochs: 3                # Number of training epochs
  learning_rate: 1.0e-4    # Learning rate
```

2. Run with custom config:

```bash
./scripts/run_train.sh --config path/to/custom_config.yaml
```

### Training Options

```bash
# Full training
./scripts/run_train.sh --config configs/training_config.yaml --output-dir ./output

# Merge existing LoRA weights with base model
./scripts/run_train.sh --merge-only --lora-path ./output/lora_weights

# Show help
./scripts/run_train.sh --help
```

### Python API

You can also use the Python API directly:

```python
from src.lora_es_latam import ChatterboxLoRATrainer, load_config

# Load configuration
config = load_config("configs/training_config.yaml")

# Initialize trainer
trainer = ChatterboxLoRATrainer(config)

# Setup and train
trainer.setup_accelerator()
trainer.load_model()
trainer.apply_lora()
trainer.load_dataset()
trainer.train()

# Save results
trainer.save_lora_weights("./output/lora_weights")
trainer.merge_and_save("./output/merged_model")
```

## Dataset

This project uses the [GianDiego/latam-spanish-speech-orpheus-tts-24khz](https://huggingface.co/datasets/GianDiego/latam-spanish-speech-orpheus-tts-24khz) dataset, which contains:

- LATAM Spanish speech samples
- 24kHz sample rate
- Text transcriptions for TTS training

## Model Architecture

The base model is [ResembleAI/chatterbox-multilingual](https://huggingface.co/ResembleAI/chatterbox-multilingual), a multilingual text-to-speech model. LoRA adapters are applied to the following layers:

- Query projection (`q_proj`)
- Key projection (`k_proj`)
- Value projection (`v_proj`)
- Output projection (`o_proj`)
- Gate projection (`gate_proj`)
- Up projection (`up_proj`)
- Down projection (`down_proj`)

## Output

After training, the following outputs are generated:

```
output/
├── lora_weights/           # LoRA adapter weights only
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── merged_model/           # Full merged model
│   ├── config.json
│   ├── model.safetensors
│   └── runpod_config.yaml
└── checkpoint-*/           # Training checkpoints
```

## Runpod Deployment

The merged model in `output/merged_model/` is ready for Runpod deployment:

1. Upload the `merged_model` directory to your Runpod storage
2. Use the model path in your Runpod inference script
3. The `runpod_config.yaml` contains deployment metadata

## Configuration Reference

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.name` | `ResembleAI/chatterbox-multilingual` | HuggingFace model ID |
| `model.cache_dir` | `null` | Local cache directory |

### LoRA Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora.r` | `16` | LoRA rank |
| `lora.alpha` | `32` | LoRA scaling factor |
| `lora.dropout` | `0.05` | Dropout rate |
| `lora.target_modules` | `[q_proj, v_proj, ...]` | Modules to adapt |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.batch_size` | `4` | Batch size per device |
| `training.epochs` | `3` | Number of epochs |
| `training.learning_rate` | `1e-4` | Learning rate |
| `training.mixed_precision` | `fp16` | Precision mode |

### Data Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.sample_rate` | `24000` | Audio sample rate |
| `data.max_audio_length` | `30.0` | Max audio length (seconds) |

## License

This project is provided for research and educational purposes. Please refer to the licenses of:
- [ResembleAI/chatterbox-multilingual](https://huggingface.co/ResembleAI/chatterbox-multilingual)
- [GianDiego/latam-spanish-speech-orpheus-tts-24khz](https://huggingface.co/datasets/GianDiego/latam-spanish-speech-orpheus-tts-24khz)

## Acknowledgements

- [ResembleAI](https://www.resemble.ai/) for the Chatterbox multilingual model
- [HuggingFace](https://huggingface.co/) for the datasets and transformers libraries
- [PEFT](https://github.com/huggingface/peft) for LoRA implementation