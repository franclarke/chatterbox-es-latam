#!/usr/bin/env python3
"""
Main training script for fine-tuning ResembleAI/chatterbox-multilingual with LoRA.
Uses Orpheus training approach without Whisper.
"""

import torch
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType


def load_chatterbox_model(model_name: str = "ResembleAI/chatterbox-multilingual"):
    """
    Load the ResembleAI/chatterbox-multilingual model.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        The loaded model
    """
    print(f"Loading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
    )
    return model


def apply_lora(model, target_modules: list = None, lora_rank: int = 8, lora_alpha: int = 16):
    """
    Apply LoRA (Low-Rank Adaptation) to the model.
    
    Args:
        model: The base model to apply LoRA to
        target_modules: List of module names to apply LoRA to (attention layers by default)
        lora_rank: Rank of the low-rank matrices
        lora_alpha: Scaling factor for LoRA
        
    Returns:
        Model with LoRA adapters applied
    """
    if target_modules is None:
        # Target attention layers - these are the typical LoRA targets
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    
    print(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
    print(f"Target modules: {target_modules}")
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def setup_orpheus_training(model, dataset_path: str = None):
    """
    Set up Orpheus training configuration (without Whisper).
    
    Orpheus is a TTS training approach that uses direct audio-text pairs
    without relying on Whisper for automatic transcription.
    
    Args:
        model: The model to train
        dataset_path: Path to the Orpheus-compatible dataset
        
    Returns:
        Training configuration dict
    """
    training_config = {
        "model": model,
        "use_whisper": False,  # Explicitly disable Whisper
        "training_args": {
            "learning_rate": 1e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_epochs": 10,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "fp16": torch.cuda.is_available(),
        },
        "orpheus_config": {
            "use_direct_audio_text_pairs": True,
            "skip_asr_transcription": True,  # No Whisper transcription
            "audio_format": "wav",
            "sample_rate": 22050,
        },
    }
    
    if dataset_path:
        training_config["dataset_path"] = dataset_path
    
    return training_config


def main():
    """Main entry point for training setup."""
    print("=" * 60)
    print("Chatterbox Multilingual Fine-tuning Script (ES-LATAM)")
    print("=" * 60)
    
    # Step 1: Load the base model
    print("\n[Step 1] Loading ResembleAI/chatterbox-multilingual model...")
    model = load_chatterbox_model()
    
    # Step 2: Apply LoRA to the correct modules
    print("\n[Step 2] Applying LoRA to attention modules...")
    model = apply_lora(model)
    
    # Step 3: Set up Orpheus training (without Whisper)
    print("\n[Step 3] Setting up Orpheus training configuration (no Whisper)...")
    training_config = setup_orpheus_training(model)
    
    print("\n" + "=" * 60)
    print("Training configuration ready!")
    print("=" * 60)
    print("\nConfiguration summary:")
    print(f"  - Whisper enabled: {training_config['use_whisper']}")
    print(f"  - Direct audio-text pairs: {training_config['orpheus_config']['use_direct_audio_text_pairs']}")
    print(f"  - Skip ASR transcription: {training_config['orpheus_config']['skip_asr_transcription']}")
    print(f"  - FP16 training: {training_config['training_args']['fp16']}")
    
    return training_config


if __name__ == "__main__":
    main()
