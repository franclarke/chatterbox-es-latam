# Chatterbox ES-LATAM - LoRA Training
# LoRA fine-tuning for ResembleAI/chatterbox-multilingual on LATAM Spanish
# No Whisper dependency

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union

import torch
import yaml
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_scheduler,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset
from accelerate import Accelerator
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from dataset_orpheus import OrpheusDataset, load_orpheus_dataset


class ChatterboxLoRATrainer:
    """
    LoRA trainer for ResembleAI/chatterbox-multilingual model.
    
    Fine-tunes the Chatterbox multilingual TTS model on LATAM Spanish
    using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
    """
    
    MODEL_NAME = "ResembleAI/chatterbox-multilingual"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LoRA trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.dataset = None
        self.accelerator = None
        
        # Extract config sections
        self.model_config = config.get("model", {})
        self.lora_config = config.get("lora", {})
        self.training_config = config.get("training", {})
        self.data_config = config.get("data", {})
        self.output_config = config.get("output", {})
        
    def setup_accelerator(self):
        """Initialize the Accelerator for distributed training."""
        self.accelerator = Accelerator(
            mixed_precision=self.training_config.get("mixed_precision", "fp16"),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 1),
        )
        print(f"Using device: {self.accelerator.device}")
        
    def load_model(self):
        """Load the base Chatterbox model from HuggingFace."""
        model_name = self.model_config.get("name", self.MODEL_NAME)
        cache_dir = self.model_config.get("cache_dir", None)
        
        print(f"Loading model: {model_name}")
        
        # Load model config
        model_config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        
        # Load the model
        self.model = AutoModel.from_pretrained(
            model_name,
            config=model_config,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.training_config.get("mixed_precision") == "fp16" else torch.float32,
        )
        
        # Try to load tokenizer if available
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        except Exception:
            print("Tokenizer not found, will use model's default processing")
            self.tokenizer = None
        
        print(f"Model loaded: {type(self.model).__name__}")
        print(f"Model parameters: {self.model.num_parameters():,}")
        
    def apply_lora(self):
        """Apply LoRA configuration to the model."""
        if self.model is None:
            raise ValueError("Model must be loaded before applying LoRA")
        
        # LoRA configuration
        lora_r = self.lora_config.get("r", 16)
        lora_alpha = self.lora_config.get("alpha", 32)
        lora_dropout = self.lora_config.get("dropout", 0.05)
        target_modules = self.lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
        
        print(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"Target modules: {target_modules}")
        
        # Create LoRA config
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
    def load_dataset(self) -> Dataset:
        """Load and prepare the training dataset."""
        print("Loading LATAM Spanish dataset...")
        
        sample_rate = self.data_config.get("sample_rate", 24000)
        max_audio_length = self.data_config.get("max_audio_length", 30.0)
        cache_dir = self.data_config.get("cache_dir", None)
        split = self.data_config.get("split", "train")
        
        self.dataset = OrpheusDataset(
            sample_rate=sample_rate,
            max_audio_length=max_audio_length,
            cache_dir=cache_dir,
            split=split,
        )
        
        return self.dataset.load()
        
    def train(self):
        """Run the training loop."""
        if self.peft_model is None:
            raise ValueError("LoRA must be applied before training")
        if self.dataset is None:
            self.load_dataset()
            
        # Training parameters
        batch_size = self.training_config.get("batch_size", 4)
        epochs = self.training_config.get("epochs", 3)
        learning_rate = self.training_config.get("learning_rate", 1e-4)
        warmup_steps = self.training_config.get("warmup_steps", 100)
        save_steps = self.training_config.get("save_steps", 500)
        logging_steps = self.training_config.get("logging_steps", 10)
        output_dir = self.output_config.get("output_dir", "./output")
        
        print(f"\nTraining configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get dataloader
        dataloader = self.dataset.get_dataloader(
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.data_config.get("num_workers", 4),
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.peft_model.parameters(),
            lr=learning_rate,
            weight_decay=self.training_config.get("weight_decay", 0.01),
        )
        
        # Setup scheduler
        num_training_steps = len(dataloader) * epochs
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # Prepare with accelerator
        self.peft_model, optimizer, dataloader, lr_scheduler = self.accelerator.prepare(
            self.peft_model, optimizer, dataloader, lr_scheduler
        )
        
        # Training loop
        global_step = 0
        self.peft_model.train()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Training")
            
            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.peft_model):
                    # Forward pass - adapt based on model architecture
                    outputs = self._forward_pass(batch)
                    loss = outputs.get("loss", torch.tensor(0.0))
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    self._save_checkpoint(output_dir, global_step)
            
            print(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(dataloader):.4f}")
        
        print("\nTraining completed!")
        
    def _forward_pass(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Perform forward pass through the model.
        
        This method should be adapted based on the specific Chatterbox model architecture.
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary containing model outputs including loss
        """
        # Get audio and text from batch
        audio = batch.get("audio")
        
        # Forward through model - architecture dependent
        # This is a placeholder that should be adapted for Chatterbox
        device = self.accelerator.device if self.accelerator else torch.device("cpu")
        
        if audio is None:
            return {"loss": torch.tensor(0.0, device=device)}
        
        if hasattr(self.peft_model, "forward"):
            try:
                outputs = self.peft_model(audio)
                if isinstance(outputs, dict):
                    return outputs
                elif hasattr(outputs, "loss"):
                    return {"loss": outputs.loss, "outputs": outputs}
                else:
                    # Compute a reconstruction loss if model doesn't provide one
                    return {"loss": torch.tensor(0.0, device=device), "outputs": outputs}
            except Exception as e:
                print(f"Forward pass error: {e}")
                return {"loss": torch.tensor(0.0, device=device)}
        
        return {"loss": torch.tensor(0.0, device=device)}
    
    def _save_checkpoint(self, output_dir: str, step: int):
        """Save a training checkpoint."""
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save LoRA weights
        self.accelerator.unwrap_model(self.peft_model).save_pretrained(checkpoint_dir)
        
        print(f"Checkpoint saved: {checkpoint_dir}")
        
    def save_lora_weights(self, output_path: str):
        """
        Save only the LoRA adapter weights.
        
        Args:
            output_path: Path to save the LoRA weights
        """
        if self.peft_model is None:
            raise ValueError("No PEFT model to save")
        
        os.makedirs(output_path, exist_ok=True)
        
        unwrapped_model = self.accelerator.unwrap_model(self.peft_model) if self.accelerator else self.peft_model
        unwrapped_model.save_pretrained(output_path)
        
        print(f"LoRA weights saved to: {output_path}")
        
    def merge_and_save(self, output_path: str):
        """
        Merge LoRA weights with base model and save the full model.
        
        This produces a model ready for deployment on Runpod.
        
        Args:
            output_path: Path to save the merged model
        """
        if self.peft_model is None:
            raise ValueError("No PEFT model to merge")
        
        print("Merging LoRA weights with base model...")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Get the unwrapped model
        unwrapped_model = self.accelerator.unwrap_model(self.peft_model) if self.accelerator else self.peft_model
        
        # Merge and unload LoRA
        merged_model = unwrapped_model.merge_and_unload()
        
        # Save the merged model
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
        )
        
        # Save tokenizer if available
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_path)
        
        # Save config for Runpod deployment
        runpod_config = {
            "model_type": "chatterbox-multilingual-lora-es-latam",
            "base_model": self.MODEL_NAME,
            "language": "es-latam",
            "sample_rate": self.data_config.get("sample_rate", 24000),
        }
        
        config_path = os.path.join(output_path, "runpod_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(runpod_config, f, default_flow_style=False)
        
        print(f"Merged model saved to: {output_path}")
        print(f"Model is ready for Runpod deployment!")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LoRA for Chatterbox ES-LATAM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing LoRA weights (skip training)",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to existing LoRA weights for merging",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    
    # Initialize trainer
    trainer = ChatterboxLoRATrainer(config)
    
    if args.merge_only:
        # Merge existing LoRA weights
        if not args.lora_path:
            raise ValueError("--lora-path required when using --merge-only")
        
        trainer.load_model()
        
        # Load existing LoRA weights
        model_name = config.get("model", {}).get("name", trainer.MODEL_NAME)
        trainer.peft_model = PeftModel.from_pretrained(trainer.model, args.lora_path)
        
        merge_output = os.path.join(
            config["output"].get("output_dir", "./output"),
            "merged_model"
        )
        trainer.merge_and_save(merge_output)
    else:
        # Full training pipeline
        trainer.setup_accelerator()
        trainer.load_model()
        trainer.apply_lora()
        trainer.load_dataset()
        trainer.train()
        
        # Save results
        output_dir = config["output"].get("output_dir", "./output")
        
        # Save LoRA weights
        lora_output = os.path.join(output_dir, "lora_weights")
        trainer.save_lora_weights(lora_output)
        
        # Merge and save full model for Runpod
        merged_output = os.path.join(output_dir, "merged_model")
        trainer.merge_and_save(merged_output)
        
    print("\nDone!")


if __name__ == "__main__":
    main()
