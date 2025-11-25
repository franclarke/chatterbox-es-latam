# Chatterbox ES-LATAM - Dataset Orpheus
# Dataset loader for GianDiego/latam-spanish-speech-orpheus-tts-24khz
# No Whisper dependency

import os
from typing import Optional, Dict, Any, List

import torch
import torchaudio
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
import soundfile as sf


class OrpheusDataset:
    """
    Dataset class for loading and processing the latam-spanish-speech-orpheus-tts-24khz dataset.
    
    This dataset contains Spanish (Latin American) speech data for TTS training.
    """
    
    def __init__(
        self,
        dataset_name: str = "GianDiego/latam-spanish-speech-orpheus-tts-24khz",
        split: str = "train",
        sample_rate: int = 24000,
        max_audio_length: Optional[float] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the Orpheus dataset loader.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load ('train', 'test', 'validation')
            sample_rate: Target sample rate for audio (default: 24000 Hz)
            max_audio_length: Maximum audio length in seconds (None = no limit)
            cache_dir: Directory for caching the dataset
        """
        self.dataset_name = dataset_name
        self.split = split
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.cache_dir = cache_dir
        self.dataset = None
        
    def load(self) -> Dataset:
        """
        Load the dataset from HuggingFace.
        
        Returns:
            Loaded dataset
        """
        print(f"Loading dataset: {self.dataset_name}")
        
        self.dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        
        print(f"Dataset loaded with {len(self.dataset)} samples")
        return self.dataset
    
    def preprocess_audio(self, audio_data: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess audio data to target sample rate.
        
        Args:
            audio_data: Dictionary containing 'array' and 'sampling_rate'
            
        Returns:
            Preprocessed audio tensor
        """
        audio_array = audio_data["array"]
        original_sr = audio_data["sampling_rate"]
        
        # Convert to tensor
        if not isinstance(audio_array, torch.Tensor):
            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        else:
            audio_tensor = audio_array.float()
        
        # Ensure 2D tensor (channels, samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Resample if necessary
        if original_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr,
                new_freq=self.sample_rate
            )
            audio_tensor = resampler(audio_tensor)
        
        # Trim to max length if specified
        if self.max_audio_length is not None:
            max_samples = int(self.max_audio_length * self.sample_rate)
            if audio_tensor.shape[-1] > max_samples:
                audio_tensor = audio_tensor[..., :max_samples]
        
        return audio_tensor.squeeze(0)
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sample from the dataset.
        
        Args:
            sample: Raw sample from the dataset
            
        Returns:
            Processed sample with normalized audio
        """
        processed = {}
        
        # Process audio if present
        if "audio" in sample:
            processed["audio"] = self.preprocess_audio(sample["audio"])
            processed["audio_length"] = processed["audio"].shape[-1] / self.sample_rate
        
        # Copy text/transcript fields
        for key in ["text", "transcript", "transcription", "sentence"]:
            if key in sample:
                processed["text"] = sample[key]
                break
        
        # Copy speaker ID if present
        if "speaker_id" in sample:
            processed["speaker_id"] = sample["speaker_id"]
        
        return processed
    
    def get_dataloader(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        collate_fn: Optional[callable] = None,
    ) -> DataLoader:
        """
        Create a DataLoader for the dataset.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            collate_fn: Custom collate function
            
        Returns:
            PyTorch DataLoader
        """
        if self.dataset is None:
            self.load()
        
        if collate_fn is None:
            collate_fn = self.default_collate_fn
        
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    def default_collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Default collate function for batching samples.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched dictionary
        """
        processed_batch = [self.process_sample(sample) for sample in batch]
        
        # Pad audio to same length
        audio_list = [s["audio"] for s in processed_batch if "audio" in s]
        if audio_list:
            max_len = max(a.shape[-1] for a in audio_list)
            padded_audio = []
            audio_lengths = []
            for audio in audio_list:
                pad_len = max_len - audio.shape[-1]
                if pad_len > 0:
                    padded = torch.nn.functional.pad(audio, (0, pad_len))
                else:
                    padded = audio
                padded_audio.append(padded)
                audio_lengths.append(audio.shape[-1])
            
            batched = {
                "audio": torch.stack(padded_audio),
                "audio_lengths": torch.tensor(audio_lengths),
            }
        else:
            batched = {}
        
        # Collect texts
        texts = [s.get("text", "") for s in processed_batch]
        batched["text"] = texts
        
        return batched


def load_orpheus_dataset(
    split: str = "train",
    sample_rate: int = 24000,
    max_audio_length: Optional[float] = None,
    cache_dir: Optional[str] = None,
) -> OrpheusDataset:
    """
    Convenience function to load the Orpheus LATAM Spanish dataset.
    
    Args:
        split: Dataset split
        sample_rate: Target sample rate
        max_audio_length: Maximum audio length in seconds
        cache_dir: Cache directory
        
    Returns:
        Initialized and loaded OrpheusDataset
    """
    dataset = OrpheusDataset(
        split=split,
        sample_rate=sample_rate,
        max_audio_length=max_audio_length,
        cache_dir=cache_dir,
    )
    dataset.load()
    return dataset


if __name__ == "__main__":
    # Test dataset loading
    print("Testing Orpheus dataset loader...")
    
    # Initialize dataset
    dataset = OrpheusDataset(
        sample_rate=24000,
        max_audio_length=30.0,
    )
    
    # Load dataset
    ds = dataset.load()
    
    # Print dataset info
    print(f"\nDataset features: {ds.features}")
    print(f"Number of samples: {len(ds)}")
    
    # Test single sample processing
    if len(ds) > 0:
        sample = ds[0]
        processed = dataset.process_sample(sample)
        print(f"\nProcessed sample keys: {processed.keys()}")
        if "audio" in processed:
            print(f"Audio shape: {processed['audio'].shape}")
            print(f"Audio length: {processed['audio_length']:.2f}s")
        if "text" in processed:
            print(f"Text: {processed['text'][:100]}...")
