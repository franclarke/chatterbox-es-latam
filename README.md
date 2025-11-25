# Chatterbox ES-LATAM

Adaptación de [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) para español latinoamericano mediante fine-tuning con LoRA.

## Tabla de Contenidos

- [Objetivo del Proyecto](#objetivo-del-proyecto)
- [Arquitectura LoRA](#arquitectura-lora)
- [Requisitos](#requisitos)
- [Entrenamiento Local](#entrenamiento-local)
- [Entrenamiento en Runpod](#entrenamiento-en-runpod)
- [Evaluación del Modelo](#evaluación-del-modelo)
- [Inferencia con el Modelo Mergeado](#inferencia-con-el-modelo-mergeado)
- [Roadmap](#roadmap)

---

## Objetivo del Proyecto

El objetivo principal de este proyecto es adaptar el modelo Chatterbox TTS al español latinoamericano, preservando las capacidades de clonación de voz y control de expresividad del modelo original.

### Metas específicas:

- **Pronunciación natural**: Lograr una pronunciación clara y natural del español latinoamericano
- **Preservación de identidad vocal**: Mantener la capacidad de clonar voces con pocos segundos de audio de referencia
- **Control de expresividad**: Conservar el control sobre la exageración emocional en la síntesis
- **Eficiencia**: Utilizar LoRA para un fine-tuning eficiente sin necesidad de reentrenar el modelo completo

---

## Arquitectura LoRA

### ¿Qué es LoRA?

**LoRA (Low-Rank Adaptation)** es una técnica de fine-tuning eficiente que introduce matrices de bajo rango entrenables en las capas del modelo, permitiendo adaptar modelos grandes con una fracción de los parámetros originales.

### Implementación en Chatterbox

```
┌─────────────────────────────────────────────────────────────┐
│                    Chatterbox TTS                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   T3 Model  │    │   S3 Model  │    │  Vocoder    │     │
│  │  (Encoder)  │───▶│  (Decoder)  │───▶│  (BigVGAN)  │     │
│  │             │    │             │    │             │     │
│  │  + LoRA     │    │  + LoRA     │    │  (frozen)   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Parámetros de LoRA recomendados

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `lora_rank` | 16-64 | Rango de las matrices LoRA |
| `lora_alpha` | 32-128 | Factor de escala |
| `lora_dropout` | 0.05-0.1 | Dropout para regularización |
| `target_modules` | `q_proj, v_proj, k_proj, o_proj` | Capas objetivo |

---

## Requisitos

### Hardware mínimo

- **GPU**: NVIDIA con 16GB+ VRAM (RTX 3090, RTX 4090, A100)
- **RAM**: 32GB
- **Almacenamiento**: 50GB libres

### Software

```bash
# Python 3.10+
python --version

# Dependencias principales
pip install torch torchaudio
pip install chatterbox-tts
pip install peft  # Para LoRA
pip install wandb  # Para logging (opcional)
```

### Estructura del dataset

```
dataset/
├── audio/
│   ├── speaker_001/
│   │   ├── audio_001.wav
│   │   ├── audio_002.wav
│   │   └── ...
│   └── speaker_002/
│       └── ...
├── metadata.csv
└── transcripts/
    ├── audio_001.txt
    └── ...
```

**Formato de `metadata.csv`:**
```csv
audio_path,transcript,speaker_id,duration
audio/speaker_001/audio_001.wav,"Hola, ¿cómo estás?",speaker_001,2.5
audio/speaker_001/audio_002.wav,"Buenos días, bienvenido.",speaker_001,3.1
```

---

## Entrenamiento Local

### 1. Preparar el entorno

```bash
# Clonar el repositorio
git clone https://github.com/franclarke/chatterbox-es-latam.git
cd chatterbox-es-latam

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar el entrenamiento

Crear archivo `config/train_config.yaml`:

```yaml
# Configuración de entrenamiento
model:
  base_model: "resemble-ai/chatterbox"
  lora_rank: 32
  lora_alpha: 64
  lora_dropout: 0.05
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"

training:
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1e-4
  num_epochs: 10
  warmup_steps: 500
  max_audio_length: 30  # segundos
  
data:
  train_path: "data/train"
  val_path: "data/val"
  
output:
  checkpoint_dir: "checkpoints/"
  save_every: 1000
  
logging:
  wandb_project: "chatterbox-es-latam"
  log_every: 100
```

### 3. Ejecutar el entrenamiento

```bash
# Entrenamiento básico
python train.py --config config/train_config.yaml

# Con múltiples GPUs (si disponible)
torchrun --nproc_per_node=2 train.py --config config/train_config.yaml

# Resumir desde checkpoint
python train.py --config config/train_config.yaml --resume checkpoints/step_5000
```

### 4. Monitorear el entrenamiento

```bash
# Iniciar TensorBoard
tensorboard --logdir logs/

# O usar Weights & Biases
wandb login
# Los logs se enviarán automáticamente si está configurado
```

---

## Entrenamiento en Runpod

### 1. Crear un Pod

1. Ir a [Runpod.io](https://runpod.io)
2. Seleccionar template: **PyTorch 2.0** o superior
3. GPU recomendada: **A100 40GB** o **RTX 4090**
4. Almacenamiento: **50GB+** en volumen persistente

### 2. Configurar el entorno en Runpod

```bash
# Conectar vía SSH o usar JupyterLab

# Clonar repositorio
git clone https://github.com/franclarke/chatterbox-es-latam.git
cd chatterbox-es-latam

# Instalar dependencias
pip install -r requirements.txt

# Descargar dataset (si está en almacenamiento externo)
# Opción 1: Desde S3
aws s3 cp s3://tu-bucket/dataset.tar.gz .
tar -xzf dataset.tar.gz

# Opción 2: Desde Google Drive
pip install gdown
gdown --id TU_ID_DE_ARCHIVO

# Opción 3: Desde Hugging Face
huggingface-cli download tu-usuario/tu-dataset --local-dir data/
```

### 3. Script de entrenamiento para Runpod

Crear `scripts/train_runpod.sh`:

```bash
#!/bin/bash

# Configuración de entorno
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="tu_api_key"

# Verificar GPU
nvidia-smi

# Activar entorno
cd /workspace/chatterbox-es-latam
source venv/bin/activate

# Ejecutar entrenamiento
python train.py \
    --config config/train_config.yaml \
    --output_dir /workspace/checkpoints \
    --wandb_project chatterbox-es-latam-runpod

# Guardar modelo final en volumen persistente
cp -r /workspace/checkpoints /runpod-volume/checkpoints_backup
```

### 4. Ejecutar en background (para sesiones largas)

```bash
# Usar screen o tmux para sesiones persistentes
screen -S training

# Dentro de screen
bash scripts/train_runpod.sh

# Desconectar: Ctrl+A, luego D
# Reconectar: screen -r training
```

### 5. Sincronizar checkpoints

```bash
# Subir checkpoints a la nube periódicamente
# En otro terminal o como cron job

# A Hugging Face
huggingface-cli upload tu-usuario/chatterbox-es-latam-checkpoints ./checkpoints

# A S3
aws s3 sync ./checkpoints s3://tu-bucket/checkpoints/
```

---

## Evaluación del Modelo

### Métricas de evaluación

| Métrica | Descripción | Objetivo |
|---------|-------------|----------|
| **WER** | Word Error Rate (ASR) | < 15% |
| **MOS** | Mean Opinion Score | > 3.5/5 |
| **Speaker Similarity** | Similitud con voz de referencia | > 0.85 |
| **RTF** | Real-Time Factor | < 1.0 |

### 1. Evaluación automática

```bash
# Ejecutar suite de evaluación
python evaluate.py \
    --checkpoint checkpoints/best_model \
    --test_data data/test \
    --output_dir results/

# Evaluar solo WER (con Whisper)
python evaluate.py --metric wer --checkpoint checkpoints/best_model

# Evaluar similitud de speaker
python evaluate.py --metric speaker_sim --checkpoint checkpoints/best_model
```

### 2. Script de evaluación

```python
# evaluate.py
from chatterbox import ChatterboxModel
from peft import PeftModel
import torchaudio
import whisper

def evaluate_wer(model, test_samples):
    """Evaluar Word Error Rate usando Whisper"""
    whisper_model = whisper.load_model("large-v3")
    
    total_wer = 0
    for sample in test_samples:
        # Generar audio
        audio = model.generate(sample["text"], sample["reference_audio"])
        
        # Transcribir con Whisper
        result = whisper_model.transcribe(audio)
        
        # Calcular WER
        wer = calculate_wer(sample["text"], result["text"])
        total_wer += wer
    
    return total_wer / len(test_samples)

def evaluate_speaker_similarity(model, test_samples):
    """Evaluar similitud de speaker usando embeddings"""
    from resemblyzer import VoiceEncoder
    encoder = VoiceEncoder()
    
    similarities = []
    for sample in test_samples:
        # Generar audio
        generated = model.generate(sample["text"], sample["reference_audio"])
        
        # Extraer embeddings
        ref_embedding = encoder.embed_utterance(sample["reference_audio"])
        gen_embedding = encoder.embed_utterance(generated)
        
        # Calcular similitud coseno
        similarity = np.dot(ref_embedding, gen_embedding)
        similarities.append(similarity)
    
    return np.mean(similarities)
```

### 3. Evaluación subjetiva (MOS)

Para evaluación MOS, generar muestras y evaluar manualmente:

```bash
# Generar muestras de prueba
python generate_samples.py \
    --checkpoint checkpoints/best_model \
    --texts data/test_sentences.txt \
    --output_dir evaluation_samples/

# Las muestras se guardan en evaluation_samples/
# Distribuir para evaluación humana
```

---

## Inferencia con el Modelo Mergeado

### 1. Mergear pesos LoRA con modelo base

```python
# merge_model.py
from chatterbox import ChatterboxModel
from peft import PeftModel

def merge_lora_weights(base_model_path, lora_path, output_path):
    """Mergear pesos LoRA con el modelo base"""
    
    # Cargar modelo base
    base_model = ChatterboxModel.from_pretrained(base_model_path)
    
    # Cargar adaptadores LoRA
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # Mergear pesos
    merged_model = model.merge_and_unload()
    
    # Guardar modelo mergeado
    merged_model.save_pretrained(output_path)
    
    print(f"Modelo mergeado guardado en: {output_path}")
    return merged_model

# Uso
merge_lora_weights(
    base_model_path="resemble-ai/chatterbox",
    lora_path="checkpoints/best_model",
    output_path="models/chatterbox-es-latam-merged"
)
```

### 2. Cargar modelo para inferencia

```python
# inference.py
from chatterbox import ChatterboxModel
import torchaudio

# Cargar modelo mergeado
model = ChatterboxModel.from_pretrained("models/chatterbox-es-latam-merged")
model.eval()

# Opcional: mover a GPU
model = model.to("cuda")

def generate_speech(text, reference_audio_path, output_path, exaggeration=0.5):
    """
    Generar audio a partir de texto.
    
    Args:
        text: Texto a sintetizar
        reference_audio_path: Audio de referencia para clonar voz
        exaggeration: Control de expresividad (0.0 - 1.0)
        output_path: Ruta para guardar el audio generado
    """
    # Cargar audio de referencia
    reference_audio, sr = torchaudio.load(reference_audio_path)
    
    # Generar audio
    with torch.no_grad():
        generated_audio = model.generate(
            text=text,
            audio_prompt=reference_audio,
            exaggeration=exaggeration
        )
    
    # Guardar audio
    torchaudio.save(output_path, generated_audio, sample_rate=24000)
    
    return output_path

# Ejemplo de uso
generate_speech(
    text="Hola, ¿cómo estás? Bienvenido a Chatterbox en español.",
    reference_audio_path="samples/reference_voice.wav",
    output_path="output/generated_speech.wav",
    exaggeration=0.5
)
```

### 3. Inferencia por lotes

```python
# batch_inference.py
import os
from pathlib import Path

def batch_generate(model, texts_file, reference_audio, output_dir):
    """Generar múltiples audios desde un archivo de textos"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(texts_file, "r", encoding="utf-8") as f:
        texts = f.readlines()
    
    for i, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue
            
        output_path = output_dir / f"audio_{i:04d}.wav"
        generate_speech(
            text=text,
            reference_audio_path=reference_audio,
            output_path=str(output_path)
        )
        print(f"Generado: {output_path}")

# Uso
batch_generate(
    model=model,
    texts_file="data/texts_to_generate.txt",
    reference_audio="samples/reference.wav",
    output_dir="output/batch/"
)
```

### 4. API simple con FastAPI

```python
# api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import tempfile
import uvicorn

app = FastAPI(title="Chatterbox ES-LATAM API")

# Cargar modelo al inicio
model = ChatterboxModel.from_pretrained("models/chatterbox-es-latam-merged")
model.eval().to("cuda")

@app.post("/generate")
async def generate_audio(
    text: str,
    reference: UploadFile = File(...),
    exaggeration: float = 0.5
):
    """Endpoint para generar audio"""
    
    # Guardar audio de referencia temporalmente
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
        tmp_ref.write(await reference.read())
        ref_path = tmp_ref.name
    
    # Generar audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        output_path = tmp_out.name
    
    generate_speech(text, ref_path, output_path, exaggeration)
    
    return FileResponse(output_path, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Roadmap

### Fase 1: Fundamentos (Actual)
- [x] Configuración del repositorio
- [x] Documentación inicial
- [ ] Recolección de dataset español LATAM
- [ ] Scripts de preprocesamiento de audio
- [ ] Configuración de pipeline de entrenamiento

### Fase 2: Entrenamiento v1
- [ ] Fine-tuning inicial con LoRA
- [ ] Evaluación de baseline
- [ ] Ajuste de hiperparámetros
- [ ] Publicación de modelo v0.1

### Fase 3: Mejoras
- [ ] Ampliar dataset con más variedad de acentos LATAM
- [ ] Optimizar para diferentes acentos regionales:
  - [ ] Mexicano
  - [ ] Argentino
  - [ ] Colombiano
  - [ ] Chileno
  - [ ] Peruano
- [ ] Mejorar naturalidad y expresividad

### Fase 4: Producción
- [ ] Optimización de inferencia (cuantización, ONNX)
- [ ] API REST para producción
- [ ] Integración con servicios de streaming
- [ ] Documentación de API completa

### Fase 5: Comunidad
- [ ] Publicar modelo en Hugging Face
- [ ] Crear demos interactivos (Gradio/Spaces)
- [ ] Tutoriales y notebooks de ejemplo
- [ ] Recolección de feedback de usuarios

---

## Contribuir

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

---

## Agradecimientos

- [Resemble AI](https://resemble.ai/) por el modelo Chatterbox original
- [Hugging Face](https://huggingface.co/) por PEFT y la infraestructura de modelos
- La comunidad de código abierto de TTS

---

## Contacto

Para preguntas o sugerencias, abre un issue en el repositorio o contacta al equipo de desarrollo.