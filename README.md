# chatterbox-es-latam

Script para sintetizar audio desde texto usando Chatterbox TTS, generando archivos WAV a 24kHz.

## Instalación

```bash
pip install -r requirements.txt
```

**Nota:** Requiere Python 3.10 o superior. Recomendado Python 3.11.

## Uso

### Síntesis básica

```bash
python synthesize.py "Hola, esto es una prueba de síntesis de voz."
```

### Opciones

```bash
python synthesize.py --help
```

| Opción | Descripción |
|--------|-------------|
| `text` | Texto a sintetizar (requerido) |
| `-o, --output` | Archivo de salida (default: output.wav) |
| `-m, --model` | Ruta al modelo local (usa HuggingFace si no se especifica) |
| `-d, --device` | Dispositivo: cuda, mps, cpu (auto-detecta) |
| `-p, --audio-prompt` | Audio de referencia para clonar voz |
| `-e, --exaggeration` | Nivel de emoción 0.0-1.0 (default: 0.5) |
| `-c, --cfg-weight` | Peso CFG (default: 0.5) |
| `-t, --temperature` | Temperatura de muestreo (default: 0.8) |

### Ejemplos

```bash
# Guardar en archivo específico
python synthesize.py -o mi_audio.wav "Texto a sintetizar"

# Usar modelo local
python synthesize.py --model ./mi_modelo_mergeado "Texto a sintetizar"

# Clonar voz con audio de referencia
python synthesize.py --audio-prompt voz_referencia.wav "Texto con voz clonada"

# Ajustar parámetros de generación
python synthesize.py -e 0.7 -c 0.3 "¡Esto es muy emocionante!"
```

## Características

- Carga modelos locales (mergeados/fine-tuned) o desde HuggingFace
- Genera audio WAV a 24kHz
- Soporte para clonación de voz con audio de referencia
- Control de expresividad emocional
- Auto-detección de GPU (CUDA/MPS)

## Requisitos

- Python 3.10+
- PyTorch con soporte CUDA (recomendado) o MPS para mejor rendimiento
- ~4GB VRAM para inferencia con GPU