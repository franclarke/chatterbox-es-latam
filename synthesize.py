#!/usr/bin/env python3
"""
Script para sintetizar audio usando Chatterbox TTS.
Carga el modelo mergeado y permite sintetizar audio desde consola usando texto libre,
guardando un archivo WAV a 24kHz.
"""

import argparse
import sys
from pathlib import Path


def detect_device() -> str:
    """Detecta automáticamente el mejor dispositivo disponible."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: str | None, device: str):
    """
    Carga el modelo Chatterbox TTS.

    Args:
        model_path: Ruta al modelo local o None para usar el modelo de HuggingFace.
        device: Dispositivo donde cargar el modelo (cuda, mps, cpu).

    Returns:
        Modelo Chatterbox TTS cargado.
    """
    from chatterbox.tts import ChatterboxTTS

    if model_path:
        print(f"Cargando modelo desde: {model_path}")
        return ChatterboxTTS.from_local(model_path, device)
    else:
        print("Cargando modelo desde HuggingFace...")
        return ChatterboxTTS.from_pretrained(device)


def synthesize(
    model,
    text: str,
    audio_prompt: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
):
    """
    Sintetiza audio a partir del texto proporcionado.

    Args:
        model: Modelo Chatterbox TTS.
        text: Texto a sintetizar.
        audio_prompt: Ruta opcional a un archivo de audio de referencia para clonar la voz.
        exaggeration: Nivel de exageración emocional (0.0-1.0).
        cfg_weight: Peso de CFG para la generación.
        temperature: Temperatura para el muestreo.

    Returns:
        Tensor de audio generado.
    """
    return model.generate(
        text=text,
        audio_prompt_path=audio_prompt,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sintetiza audio desde texto usando Chatterbox TTS (24kHz WAV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Sintetizar texto simple
  python synthesize.py "Hola, esto es una prueba de síntesis de voz."

  # Usar un modelo local
  python synthesize.py --model ./mi_modelo "Texto a sintetizar"

  # Especificar archivo de salida
  python synthesize.py -o salida.wav "Texto a sintetizar"

  # Clonar voz usando audio de referencia
  python synthesize.py --audio-prompt referencia.wav "Texto con voz clonada"
        """,
    )

    parser.add_argument(
        "text",
        type=str,
        help="Texto a sintetizar",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.wav",
        help="Ruta del archivo WAV de salida (default: output.wav)",
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Ruta al modelo local (usa HuggingFace si no se especifica)",
    )

    parser.add_argument(
        "-d", "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Dispositivo a usar (auto-detecta si no se especifica)",
    )

    parser.add_argument(
        "-p", "--audio-prompt",
        type=str,
        default=None,
        help="Archivo de audio de referencia para clonar la voz",
    )

    parser.add_argument(
        "-e", "--exaggeration",
        type=float,
        default=0.5,
        help="Nivel de exageración emocional (0.0-1.0, default: 0.5)",
    )

    parser.add_argument(
        "-c", "--cfg-weight",
        type=float,
        default=0.5,
        help="Peso de CFG para la generación (default: 0.5)",
    )

    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.8,
        help="Temperatura para el muestreo (default: 0.8)",
    )

    args = parser.parse_args()

    # Validar texto
    if not args.text.strip():
        print("Error: El texto no puede estar vacío.", file=sys.stderr)
        sys.exit(1)

    # Detectar o usar dispositivo especificado
    device = args.device if args.device else detect_device()
    print(f"Usando dispositivo: {device}")

    # Cargar modelo
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Sintetizando: \"{args.text}\"")

    # Sintetizar audio
    try:
        wav = synthesize(
            model=model,
            text=args.text,
            audio_prompt=args.audio_prompt,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
        )
    except Exception as e:
        print(f"Error durante la síntesis: {e}", file=sys.stderr)
        sys.exit(1)

    # Guardar archivo WAV a 24kHz
    output_path = Path(args.output)
    try:
        import torchaudio as ta

        ta.save(str(output_path), wav, model.sr)
        print(f"Audio guardado en: {output_path} ({model.sr}Hz)")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
