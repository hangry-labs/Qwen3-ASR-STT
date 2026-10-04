<p align="center">
  <a href="https://github.com/Hangry-Labs/Qwen3-ASR-STT">
    <img src="assets/qwen3_asr_logo_horizontal.webp" alt="Logotipo de Hangry Labs Qwen3-ASR-STT" width="900">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.nb.md">Norsk bokmål</a> ·
  <a href="README.pl.md">Polski</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.zh.md">简体中文</a> ·
  <strong>Español</strong>
</p>

# Hangry Labs Qwen3-ASR-STT

Distribución de Qwen3-ASR orientada a Docker, con una interfaz web local y una API de transcripción compatible con OpenAI.

Esta versión de Hangry Labs está diseñada para inferencia local. Inicia un solo contenedor, abre la interfaz o llama a la API y transcribe voz sin enviar el audio a un servicio externo.

## Qué ofrece el proyecto

- Interfaz web local para subir archivos, grabar y transcribir el micrófono en tiempo real
- Endpoint `/v1/audio/transcriptions` compatible con OpenAI
- Transcripción multilingüe y detección automática del idioma
- Marcas de tiempo opcionales por palabra y segmento
- Supervisión de la GPU y ajustes persistentes
- Imagen Docker completa que funciona sin conexión después de descargarla
- Imagen Docker reducida que conserva los modelos en un volumen persistente
- Inferencia GPU acelerada con vLLM y Qwen3-ASR 0.6B como modelo predeterminado

## Inicio rápido

Necesitas Docker y una GPU NVIDIA compatible. Ejecuta la imagen completa:

```bash
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all -e CUDA_VISIBLE_DEVICES=0 -v qwen3_asr_stt_data:/app/persistent hangrylabs/qwen3-asr-stt:latest
```

El comando está en una sola línea y se puede pegar directamente en Bash, PowerShell o el Símbolo del sistema de Windows. Docker crea automáticamente el volumen con nombre.

Después, abre:

- Interfaz web: [http://localhost:8000](http://localhost:8000)
- Documentación de la API: [http://localhost:8000/docs](http://localhost:8000/docs)

La imagen completa `latest` incluye los modelos y puede utilizarse sin conexión después de descargarla. El modelo predeterminado es `Qwen/Qwen3-ASR-0.6B-hf`.

## Más información

Consulta la [página completa del producto y la guía de instalación en español](https://hangrylabs.app/es/software/qwen3-asr-stt). La referencia técnica completa se mantiene en el [README en inglés](README.md).
