<p align="center">
  <a href="https://github.com/Hangry-Labs/Qwen3-ASR-STT">
    <img src="assets/qwen3_asr_logo_horizontal.webp" alt="Hangry Labs Qwen3-ASR-STT-logo" width="900">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <strong>Norsk bokmål</strong> ·
  <a href="README.pl.md">Polski</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.zh.md">简体中文</a> ·
  <a href="README.es.md">Español</a>
</p>

# Hangry Labs Qwen3-ASR-STT

Docker-først-pakking av Qwen3-ASR med et lokalt nettlesergrensesnitt og et OpenAI-kompatibelt API for transkripsjon.

Denne Hangry Labs-versjonen er laget for lokal inferens. Du kan starte én container, åpne grensesnittet eller bruke API-et og transkribere tale uten å sende lyd til en ekstern tjeneste.

## Dette får du

- Lokalt nettlesergrensesnitt for opplasting, opptak og direktetranskripsjon fra mikrofon
- OpenAI-kompatibelt endepunkt: `/v1/audio/transcriptions`
- Valgfri Streamable HTTP MCP-server på `/mcp` med helsestatus, transkripsjon fra delte filbaner eller HTTP(S)-URL-er og systemkontroller
- Flerspråklig transkripsjon og automatisk språkregistrering
- Valgfrie tidsstempler på ord- og segmentnivå
- GPU-overvåking og vedvarende innstillinger
- Komplett Docker-bilde for bruk uten nett etter nedlasting
- Lite Docker-bilde som lagrer modellene i et vedvarende volum
- Akselerert GPU-inferens med vLLM og Qwen3-ASR 0.6B som standard

## Hurtigstart

Du trenger Docker og en støttet NVIDIA-GPU. Kjør det komplette bildet:

```bash
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all -e CUDA_VISIBLE_DEVICES=0 -v qwen3_asr_stt_data:/app/persistent hangrylabs/qwen3-asr-stt:latest
```

Kommandoen står på én linje og kan limes direkte inn i Bash, PowerShell eller Windows Ledetekst. Docker oppretter det navngitte volumet automatisk.

Åpne deretter:

- Nettlesergrensesnitt: [http://localhost:8000](http://localhost:8000)
- API-dokumentasjon: [http://localhost:8000/docs](http://localhost:8000/docs)
- MCP-server: [http://localhost:8000/mcp](http://localhost:8000/mcp)

Aktiver først **MCP-tilkobling** i System-fanen, og bruk den bare på et betrodd privat nettverk.

Det komplette `latest`-bildet inkluderer modellene og kan brukes uten nett etter at bildet er lastet ned. Standardmodellen er `Qwen/Qwen3-ASR-0.6B-hf`.

## Mer informasjon

Les den [komplette norske produkt- og installasjonsveiledningen](https://hangrylabs.app/nb/software/qwen3-asr-stt). Den fullstendige tekniske referansen vedlikeholdes i den [engelske README-filen](README.md).
