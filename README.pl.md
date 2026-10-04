<p align="center">
  <a href="https://github.com/Hangry-Labs/Qwen3-ASR-STT">
    <img src="assets/qwen3_asr_logo_horizontal.webp" alt="Logo Hangry Labs Qwen3-ASR-STT" width="900">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.nb.md">Norsk bokmål</a> ·
  <strong>Polski</strong> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.zh.md">简体中文</a> ·
  <a href="README.es.md">Español</a>
</p>

# Hangry Labs Qwen3-ASR-STT

Gotowy do użycia w Dockerze pakiet Qwen3-ASR z lokalnym interfejsem przeglądarkowym oraz zgodnym z OpenAI API do transkrypcji.

Ta wersja Hangry Labs została przygotowana do lokalnego wnioskowania. Uruchom jeden kontener, otwórz interfejs lub wywołaj API i transkrybuj mowę bez wysyłania nagrań do zewnętrznej usługi.

## Co oferuje projekt

- Lokalny interfejs do przesyłania plików, nagrywania i transkrypcji mikrofonu w czasie rzeczywistym
- Zgodny z OpenAI punkt końcowy `/v1/audio/transcriptions`
- Wielojęzyczna transkrypcja i automatyczne rozpoznawanie języka
- Opcjonalne znaczniki czasu dla słów i segmentów
- Monitorowanie GPU i trwałe ustawienia
- Pełny obraz Docker działający offline po pobraniu
- Mały obraz Docker przechowujący modele w trwałym woluminie
- Przyspieszone wnioskowanie GPU z vLLM i domyślnym modelem Qwen3-ASR 0.6B

## Szybki start

Potrzebujesz Dockera i obsługiwanej karty NVIDIA. Uruchom pełny obraz:

```bash
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all -e CUDA_VISIBLE_DEVICES=0 -v qwen3_asr_stt_data:/app/persistent hangrylabs/qwen3-asr-stt:latest
```

Polecenie znajduje się w jednym wierszu i można je wkleić bezpośrednio do Bash, PowerShell lub Wiersza polecenia systemu Windows. Docker automatycznie utworzy nazwany wolumin.

Następnie otwórz:

- Interfejs przeglądarkowy: [http://localhost:8000](http://localhost:8000)
- Dokumentację API: [http://localhost:8000/docs](http://localhost:8000/docs)

Pełny obraz `latest` zawiera modele i po pobraniu może działać bez dostępu do sieci. Domyślnym modelem jest `Qwen/Qwen3-ASR-0.6B-hf`.

## Więcej informacji

Przeczytaj [pełny polski opis produktu i przewodnik instalacji](https://hangrylabs.app/pl/software/qwen3-asr-stt). Kompletna dokumentacja techniczna jest utrzymywana w [angielskim pliku README](README.md).
