<p align="center">
  <a href="https://github.com/Hangry-Labs/Qwen3-ASR-STT">
    <img src="assets/qwen3_asr_logo_horizontal.webp" alt="Hangry Labs Qwen3-ASR-STT ロゴ" width="900">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.nb.md">Norsk bokmål</a> ·
  <a href="README.pl.md">Polski</a> ·
  <strong>日本語</strong> ·
  <a href="README.zh.md">简体中文</a> ·
  <a href="README.es.md">Español</a>
</p>

# Hangry Labs Qwen3-ASR-STT

ローカルのブラウザー UI と OpenAI 互換の文字起こし API を備えた、Docker ファーストの Qwen3-ASR パッケージです。

この Hangry Labs 版はローカル推論向けです。コンテナーを 1 つ起動するだけで、UI または API から、音声を外部サービスへ送信せずに文字起こしできます。

## このプロジェクトでできること

- ファイルのアップロード、録音、マイク音声のリアルタイム文字起こしに対応したローカル UI
- OpenAI 互換の `/v1/audio/transcriptions` エンドポイント
- 状態確認、共有パスまたは HTTP(S) URL からの文字起こし、システム制御に対応した `/mcp` のオプション Streamable HTTP MCP サーバー
- 多言語文字起こしと自動言語検出
- 単語およびセグメント単位のオプションのタイムスタンプ
- GPU モニタリングと永続化される設定
- ダウンロード後はオフラインで利用できるフル Docker イメージ
- モデルを永続ボリュームに保存する小容量 Docker イメージ
- vLLM による高速 GPU 推論と、既定の Qwen3-ASR 0.6B モデル

## クイックスタート

Docker と対応する NVIDIA GPU が必要です。フルイメージを起動します。

```bash
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all -e CUDA_VISIBLE_DEVICES=0 -v qwen3_asr_stt_data:/app/persistent hangrylabs/qwen3-asr-stt:latest
```

コマンドは 1 行なので、Bash、PowerShell、Windows コマンドプロンプトへそのまま貼り付けられます。名前付きボリュームは Docker によって自動作成されます。

起動後、次の URL を開きます。

- ブラウザー UI: [http://localhost:8000](http://localhost:8000)
- API ドキュメント: [http://localhost:8000/docs](http://localhost:8000/docs)
- MCP サーバー: [http://localhost:8000/mcp](http://localhost:8000/mcp)

最初にシステムタブで **MCP接続** を有効にし、信頼できるプライベートネットワーク内でのみ使用してください。

フル版の `latest` イメージにはモデルが含まれており、イメージの取得後はオフラインで使用できます。既定のモデルは `Qwen/Qwen3-ASR-0.6B-hf` です。

## 詳細情報

[日本語の製品説明とインストールガイド](https://hangrylabs.app/ja/software/qwen3-asr-stt)をご覧ください。完全な技術リファレンスは[英語版 README](README.md)で管理されています。
