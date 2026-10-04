<p align="center">
  <a href="https://github.com/Hangry-Labs/Qwen3-ASR-STT">
    <img src="assets/qwen3_asr_logo_horizontal.webp" alt="Hangry Labs Qwen3-ASR-STT 标志" width="900">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.nb.md">Norsk bokmål</a> ·
  <a href="README.pl.md">Polski</a> ·
  <a href="README.ja.md">日本語</a> ·
  <strong>简体中文</strong> ·
  <a href="README.es.md">Español</a>
</p>

# Hangry Labs Qwen3-ASR-STT

面向 Docker 的 Qwen3-ASR 封装，提供本地浏览器界面和兼容 OpenAI 的转录 API。

此 Hangry Labs 版本专为本地推理设计。只需启动一个容器，即可通过界面或 API 转录语音，无需将音频发送到外部托管服务。

## 项目功能

- 用于上传、录音和麦克风实时转录的本地浏览器界面
- 兼容 OpenAI 的 `/v1/audio/transcriptions` 端点
- 多语言转录和自动语言检测
- 可选的单词级和片段级时间戳
- GPU 监控和持久化设置
- 下载后可离线运行的完整 Docker 镜像
- 将模型保存在持久卷中的精简 Docker 镜像
- 基于 vLLM 的 GPU 加速推理，默认使用 Qwen3-ASR 0.6B 模型

## 快速开始

你需要 Docker 和受支持的 NVIDIA GPU。运行完整镜像：

```bash
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all -e CUDA_VISIBLE_DEVICES=0 -v qwen3_asr_stt_data:/app/persistent hangrylabs/qwen3-asr-stt:latest
```

命令保持在一行，可直接粘贴到 Bash、PowerShell 或 Windows 命令提示符中。Docker 会自动创建命名卷。

启动后打开：

- 浏览器界面：[http://localhost:8000](http://localhost:8000)
- API 文档：[http://localhost:8000/docs](http://localhost:8000/docs)

完整的 `latest` 镜像已包含模型，下载镜像后即可离线使用。默认模型为 `Qwen/Qwen3-ASR-0.6B-hf`。

## 更多信息

请阅读[完整的中文产品介绍和安装指南](https://hangrylabs.app/zh/software/qwen3-asr-stt)。完整技术参考资料由[英文 README](README.md)维护。
