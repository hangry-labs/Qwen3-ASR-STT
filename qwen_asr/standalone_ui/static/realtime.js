import { encodeWav, formatTime } from './audio-utils.js'

export class RealtimeRecorder {
  constructor({ canvas, onState, onTranscript, onError }) {
    this.canvas = canvas
    this.onState = onState
    this.onTranscript = onTranscript
    this.onError = onError
    this.running = false
    this.buffers = []
    this.bufferedSamples = 0
    this.elapsedSamples = 0
    this.requestQueue = Promise.resolve()
    this.sessionId = null
  }

  async start(options) {
    if (this.running) return
    this.options = options
    this.onState('Requesting microphone')
    const constraints = options.deviceId ? { audio: { deviceId: { exact: options.deviceId } } } : { audio: true }
    this.stream = await navigator.mediaDevices.getUserMedia(constraints)
    const response = await fetch('/v1/realtime/transcriptions/sessions', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: options.model,
        language: options.language === 'Auto' ? undefined : options.language,
        prompt: options.prompt,
        temperature: 0,
        chunk_size_sec: options.chunkSize,
        unfixed_chunk_num: options.unfixedChunks,
        unfixed_token_num: options.unfixedTokens,
      }),
    })
    if (!response.ok) throw new Error(await response.text())
    this.sessionId = (await response.json()).id
    this.context = new AudioContext()
    this.source = this.context.createMediaStreamSource(this.stream)
    this.analyser = this.context.createAnalyser()
    this.analyser.fftSize = 2048
    this.processor = this.context.createScriptProcessor(4096, this.source.channelCount || 1, 1)
    this.source.connect(this.analyser)
    this.source.connect(this.processor)
    this.processor.connect(this.context.destination)
    this.processor.onaudioprocess = (event) => this.acceptSamples(event.inputBuffer.getChannelData(0))
    this.targetSamples = Math.max(1, Math.round(this.context.sampleRate * 1.0))
    this.running = true
    this.startedAt = performance.now()
    this.onState('Recording')
    this.draw()
  }

  acceptSamples(samples) {
    if (!this.running) return
    this.buffers.push(new Float32Array(samples))
    this.bufferedSamples += samples.length
    this.elapsedSamples += samples.length
    this.onState(`Recording ${formatTime(this.elapsedSamples / this.context.sampleRate)}`)
    if (this.bufferedSamples >= this.targetSamples) this.enqueueChunk()
  }

  enqueueChunk() {
    if (!this.bufferedSamples) return
    const merged = new Float32Array(this.bufferedSamples)
    let offset = 0
    for (const buffer of this.buffers) {
      merged.set(buffer, offset)
      offset += buffer.length
    }
    this.buffers = []
    this.bufferedSamples = 0
    const blob = encodeWav([merged], this.context.sampleRate)
    this.requestQueue = this.requestQueue.then(() => this.sendChunk(blob)).catch((error) => this.fail(error))
  }

  async sendChunk(blob) {
    if (!this.sessionId) return
    const form = new FormData()
    form.append('file', blob, 'chunk.wav')
    const response = await fetch(`/v1/realtime/transcriptions/sessions/${this.sessionId}/audio`, {
      method: 'POST',
      body: form,
    })
    if (!response.ok) throw new Error(await response.text())
    const payload = await response.json()
    this.onTranscript(payload.text || '', payload.language || '', false)
  }

  async stop() {
    if (!this.running) return
    this.running = false
    this.enqueueChunk()
    this.processor?.disconnect()
    this.source?.disconnect()
    this.stream?.getTracks().forEach((track) => track.stop())
    cancelAnimationFrame(this.animationFrame)
    await this.requestQueue
    if (this.sessionId) {
      const response = await fetch(`/v1/realtime/transcriptions/sessions/${this.sessionId}/finish`, { method: 'POST' })
      if (!response.ok) throw new Error(await response.text())
      const payload = await response.json()
      this.onTranscript(payload.text || '', payload.language || '', true)
    }
    await this.context?.close()
    this.sessionId = null
    this.onState('Finalized')
    this.clearCanvas()
  }

  async reset() {
    const sessionId = this.sessionId
    this.running = false
    this.processor?.disconnect()
    this.source?.disconnect()
    this.stream?.getTracks().forEach((track) => track.stop())
    cancelAnimationFrame(this.animationFrame)
    if (this.context && this.context.state !== 'closed') await this.context.close()
    if (sessionId) await fetch(`/v1/realtime/transcriptions/sessions/${sessionId}`, { method: 'DELETE' })
    this.sessionId = null
    this.buffers = []
    this.bufferedSamples = 0
    this.elapsedSamples = 0
    this.requestQueue = Promise.resolve()
    this.onTranscript('', '', false)
    this.onState('Ready')
    this.clearCanvas()
  }

  draw() {
    if (!this.running) return
    const context = this.canvas.getContext('2d')
    const values = new Uint8Array(this.analyser.fftSize)
    this.analyser.getByteTimeDomainData(values)
    const width = this.canvas.width = this.canvas.clientWidth * devicePixelRatio
    const height = this.canvas.height = this.canvas.clientHeight * devicePixelRatio
    context.clearRect(0, 0, width, height)
    context.strokeStyle = '#ff7a1a'
    context.lineWidth = 2 * devicePixelRatio
    context.beginPath()
    for (let index = 0; index < values.length; index += 1) {
      const x = index / (values.length - 1) * width
      const y = values[index] / 255 * height
      if (index === 0) context.moveTo(x, y)
      else context.lineTo(x, y)
    }
    context.stroke()
    this.animationFrame = requestAnimationFrame(() => this.draw())
  }

  clearCanvas() {
    const context = this.canvas.getContext('2d')
    context.clearRect(0, 0, this.canvas.width, this.canvas.height)
  }

  fail(error) {
    this.onError(error)
    this.reset().catch(() => {})
  }
}
