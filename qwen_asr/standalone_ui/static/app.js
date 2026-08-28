import { AudioEditor } from './audio-editor.js'
import { formatTime } from './audio-utils.js'
import { RealtimeRecorder } from './realtime.js'

const $ = (selector, root = document) => root.querySelector(selector)
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)]

const state = {
  source: 'upload',
  backendReady: false,
  model: 'qwen3-asr',
  timestampsAvailable: null,
}

const uploadEditor = new AudioEditor($('#upload-editor'), { label: 'Upload audio' })
const recordEditor = new AudioEditor($('#record-editor'), { label: 'Recorded audio' })
recordEditor.container.append($('#record-controls'))

function setStatus(message, tone = 'neutral') {
  const status = $('#global-status')
  status.textContent = message
  status.dataset.tone = tone
}

function showToast(message, tone = 'error') {
  const toast = $('#toast')
  toast.textContent = message
  toast.dataset.tone = tone
  toast.hidden = false
  clearTimeout(showToast.timer)
  showToast.timer = setTimeout(() => { toast.hidden = true }, 5000)
}

function errorMessage(error) {
  if (error instanceof Error) return error.message
  return String(error)
}

async function responseError(response) {
  const text = await response.text()
  try {
    const payload = JSON.parse(text)
    return payload.error?.message || payload.detail || text
  } catch {
    return text || `HTTP ${response.status}`
  }
}

function activateTab(name) {
  $$('.tab-button').forEach((button) => {
    const active = button.dataset.tab === name
    button.classList.toggle('active', active)
    button.setAttribute('aria-selected', String(active))
  })
  $$('.tab-panel').forEach((panel) => { panel.hidden = panel.dataset.panel !== name })
  if (name === 'api') refreshApiStatus()
  if (name === 'system') refreshSystem()
}

$$('.tab-button').forEach((button) => button.addEventListener('click', () => activateTab(button.dataset.tab)))

$$('[data-source]').forEach((button) => button.addEventListener('click', () => {
  state.source = button.dataset.source
  $$('[data-source]').forEach((item) => item.classList.toggle('active', item === button))
  $('#upload-source').hidden = state.source !== 'upload'
  $('#record-source').hidden = state.source !== 'record'
}))

async function refreshAudioDevices(select) {
  const current = select.value
  const devices = await AudioEditor.audioInputDevices()
  select.innerHTML = '<option value="">Default microphone</option>'
  devices.forEach((device, index) => {
    const option = document.createElement('option')
    option.value = device.deviceId
    option.textContent = device.label || `Microphone ${index + 1}`
    select.append(option)
  })
  if ([...select.options].some((option) => option.value === current)) select.value = current
}

const recordPlugin = recordEditor.attachRecorder({
  onStart: () => {
    $('#record-start').disabled = true
    $('#record-stop').disabled = false
    $('#record-state').textContent = 'Recording 0:00'
  },
  onProgress: (duration) => {
    $('#record-state').textContent = `Recording ${formatTime(duration / 1000)}`
  },
  onEnd: () => {
    $('#record-start').disabled = false
    $('#record-stop').disabled = true
    $('#record-state').textContent = 'Recording ready'
  },
})

$('#record-start').addEventListener('click', async () => {
  try {
    const deviceId = $('#record-device').value
    await recordPlugin.startRecording(deviceId ? { deviceId: { exact: deviceId } } : undefined)
    await refreshAudioDevices($('#record-device'))
  } catch (error) {
    $('#record-start').disabled = false
    $('#record-stop').disabled = true
    showToast(errorMessage(error))
  }
})

$('#record-stop').addEventListener('click', () => recordPlugin.stopRecording())
$('#record-device-refresh').addEventListener('click', async () => {
  try {
    await navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => stream.getTracks().forEach((track) => track.stop()))
    await refreshAudioDevices($('#record-device'))
  } catch (error) {
    showToast(errorMessage(error))
  }
})

async function loadExamples() {
  const response = await fetch('/examples')
  if (!response.ok) return
  const payload = await response.json()
  const select = $('#example-select')
  payload.examples.forEach((example) => {
    const option = document.createElement('option')
    option.value = example.url
    option.textContent = example.label
    option.dataset.language = example.language
    option.dataset.name = example.name
    select.append(option)
  })
}

$('#example-load').addEventListener('click', async () => {
  const option = $('#example-select').selectedOptions[0]
  if (!option?.value) return
  try {
    setStatus('Loading example')
    const response = await fetch(option.value)
    if (!response.ok) throw new Error(await responseError(response))
    await uploadEditor.load(await response.blob(), option.dataset.name)
    state.source = 'upload'
    $('[data-source="upload"]').click()
    setStatus('Example ready', 'success')
  } catch (error) {
    showToast(errorMessage(error))
  }
})

function selectedTimestampGranularities() {
  return $$('input[name="timestamp"]:checked').map((input) => input.value)
}

function updateTimestampAvailability(available) {
  state.timestampsAvailable = available
  const support = $('#timestamp-support')
  support.dataset.state = available ? 'available' : 'unavailable'
  support.textContent = available ? 'Aligner ready' : 'Aligner disabled'
  support.title = available
    ? 'Word and segment timestamps are available.'
    : 'This deployment was started without the forced aligner. Set QWEN_ASR_ENABLE_ALIGNER=1 when starting the container to enable timestamps.'
}

$$('input[name="timestamp"]').forEach((input) => input.addEventListener('change', () => {
  if (!input.checked || state.timestampsAvailable === true) return
  input.checked = false
  showToast(
    state.timestampsAvailable === false
      ? 'Timestamps are unavailable because this deployment has the forced aligner disabled. Start with QWEN_ASR_ENABLE_ALIGNER=1 to enable Word and Segment timestamps.'
      : 'Timestamp support is still being checked. Please try again when inference is ready.',
  )
}))

$('#transcribe-form').addEventListener('submit', async (event) => {
  event.preventDefault()
  const editor = state.source === 'upload' ? uploadEditor : recordEditor
  const file = editor.currentFile()
  if (!file) {
    showToast('Select or record audio before transcribing.')
    return
  }

  const button = $('#transcribe-button')
  const form = new FormData()
  form.append('file', file, file.name)
  form.append('model', state.model)
  form.append('temperature', '0')
  form.append('response_format', $('#response-format').value)
  form.append('prompt', $('#prompt').value)
  if ($('#language').value !== 'Auto') form.append('language', $('#language').value)
  selectedTimestampGranularities().forEach((value) => form.append('timestamp_granularities', value))

  button.disabled = true
  $('#transcript-output').textContent = ''
  $('#response-output').textContent = ''
  setStatus('Transcribing')
  const started = performance.now()
  try {
    const response = await fetch('/v1/audio/transcriptions', { method: 'POST', body: form })
    const responseText = await response.text()
    if (!response.ok) throw new Error(await responseError(new Response(responseText, { status: response.status })))
    let payload = responseText
    try { payload = JSON.parse(responseText) } catch {}
    $('#transcript-output').textContent = typeof payload === 'string' ? payload : payload.text || ''
    $('#response-output').textContent = typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2)
    setStatus(`HTTP ${response.status} in ${((performance.now() - started) / 1000).toFixed(3)}s`, 'success')
  } catch (error) {
    setStatus('Transcription failed', 'error')
    showToast(errorMessage(error))
  } finally {
    button.disabled = false
  }
})

function sliderOutput(input) {
  const output = document.querySelector(`[data-output="${input.id}"]`)
  if (output) output.textContent = input.value
}

$$('input[type="range"][data-setting]').forEach((input) => {
  sliderOutput(input)
  input.addEventListener('input', () => sliderOutput(input))
})

const realtime = new RealtimeRecorder({
  canvas: $('#realtime-wave'),
  onState: (value) => {
    $('#realtime-state').textContent = value
    setStatus(value, value === 'Finalized' ? 'success' : 'neutral')
  },
  onTranscript: (text, language, final) => {
    $('#realtime-output').textContent = text
    $('#realtime-language').textContent = language || 'Auto detection'
    $('#realtime-final').hidden = !final
  },
  onError: (error) => showToast(errorMessage(error)),
})

$('#realtime-start').addEventListener('click', async () => {
  try {
    await realtime.start({
      deviceId: $('#realtime-device').value,
      model: state.model,
      language: $('#language').value,
      prompt: $('#prompt').value,
      chunkSize: Number($('#chunk-size').value),
      unfixedChunks: Number($('#unfixed-chunks').value),
      unfixedTokens: Number($('#unfixed-tokens').value),
    })
    $('#realtime-start').disabled = true
    $('#realtime-stop').disabled = false
    await refreshAudioDevices($('#realtime-device'))
  } catch (error) {
    showToast(errorMessage(error))
    await realtime.reset()
  }
})

$('#realtime-stop').addEventListener('click', async () => {
  $('#realtime-stop').disabled = true
  try {
    await realtime.stop()
  } catch (error) {
    showToast(errorMessage(error))
  } finally {
    $('#realtime-start').disabled = false
  }
})

$('#realtime-reset').addEventListener('click', async () => {
  await realtime.reset()
  $('#realtime-start').disabled = false
  $('#realtime-stop').disabled = true
})

$('#realtime-device-refresh').addEventListener('click', async () => {
  try {
    await navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => stream.getTracks().forEach((track) => track.stop()))
    await refreshAudioDevices($('#realtime-device'))
  } catch (error) {
    showToast(errorMessage(error))
  }
})

async function fetchJson(path) {
  const response = await fetch(path)
  const text = await response.text()
  if (!response.ok) throw new Error(await responseError(new Response(text, { status: response.status })))
  return JSON.parse(text)
}

async function refreshApiStatus() {
  $('#api-output').textContent = 'Loading...'
  const paths = ['/health', '/v1/models', '/v1/audio/supported_languages', '/metrics/inference']
  const values = await Promise.all(paths.map(async (path) => {
    try { return [path, await fetchJson(path)] } catch (error) { return [path, { error: errorMessage(error) }] }
  }))
  $('#api-output').textContent = JSON.stringify(Object.fromEntries(values), null, 2)
}

async function refreshSystem() {
  try {
    const [readiness, gpu] = await Promise.all([
      fetchJson('/health/ready'),
      fetch('/system/gpu').then((response) => response.text()),
    ])
    $('#readiness-output').textContent = JSON.stringify(readiness, null, 2)
    $('#gpu-output').innerHTML = gpu
  } catch (error) {
    showToast(errorMessage(error))
  }
}

$('#api-refresh').addEventListener('click', refreshApiStatus)
$('#system-refresh').addEventListener('click', refreshSystem)

async function pollReadiness() {
  const badge = $('#runtime-badge')
  const model = $('#runtime-model')
  try {
    const readiness = await fetchJson('/health/ready')
    state.backendReady = readiness.status === 'ok'
    state.model = readiness.model || state.model
    $('#model-name').textContent = state.model
    updateTimestampAvailability(readiness.capabilities?.timestamps === true)
    badge.dataset.state = state.backendReady ? 'ready' : 'starting'
    badge.querySelector('strong').textContent = state.backendReady ? 'Inference ready' : 'Inference starting'
    model.textContent = readiness.model || 'Qwen3-ASR'
    if (state.backendReady && $('#global-status').textContent === 'Connecting to inference') {
      setStatus('Ready', 'success')
    }
  } catch {
    state.backendReady = false
    badge.dataset.state = 'starting'
    badge.querySelector('strong').textContent = 'Inference starting'
    model.textContent = 'Waiting for inference service'
  }
  setTimeout(pollReadiness, state.backendReady ? 15000 : 3000)
}

document.addEventListener('audio-error', (event) => showToast(errorMessage(event.detail)))
window.addEventListener('beforeunload', () => {
  if (realtime.sessionId) {
    fetch(`/v1/realtime/transcriptions/sessions/${realtime.sessionId}`, { method: 'DELETE', keepalive: true })
  }
})

loadExamples().catch(() => {})
pollReadiness()
