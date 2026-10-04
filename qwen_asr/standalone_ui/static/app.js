import { AudioEditor } from './audio-editor.js'
import { formatTime } from './audio-utils.js'
import { browserLanguage, initializeI18n, languageLabel, t } from './i18n.js'
import { RealtimeRecorder } from './realtime.js'

await initializeI18n()

const $ = (selector, root = document) => root.querySelector(selector)
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)]

const state = {
  source: 'upload',
  activeTab: 'transcribe',
  headerCollapsed: false,
  backendReady: false,
  model: 'qwen3-asr',
  timestampsAvailable: null,
  aligner: null,
  alignerBusy: false,
  alignerUnloadTimer: null,
  timestampGranularities: [],
  timestampSessionRestored: false,
  responseFormatBeforeTimestamps: null,
  gpuHistory: new Map(),
  gpuStats: [],
  gpuWindowMs: 60 * 1000,
  gpuTimer: null,
  gpuRefreshActive: false,
  gpuHovering: false,
  headerAnimation: null,
}

const GPU_HISTORY_RETENTION_MS = 10 * 60 * 1000
const GPU_POLL_INTERVAL_MS = 1000
const ALIGNER_IDLE_UNLOAD_MS = 60 * 1000
const UI_SESSION_KEY = 'qwen-asr-ui-state-v1'
const GPU_SESSION_KEY = 'qwen-asr-gpu-history-v1'
const GPU_METRICS = [
  { key: 'utilization', label: t('gpu.metric.gpu'), color: '#ff7a1a' },
  { key: 'memory_utilization', label: t('gpu.metric.memoryActivity'), color: '#c586c0' },
  { key: 'memory_used', label: t('gpu.metric.vram'), color: '#72a7ff' },
  { key: 'temperature', label: t('gpu.metric.temperature'), color: '#ef6b73' },
  { key: 'power', label: t('gpu.metric.power'), color: '#f2c94c' },
  { key: 'fan_speed', label: t('gpu.metric.fan'), color: '#55c58a' },
  { key: 'graphics_clock', label: t('gpu.metric.graphicsClock'), color: '#9cdcfe' },
  { key: 'memory_clock', label: t('gpu.metric.memoryClock'), color: '#ce9178' },
]

const uploadEditor = new AudioEditor($('#upload-editor'), { label: t('audio.uploadLabel') })
const recordEditor = new AudioEditor($('#record-editor'), { label: t('audio.recordedLabel') })
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

function readSessionJson(key) {
  try {
    return JSON.parse(sessionStorage.getItem(key) || 'null')
  } catch {
    return null
  }
}

function persistUiSession() {
  try {
    sessionStorage.setItem(UI_SESSION_KEY, JSON.stringify({
      activeTab: state.activeTab,
      headerCollapsed: state.headerCollapsed,
      gpuWindowMs: state.gpuWindowMs,
      timestampGranularities: state.timestampSessionRestored
        ? selectedTimestampGranularities()
        : state.timestampGranularities,
    }))
  } catch {
    // Session storage is optional in privacy-restricted browsers.
  }
}

function persistGpuSession() {
  try {
    sessionStorage.setItem(GPU_SESSION_KEY, JSON.stringify({
      savedAt: Date.now(),
      stats: state.gpuStats,
      history: Object.fromEntries(state.gpuHistory),
    }))
  } catch {
    // Monitoring continues in memory if session storage is unavailable.
  }
}

function restoreSessionState() {
  const ui = readSessionJson(UI_SESSION_KEY)
  if (['transcribe', 'stream', 'api', 'system'].includes(ui?.activeTab)) state.activeTab = ui.activeTab
  if (typeof ui?.headerCollapsed === 'boolean') state.headerCollapsed = ui.headerCollapsed
  if ([60 * 1000, 10 * 60 * 1000].includes(ui?.gpuWindowMs)) state.gpuWindowMs = ui.gpuWindowMs
  if (Array.isArray(ui?.timestampGranularities)) {
    state.timestampGranularities = ui.timestampGranularities.filter((value) => ['word', 'segment'].includes(value))
  }

  const cached = readSessionJson(GPU_SESSION_KEY)
  const cutoff = Date.now() - GPU_HISTORY_RETENTION_MS
  if (!cached || !Number.isFinite(cached.savedAt) || cached.savedAt < cutoff) return
  if (Array.isArray(cached.stats)) state.gpuStats = cached.stats
  if (!cached.history || typeof cached.history !== 'object') return
  Object.entries(cached.history).forEach(([index, samples]) => {
    if (!Array.isArray(samples)) return
    const recent = samples.filter((sample) => Number.isFinite(sample?.timestamp) && sample.timestamp >= cutoff)
    if (recent.length) state.gpuHistory.set(Number(index), recent)
  })
}

function setHeroCollapsed(collapsed, persist = true, animate = true) {
  const hero = $('#brand-hero')
  const toggle = $('#hero-toggle')
  state.headerAnimation?.cancel()
  const startHeight = hero.getBoundingClientRect().height

  state.headerCollapsed = collapsed
  const action = collapsed ? t('hero.expand') : t('hero.collapse')
  document.documentElement.dataset.headerCollapsed = String(collapsed)
  hero.dataset.collapsed = String(collapsed)
  toggle.setAttribute('aria-expanded', String(!collapsed))
  toggle.setAttribute('aria-label', action)
  toggle.title = action
  toggle.querySelector('i').className = collapsed ? 'icon-chevron-down' : 'icon-chevron-up'

  const endHeight = hero.getBoundingClientRect().height
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  if (animate && !reducedMotion && Math.abs(startHeight - endHeight) > 1) {
    hero.classList.add('is-rolling')
    const animation = hero.animate(
      [{ height: `${startHeight}px` }, { height: `${endHeight}px` }],
      { duration: 320, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'both' },
    )
    state.headerAnimation = animation
    animation.finished
      .catch(() => {})
      .finally(() => {
        if (state.headerAnimation !== animation) return
        hero.classList.remove('is-rolling')
        state.headerAnimation = null
        animation.cancel()
      })
  }
  if (persist) persistUiSession()
}

$('#hero-toggle').addEventListener('click', () => setHeroCollapsed(!state.headerCollapsed))

function activateTab(name) {
  state.activeTab = name
  persistUiSession()
  $$('.tab-button').forEach((button) => {
    const active = button.dataset.tab === name
    button.classList.toggle('active', active)
    button.setAttribute('aria-selected', String(active))
  })
  $$('.tab-panel').forEach((panel) => { panel.hidden = panel.dataset.panel !== name })
  if (name === 'api') refreshApiStatus()
  if (name === 'system') {
    refreshSystem()
    startGpuMonitor()
  } else {
    state.gpuHovering = false
    stopGpuMonitor()
  }
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
  select.replaceChildren(new Option(t('record.defaultMicrophone'), ''))
  devices.forEach((device, index) => {
    const option = document.createElement('option')
    option.value = device.deviceId
    option.textContent = device.label || t('record.microphoneNumber', { number: index + 1 })
    select.append(option)
  })
  if ([...select.options].some((option) => option.value === current)) select.value = current
}

function setRecordButton(recording, busy = false) {
  const button = $('#record-toggle')
  const changed = button.dataset.recording !== String(recording)
  button.dataset.recording = String(recording)
  button.setAttribute('aria-pressed', String(recording))
  button.disabled = busy
  $('#record-device').disabled = recording || busy
  $('#record-device-refresh').disabled = recording || busy
  if (changed) button.innerHTML = recording
    ? `<i class="icon-square"></i> ${t('record.stop')}`
    : `<i class="icon-mic"></i> ${t('audio.record')}`
}

recordEditor.attachRecorder({
  onStart: () => {
    setRecordButton(true)
    $('#record-state').textContent = t('record.recording', { time: '0:00' })
  },
  onProgress: (duration) => {
    $('#record-state').textContent = t('record.recording', { time: formatTime(duration / 1000) })
  },
  onEnd: (_file, duration) => {
    setRecordButton(false)
    $('#record-state').textContent = t('record.recordingReady', { time: formatTime(duration / 1000) })
  },
})
setRecordButton(false)

$('#record-toggle').addEventListener('click', async () => {
  if (recordEditor.isRecording()) {
    setRecordButton(true, true)
    recordEditor.stopRecording()
    return
  }

  setRecordButton(false, true)
  try {
    const deviceId = $('#record-device').value
    await recordEditor.startRecording(deviceId ? { deviceId: { exact: deviceId } } : undefined)
    await refreshAudioDevices($('#record-device'))
  } catch (error) {
    setRecordButton(false)
    $('#record-state').textContent = t('record.ready')
    showToast(errorMessage(error))
  }
})

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
    option.textContent = `${languageLabel(example.language)} · ${example.name}`
    option.dataset.language = example.language
    option.dataset.name = example.name
    select.append(option)
  })
  syncTimestampExampleOptions()
}

$('#example-select').addEventListener('change', async () => {
  const option = $('#example-select').selectedOptions[0]
  if (!option?.value) return
  const select = $('#example-select')
  select.disabled = true
  try {
    setStatus(t('status.loadingExample'))
    const response = await fetch(option.value)
    if (!response.ok) throw new Error(await responseError(response))
    await uploadEditor.load(await response.blob(), option.dataset.name)
    state.source = 'upload'
    $('[data-source="upload"]').click()
    setStatus(t('status.exampleReady'), 'success')
  } catch (error) {
    select.value = ''
    setStatus(t('status.exampleFailed'), 'error')
    showToast(errorMessage(error))
  } finally {
    select.disabled = false
  }
})

function selectedTimestampGranularities() {
  return $$('input[name="timestamp"]:checked').map((input) => input.value)
}

function syncTimestampResponseFormat() {
  const select = $('#response-format')
  const timestampsSelected = selectedTimestampGranularities().length > 0
  if (timestampsSelected) {
    if (select.value !== 'verbose_json' && state.responseFormatBeforeTimestamps === null) {
      state.responseFormatBeforeTimestamps = select.value
    }
    select.value = 'verbose_json'
    select.disabled = true
    select.title = t('timestamp.verboseRequired')
    return
  }
  select.disabled = false
  select.title = ''
  if (state.responseFormatBeforeTimestamps !== null) {
    select.value = state.responseFormatBeforeTimestamps
    state.responseFormatBeforeTimestamps = null
  }
}

function timestampLanguageOptionIssue(language, aligner = state.aligner) {
  if (!language || language === 'Auto' || !aligner) return null
  const supported = Array.isArray(aligner.model_supported_languages) ? aligner.model_supported_languages : []
  if (supported.length && !supported.includes(language)) {
    const label = languageLabel(language)
    return t('timestamp.languageUnsupported', { language: label })
  }
  const unavailable = aligner.unavailable_languages && typeof aligner.unavailable_languages === 'object'
    ? aligner.unavailable_languages
    : {}
  return unavailable[language] || null
}

function timestampLanguageIssue(aligner = state.aligner) {
  return timestampLanguageOptionIssue($('#language')?.value, aligner)
}

function resetUnsupportedExampleForTimestamps(aligner = state.aligner) {
  if (state.source !== 'upload') return null
  const select = $('#example-select')
  const option = select?.selectedOptions[0]
  const language = option?.dataset.language
  const issue = language ? timestampLanguageOptionIssue(language, aligner) : null
  if (!issue) return null

  uploadEditor.clear()
  select.value = ''
  $('#transcript-output').textContent = ''
  $('#response-output').textContent = ''
  setStatus(t('timestamp.chooseCompatible'))
  return t('timestamp.exampleRemoved', { language: languageLabel(language) })
}

function syncTimestampLanguageOptions(aligner = state.aligner) {
  const select = $('#language')
  if (!select) return
  const timestampsSelected = selectedTimestampGranularities().length > 0
  Array.from(select.options).forEach((option) => {
    const issue = timestampsSelected ? timestampLanguageOptionIssue(option.value, aligner) : null
    option.disabled = Boolean(issue)
    option.title = issue || ''
  })
  select.title = timestampsSelected
    ? t('timestamp.languagesDisabled')
    : ''
}

function syncTimestampExampleOptions(aligner = state.aligner) {
  const select = $('#example-select')
  if (!select) return
  const timestampsSelected = selectedTimestampGranularities().length > 0
  Array.from(select.options).forEach((option) => {
    const language = option.dataset.language
    const issue = timestampsSelected && language ? timestampLanguageOptionIssue(language, aligner) : null
    option.disabled = Boolean(issue)
    option.title = issue || ''
  })
  select.title = timestampsSelected
    ? t('timestamp.examplesDisabled')
    : ''
}

function syncTimestampSelectionUi() {
  syncTimestampResponseFormat()
  syncTimestampLanguageOptions()
  syncTimestampExampleOptions()
}

function cancelAlignerIdleUnload() {
  clearTimeout(state.alignerUnloadTimer)
  state.alignerUnloadTimer = null
}

async function releaseIdleAligner() {
  state.alignerUnloadTimer = null
  if (selectedTimestampGranularities().length || !state.aligner?.loaded || state.aligner?.load_always) return
  setAlignerBusy(true)
  try {
    const aligner = await fetchJson('/system/aligner/unload', { method: 'POST' })
    updateTimestampAvailability(aligner)
    showToast(t('aligner.unusedReleased'), 'success')
  } catch (error) {
    showToast(errorMessage(error))
    await refreshAlignerSettings().catch(() => {})
  } finally {
    setAlignerBusy(false)
  }
}

function scheduleAlignerIdleUnload() {
  cancelAlignerIdleUnload()
  if (selectedTimestampGranularities().length || !state.aligner?.loaded || state.aligner?.load_always) return
  state.alignerUnloadTimer = setTimeout(releaseIdleAligner, ALIGNER_IDLE_UNLOAD_MS)
  updateTimestampAvailability(state.aligner)
}

function updateTimestampAvailability(aligner) {
  const configured = aligner?.configured === true
  const loaded = aligner?.loaded === true
  const status = aligner?.status || (configured ? 'unloaded' : 'unavailable')
  const languageIssue = timestampLanguageIssue(aligner)
  state.aligner = aligner || null
  state.timestampsAvailable = configured
  const support = $('#timestamp-support')
  support.dataset.state = languageIssue ? 'unavailable' : loaded ? 'available' : configured ? 'on-demand' : 'unavailable'
  support.textContent = languageIssue
    ? t('timestamp.languageUnavailable', { language: languageLabel($('#language').value) })
    : loaded && state.alignerUnloadTimer
      ? t('aligner.idleRelease')
      : loaded ? t('aligner.readyStatus') : status === 'loading' ? t('aligner.loading') : configured ? t('aligner.loadsOnSelection') : t('aligner.unavailable')
  support.title = languageIssue || (loaded
    ? t('aligner.wordsReady')
    : configured
      ? t('aligner.selectionLoads')
      : t('aligner.notConfigured'))
  syncTimestampLanguageOptions(aligner)
  syncTimestampExampleOptions(aligner)
  renderAlignerSettings(aligner)
}

function setAlignerBusy(busy) {
  state.alignerBusy = busy
  const languageUnavailable = Boolean(timestampLanguageIssue())
  $$('input[name="timestamp"]').forEach((input) => { input.disabled = busy || languageUnavailable })
  $('#aligner-load-always').disabled = busy || state.aligner?.configured !== true
  $('#aligner-load-now').disabled = busy || state.aligner?.configured !== true || state.aligner?.loaded === true
  $('#aligner-unload-now').disabled = busy || state.aligner?.loaded !== true || state.aligner?.load_always === true
}

async function loadAlignerOnDemand() {
  if (state.aligner?.loaded) return state.aligner
  setAlignerBusy(true)
  updateTimestampAvailability({ ...state.aligner, configured: true, loaded: false, status: 'loading' })
  setStatus(t('aligner.loadingStatus'))
  const statusPoll = setInterval(async () => {
    try {
      const aligner = await fetchJson('/system/aligner')
      if (state.alignerBusy) updateTimestampAvailability(aligner)
    } catch {
      // The load request remains authoritative; transient status polling is optional.
    }
  }, 750)
  try {
    const aligner = await fetchJson('/system/aligner/load', { method: 'POST' })
    updateTimestampAvailability(aligner)
    setStatus(t('aligner.readyStatus'), 'success')
    showToast(t('aligner.loadedToast'), 'success')
    return aligner
  } finally {
    clearInterval(statusPoll)
    setAlignerBusy(false)
  }
}

async function restoreTimestampSession() {
  if (state.timestampSessionRestored) return
  state.timestampSessionRestored = true
  const saved = state.timestampGranularities
  state.timestampGranularities = []
  if (!saved.length) {
    scheduleAlignerIdleUnload()
    return
  }

  if (state.timestampsAvailable !== true || timestampLanguageIssue()) {
    persistUiSession()
    return
  }
  $$('input[name="timestamp"]').forEach((input) => {
    input.checked = saved.includes(input.value)
  })
  syncTimestampSelectionUi()
  cancelAlignerIdleUnload()
  if (!selectedTimestampGranularities().length || state.aligner?.loaded) return

  try {
    await loadAlignerOnDemand()
  } catch (error) {
    $$('input[name="timestamp"]').forEach((input) => { input.checked = false })
    syncTimestampSelectionUi()
    persistUiSession()
    showToast(errorMessage(error))
    await refreshAlignerSettings().catch(() => {})
  }
}

$$('input[name="timestamp"]').forEach((input) => input.addEventListener('change', async () => {
  let resetMessage = null
  if (input.checked) resetMessage = resetUnsupportedExampleForTimestamps()
  syncTimestampSelectionUi()
  persistUiSession()
  if (resetMessage) showToast(resetMessage)
  if (!input.checked) {
    scheduleAlignerIdleUnload()
    return
  }
  cancelAlignerIdleUnload()
  const languageIssue = timestampLanguageIssue()
  if (languageIssue) {
    input.checked = false
    syncTimestampSelectionUi()
    persistUiSession()
    updateTimestampAvailability(state.aligner)
    showToast(languageIssue)
    return
  }
  if (state.aligner?.loaded === true) {
    updateTimestampAvailability(state.aligner)
    return
  }
  if (state.timestampsAvailable !== true) {
    input.checked = false
    syncTimestampSelectionUi()
    persistUiSession()
    showToast(
      state.timestampsAvailable === false
        ? t('timestamp.noModel')
        : t('timestamp.stillChecking'),
    )
    return
  }
  try {
    await loadAlignerOnDemand()
  } catch (error) {
    input.checked = false
    syncTimestampSelectionUi()
    persistUiSession()
    showToast(errorMessage(error))
    await refreshAlignerSettings().catch(() => {})
  }
}))

$('#language').addEventListener('change', () => {
  const languageIssue = timestampLanguageIssue()
  if (languageIssue && selectedTimestampGranularities().length) {
    $$('input[name="timestamp"]').forEach((input) => { input.checked = false })
    syncTimestampSelectionUi()
    persistUiSession()
    scheduleAlignerIdleUnload()
    showToast(languageIssue)
  }
  updateTimestampAvailability(state.aligner)
})

$('#transcribe-form').addEventListener('submit', async (event) => {
  event.preventDefault()
  const editor = state.source === 'upload' ? uploadEditor : recordEditor
  const file = editor.currentFile()
  if (!file) {
    showToast(t('errors.selectAudio'))
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
  setStatus(t('status.transcribing'))
  const started = performance.now()
  try {
    const response = await fetch('/v1/audio/transcriptions', { method: 'POST', body: form })
    const responseText = await response.text()
    if (!response.ok) throw new Error(await responseError(new Response(responseText, { status: response.status })))
    let payload = responseText
    try { payload = JSON.parse(responseText) } catch {}
    $('#transcript-output').textContent = typeof payload === 'string' ? payload : payload.text || ''
    $('#response-output').textContent = typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2)
    setStatus(t('status.httpComplete', {
      status: response.status,
      seconds: ((performance.now() - started) / 1000).toFixed(3),
    }), 'success')
  } catch (error) {
    setStatus(t('status.transcriptionFailed'), 'error')
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

function setRealtimeButton(recording, busy = false) {
  const button = $('#realtime-toggle')
  const changed = button.dataset.recording !== String(recording)
  button.dataset.recording = String(recording)
  button.setAttribute('aria-pressed', String(recording))
  button.disabled = busy
  $('#realtime-device').disabled = recording || busy
  $('#realtime-device-refresh').disabled = recording || busy
  if (changed) button.innerHTML = recording
    ? `<i class="icon-square"></i> ${t('realtime.stop')}`
    : `<span class="record-dot"></span> ${t('realtime.start')}`
}

const realtime = new RealtimeRecorder({
  canvas: $('#realtime-wave'),
  onState: (stateCode, details = {}) => {
    const key = `realtime.${details.time ? 'recordingTime' : stateCode}`
    const value = t(key, details)
    $('#realtime-state').textContent = value
    setStatus(value, stateCode === 'finalized' ? 'success' : 'neutral')
    if (stateCode === 'recording') setRealtimeButton(true)
    if (stateCode === 'ready' || stateCode === 'finalized') setRealtimeButton(false)
  },
  onTranscript: (text, language, final) => {
    $('#realtime-output').textContent = text
    $('#realtime-language').textContent = language ? languageLabel(language) : t('realtime.autoDetection')
    $('#realtime-final').hidden = !final
  },
  onError: (error) => showToast(errorMessage(error)),
})
setRealtimeButton(false)

$('#realtime-toggle').addEventListener('click', async () => {
  if (realtime.running) {
    setRealtimeButton(true, true)
    try {
      await realtime.stop()
    } catch (error) {
      showToast(errorMessage(error))
    } finally {
      setRealtimeButton(false)
    }
    return
  }

  setRealtimeButton(false, true)
  try {
    await realtime.start({
      deviceId: $('#realtime-device').value,
      model: state.model,
      language: $('#language').value,
      prompt: $('#prompt').value,
      chunkSize: Number($('#chunk-size').value),
      maxWindow: Number($('#max-window').value),
      unfixedChunks: Number($('#unfixed-chunks').value),
      unfixedTokens: Number($('#unfixed-tokens').value),
    })
    await refreshAudioDevices($('#realtime-device'))
  } catch (error) {
    showToast(errorMessage(error))
    await realtime.reset()
  }
})

$('#realtime-reset').addEventListener('click', async () => {
  setRealtimeButton(realtime.running, true)
  try {
    await realtime.reset()
  } finally {
    setRealtimeButton(false)
  }
})

$('#realtime-device-refresh').addEventListener('click', async () => {
  try {
    await navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => stream.getTracks().forEach((track) => track.stop()))
    await refreshAudioDevices($('#realtime-device'))
  } catch (error) {
    showToast(errorMessage(error))
  }
})

async function fetchJson(path, options) {
  const response = await fetch(path, options)
  const text = await response.text()
  if (!response.ok) throw new Error(await responseError(new Response(text, { status: response.status })))
  return JSON.parse(text)
}

async function refreshApiStatus() {
  $('#api-output').textContent = t('api.loading')
  const paths = ['/health', '/v1/models', '/v1/audio/supported_languages', '/metrics/inference']
  const values = await Promise.all(paths.map(async (path) => {
    try { return [path, await fetchJson(path)] } catch (error) { return [path, { error: errorMessage(error) }] }
  }))
  $('#api-output').textContent = JSON.stringify(Object.fromEntries(values), null, 2)
}

async function refreshSystem() {
  try {
    const [readiness, aligner, settings] = await Promise.all([
      fetchJson('/health/ready'),
      fetchJson('/system/aligner'),
      fetchJson('/system/settings'),
    ])
    $('#readiness-output').textContent = JSON.stringify(readiness, null, 2)
    updateTimestampAvailability(aligner)
    applyRealtimeDefaults(settings.realtime_defaults, { includeStream: false })
  } catch (error) {
    showToast(errorMessage(error))
  }
}

async function loadRuntimeSettings() {
  const settings = await fetchJson('/system/settings')
  applyRealtimeDefaults(settings.realtime_defaults)
}

function applyRealtimeDefaults(defaults, { includeStream = true } = {}) {
  if (!defaults) return
  const values = {
    '#system-chunk-size': defaults.chunk_size_sec,
    '#system-max-window': defaults.max_window_sec,
    '#system-unfixed-chunks': defaults.unfixed_chunk_num,
    '#system-unfixed-tokens': defaults.unfixed_token_num,
  }
  if (includeStream) {
    Object.assign(values, {
      '#chunk-size': defaults.chunk_size_sec,
      '#max-window': defaults.max_window_sec,
      '#unfixed-chunks': defaults.unfixed_chunk_num,
      '#unfixed-tokens': defaults.unfixed_token_num,
    })
  }
  Object.entries(values).forEach(([selector, value]) => {
    const input = $(selector)
    if (input && Number.isFinite(Number(value))) {
      input.value = String(value)
      sliderOutput(input)
    }
  })
}

$('#realtime-defaults-save').addEventListener('click', async (event) => {
  const button = event.currentTarget
  button.disabled = true
  try {
    const settings = await fetchJson('/system/settings/realtime', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chunk_size_sec: Number($('#system-chunk-size').value),
        max_window_sec: Number($('#system-max-window').value),
        unfixed_chunk_num: Number($('#system-unfixed-chunks').value),
        unfixed_token_num: Number($('#system-unfixed-tokens').value),
      }),
    })
    applyRealtimeDefaults(settings.realtime_defaults)
    showToast(t('system.defaultsSaved'), 'success')
  } catch (error) {
    showToast(errorMessage(error))
  } finally {
    button.disabled = false
  }
})

function renderAlignerSettings(aligner) {
  if (!aligner) return
  const badge = $('#aligner-status-badge')
  const labels = {
    unavailable: t('aligner.unavailable'),
    unloaded: t('aligner.notLoaded'),
    loading: t('aligner.loading'),
    loaded: aligner.load_always ? t('aligner.loadedPersistent') : t('aligner.loadedOnDemand'),
    unloading: t('aligner.releasing'),
    error: t('aligner.loadFailed'),
  }
  badge.dataset.state = aligner.status || 'unavailable'
  badge.textContent = labels[aligner.status] || aligner.status
  $('#aligner-model-name').textContent = aligner.model || t('aligner.noModel')
  $('#aligner-load-always').checked = aligner.load_always === true
  setAlignerBusy(state.alignerBusy)
}

async function refreshAlignerSettings() {
  const aligner = await fetchJson('/system/aligner')
  updateTimestampAvailability(aligner)
  return aligner
}

$('#aligner-load-always').addEventListener('change', async (event) => {
  const enabled = event.currentTarget.checked
  cancelAlignerIdleUnload()
  setAlignerBusy(true)
  try {
    const aligner = await fetchJson('/system/aligner', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ load_aligner_always: enabled }),
    })
    if (!aligner.loaded) {
      $$('input[name="timestamp"]').forEach((input) => { input.checked = false })
      syncTimestampSelectionUi()
      persistUiSession()
    }
    updateTimestampAvailability(aligner)
    showToast(enabled ? t('aligner.persistEnabled') : t('aligner.persistDisabled'), 'success')
  } catch (error) {
    event.currentTarget.checked = !enabled
    showToast(errorMessage(error))
    await refreshAlignerSettings().catch(() => {})
  } finally {
    setAlignerBusy(false)
  }
})

$('#aligner-load-now').addEventListener('click', async () => {
  try {
    await loadAlignerOnDemand()
  } catch (error) {
    showToast(errorMessage(error))
    await refreshAlignerSettings().catch(() => {})
  }
})

$('#aligner-unload-now').addEventListener('click', async () => {
  cancelAlignerIdleUnload()
  setAlignerBusy(true)
  try {
    const aligner = await fetchJson('/system/aligner/unload', { method: 'POST' })
    $$('input[name="timestamp"]').forEach((input) => { input.checked = false })
    syncTimestampSelectionUi()
    persistUiSession()
    updateTimestampAvailability(aligner)
    showToast(t('aligner.released'), 'success')
  } catch (error) {
    showToast(errorMessage(error))
    await refreshAlignerSettings().catch(() => {})
  } finally {
    setAlignerBusy(false)
  }
})

function element(tag, className, text) {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

function gpuHistoryPoints(samples, now, windowMs, metricKey, maximum, width = 300, height = 70) {
  const windowStart = now - windowMs
  return samples.filter((sample) => Number.isFinite(sample[metricKey])).map((sample) => {
    const x = Math.min(width, Math.max(0, (sample.timestamp - windowStart) / windowMs * width))
    const y = height - (Math.min(maximum, Math.max(0, sample[metricKey])) / maximum * height)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

function addGpuChartGrid(svg, width, height) {
  for (let column = 0; column <= 10; column += 1) {
    const x = column * width / 10
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line')
    line.setAttribute('class', 'gpu-grid-line')
    line.setAttribute('x1', String(x))
    line.setAttribute('x2', String(x))
    line.setAttribute('y1', '0')
    line.setAttribute('y2', String(height))
    svg.append(line)
  }
  for (let row = 0; row <= 4; row += 1) {
    const y = row * height / 4
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line')
    line.setAttribute('class', 'gpu-grid-line')
    line.setAttribute('x1', '0')
    line.setAttribute('x2', String(width))
    line.setAttribute('y1', String(y))
    line.setAttribute('y2', String(y))
    svg.append(line)
  }
}

function mergeGpuHistory(historyPayload) {
  const cutoff = Date.now() - GPU_HISTORY_RETENTION_MS
  Object.entries(historyPayload || {}).forEach(([index, incoming]) => {
    if (!Array.isArray(incoming)) return
    const samplesByTimestamp = new Map()
    const combinedSamples = [...(state.gpuHistory.get(Number(index)) || []), ...incoming]
    combinedSamples.forEach((sample) => {
      if (Number.isFinite(sample?.timestamp) && sample.timestamp >= cutoff) {
        samplesByTimestamp.set(sample.timestamp, sample)
      }
    })
    const merged = [...samplesByTimestamp.values()].sort((left, right) => left.timestamp - right.timestamp)
    if (merged.length) state.gpuHistory.set(Number(index), merged)
  })
  state.gpuHistory.forEach((samples, index) => {
    const recent = samples.filter((sample) => sample.timestamp >= cutoff)
    if (recent.length) state.gpuHistory.set(index, recent)
    else state.gpuHistory.delete(index)
  })
  persistGpuSession()
}

function gpuMetricMaximum(metric, gpu, history) {
  const observedMaximum = Math.max(1, ...history.map((sample) => sample[metric.key] || 0))
  if (['utilization', 'memory_utilization', 'temperature', 'fan_speed'].includes(metric.key)) return 100
  if (metric.key === 'memory_used' && Number.isFinite(gpu.memory_total)) return Math.max(1, gpu.memory_total)
  if (metric.key === 'power' && Number.isFinite(gpu.power_limit)) return Math.max(1, gpu.power_limit)
  if (metric.key === 'graphics_clock' && Number.isFinite(gpu.graphics_clock_max)) {
    return Math.max(1, gpu.graphics_clock_max)
  }
  if (metric.key === 'memory_clock' && Number.isFinite(gpu.memory_clock_max)) {
    return Math.max(1, gpu.memory_clock_max)
  }
  return Math.ceil(observedMaximum * 1.1)
}

function formatGpuMetric(metric, value) {
  if (!Number.isFinite(value)) return 'N/A'
  if (['utilization', 'memory_utilization', 'fan_speed'].includes(metric.key)) return `${Math.round(value)}%`
  if (metric.key === 'memory_used') return `${(value / 1024).toFixed(1)} GB`
  if (metric.key === 'temperature') return `${Math.round(value)} C`
  if (metric.key === 'power') return `${Math.round(value)} W`
  return `${Math.round(value)} MHz`
}

function attachGpuChartHover(plot, samples, metric, now) {
  const line = element('div', 'gpu-hover-line')
  const tooltip = element('div', 'gpu-hover-tooltip')
  line.hidden = true
  tooltip.hidden = true
  plot.append(line, tooltip)

  plot.addEventListener('pointermove', (event) => {
    state.gpuHovering = true
    const bounds = plot.getBoundingClientRect()
    const offset = Math.min(bounds.width, Math.max(0, event.clientX - bounds.left))
    const ratio = bounds.width ? offset / bounds.width : 0
    const targetTime = now - state.gpuWindowMs + (ratio * state.gpuWindowMs)
    const nearest = samples.reduce((best, sample) => {
      if (!best) return sample
      return Math.abs(sample.timestamp - targetTime) < Math.abs(best.timestamp - targetTime) ? sample : best
    }, null)
    const tolerance = Math.max(1500, state.gpuWindowMs * 10 / Math.max(1, bounds.width))
    const hasSample = nearest && Math.abs(nearest.timestamp - targetTime) <= tolerance
    const shownTime = new Date(hasSample ? nearest.timestamp : targetTime).toLocaleTimeString(browserLanguage())
    tooltip.textContent = hasSample
      ? `${formatGpuMetric(metric, nearest[metric.key])} / ${shownTime}`
      : `${t('gpu.noSample')} / ${shownTime}`
    const percent = ratio * 100
    line.style.left = `${percent}%`
    tooltip.style.left = `${percent}%`
    tooltip.classList.toggle('align-start', percent < 18)
    tooltip.classList.toggle('align-end', percent > 82)
    line.hidden = false
    tooltip.hidden = false
  })
  plot.addEventListener('pointerleave', () => {
    state.gpuHovering = false
    line.hidden = true
    tooltip.hidden = true
    renderGpuMonitor(state.gpuStats)
  })
}

function createGpuMetricChart(metric, gpu, history, now) {
  const current = gpu[metric.key]
  if (!Number.isFinite(current)) return null
  const samples = history.filter((sample) => Number.isFinite(sample[metric.key]))
  const maximum = gpuMetricMaximum(metric, gpu, samples)
  const average = samples.length
    ? samples.reduce((total, sample) => total + sample[metric.key], 0) / samples.length
    : current
  const peak = samples.length ? Math.max(...samples.map((sample) => sample[metric.key])) : current
  const chart = element('div', 'gpu-metric-chart')
  chart.style.setProperty('--chart-color', metric.color)
  const chartScale = element('div', 'gpu-chart-scale')
  chartScale.append(element('span', '', metric.label), element('strong', '', formatGpuMetric(metric, current)))
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('class', 'gpu-sparkline')
  svg.setAttribute('viewBox', '0 0 300 70')
  svg.setAttribute('preserveAspectRatio', 'none')
  svg.setAttribute(
    'aria-label',
    t('gpu.historyAria', {
      metric: metric.label,
      average: formatGpuMetric(metric, average),
      peak: formatGpuMetric(metric, peak),
    }),
  )
  svg.setAttribute('role', 'img')
  addGpuChartGrid(svg, 300, 70)
  const points = gpuHistoryPoints(samples, now, state.gpuWindowMs, metric.key, maximum)
  if (samples.length > 1) {
    const pointList = points.split(' ')
    const firstX = pointList[0].split(',')[0]
    const lastX = pointList.at(-1).split(',')[0]
    const area = document.createElementNS('http://www.w3.org/2000/svg', 'polygon')
    area.setAttribute('class', 'gpu-chart-area')
    area.setAttribute('points', `${firstX},70 ${points} ${lastX},70`)
    svg.append(area)
  }
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline')
  line.setAttribute('class', 'gpu-chart-line')
  line.setAttribute('points', points)
  svg.append(line)
  if (samples.length) {
    const latestPoint = points.split(' ').at(-1).split(',')
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle')
    marker.setAttribute('class', 'gpu-chart-marker')
    marker.setAttribute('cx', latestPoint[0])
    marker.setAttribute('cy', latestPoint[1])
    marker.setAttribute('r', '2.5')
    svg.append(marker)
  }
  const plot = element('div', 'gpu-chart-plot')
  plot.append(svg)
  attachGpuChartHover(plot, samples, metric, now)
  const chartAxis = element('div', 'gpu-chart-axis')
  chartAxis.append(
    element('span', '', state.gpuWindowMs === 60 * 1000 ? t('gpu.oneMinute') : t('gpu.tenMinutes')),
    element('span', '', t('gpu.averagePeak', {
      average: formatGpuMetric(metric, average),
      peak: formatGpuMetric(metric, peak),
    })),
  )
  chart.append(chartScale, plot, chartAxis)
  return chart
}

function renderGpuMonitor(gpus) {
  const output = $('#gpu-output')
  const monitor = element('div', 'gpu-monitor')
  const heading = element('div', 'gpu-monitor-heading')
  const windowControl = element('div', 'gpu-window-control')
  windowControl.setAttribute('role', 'group')
  windowControl.setAttribute('aria-label', t('gpu.historyWindow'))
  const historyWindows = [[60 * 1000, t('gpu.oneMinute')], [10 * 60 * 1000, t('gpu.tenMinutes')]]
  historyWindows.forEach(([windowMs, label]) => {
    const button = element('button', windowMs === state.gpuWindowMs ? 'active' : '', label)
    button.type = 'button'
    button.setAttribute('aria-pressed', String(windowMs === state.gpuWindowMs))
    button.addEventListener('click', () => {
      state.gpuWindowMs = windowMs
      persistUiSession()
      renderGpuMonitor(state.gpuStats)
    })
    windowControl.append(button)
  })
  heading.append(element('div', 'gpu-monitor-title', t('gpu.monitor')), windowControl)
  monitor.append(heading)

  if (!gpus.length) {
    monitor.append(element('div', 'gpu-monitor-muted', t('gpu.unavailable')))
    output.replaceChildren(monitor)
    return
  }

  const grid = element('div', 'gpu-card-grid')
  gpus.forEach((gpu) => {
    const now = Date.now()
    const history = (state.gpuHistory.get(gpu.index) || [])
      .filter((sample) => sample.timestamp >= now - state.gpuWindowMs)
    const card = element('div', 'gpu-card')
    const cardHead = element('div', 'gpu-card-head')
    cardHead.append(element('strong', '', `GPU ${gpu.index}`), element('span', '', gpu.name))
    const metrics = element('div', 'gpu-metrics-grid')
    GPU_METRICS.forEach((metric) => {
      const chart = createGpuMetricChart(metric, gpu, history, now)
      if (chart) metrics.append(chart)
    })
    const details = element('div', 'gpu-live-details')
    if (gpu.performance_state) details.append(element('span', '', t('gpu.state', { state: gpu.performance_state })))
    if (Number.isFinite(gpu.pcie_generation) && Number.isFinite(gpu.pcie_width)) {
      details.append(element('span', '', t('gpu.pcie', { generation: gpu.pcie_generation, width: gpu.pcie_width })))
    }
    if (Number.isFinite(gpu.power_limit)) {
      details.append(element('span', '', t('gpu.powerLimit', { power: Math.round(gpu.power_limit) })))
    }
    card.append(cardHead, metrics, details)
    grid.append(card)
  })
  monitor.append(grid)
  output.replaceChildren(monitor)
}

async function refreshGpuMonitor() {
  if (state.gpuRefreshActive) return
  state.gpuRefreshActive = true
  try {
    const payload = await fetchJson('/system/gpu')
    state.gpuStats = Array.isArray(payload.gpus) ? payload.gpus : []
    mergeGpuHistory(payload.history)
    if (!state.gpuHovering) renderGpuMonitor(state.gpuStats)
  } catch {
    if (!state.gpuHovering) renderGpuMonitor(state.gpuStats)
  } finally {
    state.gpuRefreshActive = false
  }
}

function startGpuMonitor() {
  if (state.gpuTimer || document.hidden) return
  if (state.gpuStats.length) renderGpuMonitor(state.gpuStats)
  refreshGpuMonitor()
  state.gpuTimer = setInterval(refreshGpuMonitor, GPU_POLL_INTERVAL_MS)
}

function stopGpuMonitor() {
  clearInterval(state.gpuTimer)
  state.gpuTimer = null
}

$('#api-refresh').addEventListener('click', refreshApiStatus)

async function pollReadiness() {
  const badge = $('#runtime-badge')
  const model = $('#runtime-model')
  try {
    const readiness = await fetchJson('/health/ready')
    state.backendReady = readiness.status === 'ok'
    state.model = readiness.model || state.model
    $('#model-name').textContent = state.model
    const aligner = readiness.capabilities?.forced_aligner || {
      configured: readiness.capabilities?.timestamps === true,
      loaded: readiness.capabilities?.timestamps_loaded === true,
      status: readiness.capabilities?.timestamps_loaded === true
        ? 'loaded'
        : readiness.capabilities?.timestamps === true ? 'unloaded' : 'unavailable',
    }
    if (!state.alignerBusy || aligner.loaded || aligner.status === 'error') {
      updateTimestampAvailability(aligner)
    }
    await restoreTimestampSession()
    badge.dataset.state = state.backendReady ? 'ready' : 'starting'
    badge.querySelector('strong').textContent = state.backendReady ? t('runtime.ready') : t('runtime.starting')
    model.textContent = readiness.model || 'Qwen3-ASR'
    badge.title = readiness.model || 'Qwen3-ASR'
    if (state.backendReady && $('#global-status').textContent === t('status.connecting')) {
      setStatus(t('status.ready'), 'success')
    }
  } catch {
    state.backendReady = false
    badge.dataset.state = 'starting'
    badge.querySelector('strong').textContent = t('runtime.starting')
    model.textContent = t('runtime.waiting')
    badge.title = t('runtime.waiting')
  }
  setTimeout(pollReadiness, state.backendReady ? 15000 : 3000)
}

document.addEventListener('audio-error', (event) => showToast(errorMessage(event.detail)))
document.addEventListener('keydown', (event) => {
  const key = event.key.toLowerCase()
  const hardReload = ((event.ctrlKey || event.metaKey) && event.shiftKey && key === 'r')
    || ((event.ctrlKey || event.metaKey) && key === 'f5')
  if (!hardReload) return
  const ui = readSessionJson(UI_SESSION_KEY)
  if (!ui || typeof ui !== 'object') return
  delete ui.timestampGranularities
  try { sessionStorage.setItem(UI_SESSION_KEY, JSON.stringify(ui)) } catch {}
}, true)
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopGpuMonitor()
  } else if (state.activeTab === 'system') {
    startGpuMonitor()
  }
})
window.addEventListener('beforeunload', () => {
  if (realtime.sessionId) {
    fetch(`/v1/realtime/transcriptions/sessions/${realtime.sessionId}`, { method: 'DELETE', keepalive: true })
  }
  stopGpuMonitor()
})

restoreSessionState()
setHeroCollapsed(state.headerCollapsed, false, false)
activateTab(state.activeTab)
loadExamples().catch(() => {})
loadRuntimeSettings().catch(() => {})
pollReadiness()
