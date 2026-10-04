const LANGUAGE_OPTIONS = window.QWEN_EXAMPLE_LANGUAGES || []
const EXAMPLES = window.QWEN_EXAMPLE_CATALOG || []
const LANGUAGE_BY_CODE = new Map(LANGUAGE_OPTIONS.map((entry) => [entry.code, entry]))
const LANGUAGE_BY_NAME = new Map(LANGUAGE_OPTIONS.map((entry) => [entry.language, entry]))
const EXAMPLES_STORAGE_KEY = 'qwen-asr-examples-language-v1'
const params = new URLSearchParams(window.location.search)

function storedLanguageCode() {
  try { return localStorage.getItem(EXAMPLES_STORAGE_KEY) } catch { return null }
}

function initialLanguage() {
  const queryLocale = LANGUAGE_BY_CODE.get(params.get('lang'))
  if (queryLocale) return queryLocale
  const legacyFilter = LANGUAGE_BY_NAME.get(params.get('filter'))
  if (legacyFilter) return legacyFilter
  return LANGUAGE_BY_CODE.get(storedLanguageCode()) || LANGUAGE_BY_CODE.get('en') || LANGUAGE_OPTIONS[0]
}

let activeLanguage = initialLanguage()
let filterLanguage = params.get('filter') === 'all' ? 'all' : activeLanguage?.language || 'English'
let messages = activeLanguage?.messages || {}
let currentVolume = 0.85
let lastVolume = currentVolume

function t(key, variables = {}, fallback = key) {
  const template = typeof messages[key] === 'string' ? messages[key] : fallback
  return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (match, name) => (
    Object.hasOwn(variables, name) ? String(variables[name]) : match
  ))
}

function nativeLanguageName(language) {
  return LANGUAGE_BY_NAME.get(language)?.nativeName || language
}

function replaceQuery(nextParams) {
  const query = nextParams.toString()
  history.replaceState(null, '', `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`)
}

function applyTranslations() {
  document.documentElement.lang = activeLanguage?.code || 'en'
  document.documentElement.dir = activeLanguage?.direction || 'ltr'
  document.title = `Hangry Labs ${t('headline', {}, 'Qwen3-ASR-STT language examples')}`
  document.querySelectorAll('[data-i18n]').forEach((element) => {
    const translated = t(element.dataset.i18n, {}, element.textContent)
    element.textContent = element.dataset.i18n === 'headline'
      ? translated.replaceAll('Qwen3-ASR-STT', 'Qwen3‑ASR‑STT')
      : translated
  })
  document.querySelectorAll('[data-i18n-aria-label]').forEach((element) => {
    element.setAttribute('aria-label', t(element.dataset.i18nAriaLabel, {}, element.getAttribute('aria-label') || ''))
  })
  updateVolumeControl()
}

function playIcon(playing) {
  return playing
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6.5 5h4v14h-4zM13.5 5h4v14h-4z"/></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10-6.5z"/></svg>'
}

function formatTime(value) {
  if (!Number.isFinite(value)) return '0:00'
  const minutes = Math.floor(value / 60)
  const seconds = Math.floor(value % 60).toString().padStart(2, '0')
  return `${minutes}:${seconds}`
}

function pauseOthers(currentAudio) {
  document.querySelectorAll('.example-card audio').forEach((audio) => {
    if (audio !== currentAudio) audio.pause()
  })
}

function setPlayerProgress(player, value) {
  const progress = Math.max(0, Math.min(100, value))
  player.querySelector('.progress-fill').style.width = `${progress}%`
  player.querySelector('.progress-knob').style.left = `${progress}%`
}

function bindPlayer(card) {
  const audio = card.querySelector('audio')
  const playButton = card.querySelector('.play-button')
  const progressButton = card.querySelector('.progress-button')
  const duration = card.querySelector('.duration')

  function updatePlayState(playing) {
    card.classList.toggle('is-playing', playing)
    playButton.innerHTML = playIcon(playing)
    playButton.setAttribute('aria-label', t(playing ? 'pause' : 'play', {
      language: nativeLanguageName(card.dataset.language),
    }))
  }

  function seek(event) {
    if (!Number.isFinite(audio.duration)) return
    const bounds = progressButton.getBoundingClientRect()
    const ratio = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width))
    audio.currentTime = ratio * audio.duration
    setPlayerProgress(card, ratio * 100)
  }

  playButton.addEventListener('click', async () => {
    if (audio.paused) {
      pauseOthers(audio)
      try { await audio.play() } catch { updatePlayState(false) }
    } else {
      audio.pause()
    }
  })
  progressButton.addEventListener('click', seek)
  progressButton.addEventListener('pointerdown', (event) => {
    event.preventDefault()
    progressButton.setPointerCapture(event.pointerId)
    seek(event)
  })
  progressButton.addEventListener('pointermove', (event) => {
    if (progressButton.hasPointerCapture(event.pointerId)) seek(event)
  })
  progressButton.addEventListener('pointerup', (event) => {
    if (progressButton.hasPointerCapture(event.pointerId)) progressButton.releasePointerCapture(event.pointerId)
  })
  audio.addEventListener('loadedmetadata', () => {
    duration.textContent = `0:00 / ${formatTime(audio.duration)}`
  })
  audio.addEventListener('timeupdate', () => {
    if (!Number.isFinite(audio.duration) || audio.duration <= 0) return
    setPlayerProgress(card, audio.currentTime / audio.duration * 100)
    duration.textContent = `${formatTime(audio.currentTime)} / ${formatTime(audio.duration)}`
  })
  audio.addEventListener('play', () => updatePlayState(true))
  audio.addEventListener('pause', () => updatePlayState(false))
  audio.addEventListener('ended', () => {
    setPlayerProgress(card, 0)
    updatePlayState(false)
  })
  audio.volume = currentVolume
  updatePlayState(false)
}

function createCard(example) {
  const card = document.createElement('article')
  card.className = 'example-card'
  card.dataset.language = example.language

  const head = document.createElement('div')
  head.className = 'card-head'
  const title = document.createElement('div')
  title.className = 'card-title'
  const heading = document.createElement('h3')
  heading.textContent = nativeLanguageName(example.language)
  const canonical = document.createElement('p')
  canonical.textContent = example.language
  title.append(heading, canonical)
  const badge = document.createElement('span')
  badge.className = 'sample-badge'
  badge.textContent = '#01'
  head.append(title, badge)

  const reference = document.createElement('div')
  reference.className = 'reference'
  const referenceLabel = document.createElement('span')
  referenceLabel.className = 'reference-label'
  referenceLabel.dataset.referenceLabel = ''
  referenceLabel.textContent = t('reference')
  const transcript = document.createElement('p')
  transcript.textContent = example.expectedText
  transcript.dir = 'auto'
  reference.append(referenceLabel, transcript)

  const player = document.createElement('div')
  player.className = 'player'
  const playButton = document.createElement('button')
  playButton.className = 'play-button'
  playButton.type = 'button'
  playButton.innerHTML = playIcon(false)
  const playerTrack = document.createElement('div')
  playerTrack.className = 'player-track'
  const progressButton = document.createElement('button')
  progressButton.className = 'progress-button'
  progressButton.type = 'button'
  progressButton.setAttribute('aria-label', t('seek'))
  progressButton.innerHTML = '<span class="progress-track" aria-hidden="true"><span class="progress-fill"></span><span class="progress-knob"></span></span>'
  const duration = document.createElement('span')
  duration.className = 'duration'
  duration.textContent = '0:00'
  playerTrack.append(progressButton, duration)
  player.append(playButton, playerTrack)

  const audio = document.createElement('audio')
  audio.preload = 'metadata'
  audio.src = new URL(`../testbench/${example.audio}`, document.baseURI)
  card.append(head, reference, audio, player)
  bindPlayer(card)
  return card
}

function renderCards() {
  const grid = document.querySelector('#example-grid')
  grid.replaceChildren(...EXAMPLES.map(createCard))
}

function renderFilters() {
  const filter = document.querySelector('#language-filter')
  const choices = [{ language: 'all', nativeName: t('all') }, ...LANGUAGE_OPTIONS]
  filter.replaceChildren(...choices.map((entry) => {
    const language = entry.language
    const button = document.createElement('button')
    button.className = `filter-button${language === filterLanguage ? ' is-active' : ''}`
    button.type = 'button'
    button.dataset.language = language
    button.setAttribute('aria-pressed', String(language === filterLanguage))
    button.textContent = language === 'all' ? t('all') : entry.nativeName
    button.addEventListener('click', () => setFilter(language))
    return button
  }))
}

function updateDynamicTranslations() {
  document.querySelectorAll('.example-card').forEach((card) => {
    card.querySelector('[data-reference-label]').textContent = t('reference')
    card.querySelector('.progress-button').setAttribute('aria-label', t('seek'))
    const audio = card.querySelector('audio')
    card.querySelector('.play-button').setAttribute('aria-label', t(
      audio.paused ? 'play' : 'pause',
      { language: nativeLanguageName(card.dataset.language) },
    ))
  })
}

function setFilter(language, updateUrl = true) {
  filterLanguage = language
  document.querySelectorAll('.example-card audio').forEach((audio) => audio.pause())

  activeLanguage = language === 'all'
    ? LANGUAGE_BY_CODE.get('en')
    : LANGUAGE_BY_NAME.get(language) || LANGUAGE_BY_CODE.get('en')
  messages = activeLanguage.messages
  try { localStorage.setItem(EXAMPLES_STORAGE_KEY, activeLanguage.code) } catch {}

  applyTranslations()
  updateDynamicTranslations()
  renderFilters()
  document.querySelectorAll('.example-card').forEach((card) => {
    card.hidden = language !== 'all' && card.dataset.language !== language
  })

  if (updateUrl) {
    const nextParams = new URLSearchParams(window.location.search)
    nextParams.set('lang', activeLanguage.code)
    if (language === 'all') nextParams.set('filter', 'all')
    else nextParams.delete('filter')
    replaceQuery(nextParams)
  }
}

function updateVolumeControl() {
  const muted = currentVolume <= 0.001
  document.querySelectorAll('.example-card audio').forEach((audio) => {
    audio.volume = currentVolume
    audio.muted = muted
  })
  const slider = document.querySelector('.volume-slider')
  const button = document.querySelector('.volume-button')
  if (slider) slider.value = String(currentVolume)
  if (button) {
    button.dataset.muted = String(muted)
    button.setAttribute('aria-label', t(muted ? 'unmute' : 'mute'))
    button.querySelector('.volume-icon-on').hidden = muted
    button.querySelector('.volume-icon-muted').hidden = !muted
  }
}

document.querySelector('.volume-slider').addEventListener('input', (event) => {
  currentVolume = Number(event.currentTarget.value)
  if (currentVolume > 0) lastVolume = currentVolume
  updateVolumeControl()
})

document.querySelector('.volume-button').addEventListener('click', () => {
  if (currentVolume > 0) {
    lastVolume = currentVolume
    currentVolume = 0
  } else {
    currentVolume = lastVolume || 0.85
  }
  updateVolumeControl()
})

function initialize() {
  const loading = document.querySelector('#loading-state')
  if (LANGUAGE_OPTIONS.length !== 30 || EXAMPLES.length !== 30) {
    loading.dataset.state = 'error'
    loading.textContent = 'Examples could not be loaded.'
    delete document.documentElement.dataset.loading
    return
  }
  applyTranslations()
  renderCards()
  setFilter(filterLanguage, false)
  loading.hidden = true
  delete document.documentElement.dataset.loading
}

initialize()
