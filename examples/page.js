const UI_LOCALES = [
  { code: 'en', name: 'English' },
  { code: 'pl', name: 'Polski' },
  { code: 'ja', name: '日本語' },
  { code: 'zh', name: '中文' },
  { code: 'es', name: 'Español' },
  { code: 'de', name: 'Deutsch' },
]
const UI_LOCALE_CODES = new Set(UI_LOCALES.map((locale) => locale.code))
const LOCALE_STORAGE_KEY = 'qwen-asr-ui-locale-v1'
const params = new URLSearchParams(window.location.search)

function storedLocale() {
  try { return localStorage.getItem(LOCALE_STORAGE_KEY) } catch { return null }
}

let currentLocale = UI_LOCALE_CODES.has(params.get('lang'))
  ? params.get('lang')
  : UI_LOCALE_CODES.has(storedLocale()) ? storedLocale() : 'en'
let messages = {}
let examples = []
let filterLanguage = params.get('filter') || 'all'
let currentVolume = 0.85
let lastVolume = currentVolume

function t(key, variables = {}, fallback = key) {
  const template = typeof messages[key] === 'string' ? messages[key] : fallback
  return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (match, name) => (
    Object.hasOwn(variables, name) ? String(variables[name]) : match
  ))
}

function languageLabel(language) {
  const key = String(language).toLowerCase().replaceAll(' ', '_')
  return t(`languages.${key}`, {}, language)
}

function replaceQuery(nextParams) {
  const query = nextParams.toString()
  history.replaceState(null, '', `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`)
}

async function fetchJson(relativePath) {
  const response = await fetch(new URL(relativePath, document.baseURI))
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
  return response.json()
}

async function loadMessages(locale) {
  const [english, selected] = await Promise.all([
    fetchJson('../qwen_asr/standalone_ui/static/locales/en.json'),
    locale === 'en'
      ? Promise.resolve({})
      : fetchJson(`../qwen_asr/standalone_ui/static/locales/${locale}.json`),
  ])
  messages = { ...english, ...selected }
}

function applyTranslations() {
  document.documentElement.lang = currentLocale
  document.title = t('examplesPage.title', {}, document.title)
  document.querySelectorAll('[data-i18n]').forEach((element) => {
    element.textContent = t(element.dataset.i18n, {}, element.textContent)
  })
  document.querySelectorAll('[data-i18n-aria-label]').forEach((element) => {
    element.setAttribute('aria-label', t(element.dataset.i18nAriaLabel, {}, element.getAttribute('aria-label') || ''))
  })
  updateVolumeControl()
}

function populateLocalePicker() {
  const picker = document.querySelector('#page-locale')
  picker.replaceChildren(...UI_LOCALES.map((locale) => {
    const option = new Option(locale.name, locale.code)
    option.selected = locale.code === currentLocale
    return option
  }))
  picker.setAttribute('aria-label', t('locale.label'))
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
    playButton.setAttribute('aria-label', t(playing ? 'examplesPage.pause' : 'examplesPage.play', {
      language: languageLabel(card.dataset.language),
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
  heading.textContent = languageLabel(example.language)
  heading.dataset.cardLanguage = ''
  const canonical = document.createElement('p')
  canonical.textContent = example.language
  title.append(heading, canonical)
  const badge = document.createElement('span')
  badge.className = 'sample-badge'
  badge.dataset.sampleBadge = ''
  badge.textContent = t('examplesPage.sampleNumber', { number: '01' })
  head.append(title, badge)

  const reference = document.createElement('div')
  reference.className = 'reference'
  const referenceLabel = document.createElement('span')
  referenceLabel.className = 'reference-label'
  referenceLabel.dataset.referenceLabel = ''
  referenceLabel.textContent = t('examplesPage.reference')
  const transcript = document.createElement('p')
  transcript.textContent = example.expected_text
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
  progressButton.setAttribute('aria-label', t('examplesPage.seek'))
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
  grid.replaceChildren(...examples.map(createCard))
}

function renderFilters() {
  const filter = document.querySelector('#language-filter')
  const languages = examples.map((example) => example.language)
  if (filterLanguage !== 'all' && !languages.includes(filterLanguage)) filterLanguage = 'all'
  const choices = ['all', ...languages]
  filter.replaceChildren(...choices.map((language) => {
    const button = document.createElement('button')
    button.className = `filter-button${language === filterLanguage ? ' is-active' : ''}`
    button.type = 'button'
    button.dataset.language = language
    button.setAttribute('aria-pressed', String(language === filterLanguage))
    button.textContent = language === 'all' ? t('examplesPage.all') : languageLabel(language)
    button.addEventListener('click', () => setFilter(language))
    return button
  }))
}

function setFilter(language) {
  filterLanguage = language
  document.querySelectorAll('.example-card audio').forEach((audio) => audio.pause())
  document.querySelectorAll('.filter-button').forEach((button) => {
    const selected = button.dataset.language === language
    button.classList.toggle('is-active', selected)
    button.setAttribute('aria-pressed', String(selected))
  })
  document.querySelectorAll('.example-card').forEach((card) => {
    card.hidden = language !== 'all' && card.dataset.language !== language
  })
  const nextParams = new URLSearchParams(window.location.search)
  if (language === 'all') nextParams.delete('filter')
  else nextParams.set('filter', language)
  replaceQuery(nextParams)
}

function updateDynamicTranslations() {
  document.querySelectorAll('.example-card').forEach((card) => {
    card.querySelector('[data-card-language]').textContent = languageLabel(card.dataset.language)
    card.querySelector('[data-reference-label]').textContent = t('examplesPage.reference')
    card.querySelector('[data-sample-badge]').textContent = t('examplesPage.sampleNumber', { number: '01' })
    card.querySelector('.progress-button').setAttribute('aria-label', t('examplesPage.seek'))
    const audio = card.querySelector('audio')
    card.querySelector('.play-button').setAttribute('aria-label', t(
      audio.paused ? 'examplesPage.play' : 'examplesPage.pause',
      { language: languageLabel(card.dataset.language) },
    ))
  })
  renderFilters()
  setFilter(filterLanguage)
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
    button.setAttribute('aria-label', t(muted ? 'examplesPage.unmute' : 'examplesPage.mute'))
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

document.querySelector('#page-locale').addEventListener('change', async (event) => {
  currentLocale = event.currentTarget.value
  try { localStorage.setItem(LOCALE_STORAGE_KEY, currentLocale) } catch {}
  const nextParams = new URLSearchParams(window.location.search)
  nextParams.set('lang', currentLocale)
  replaceQuery(nextParams)
  await loadMessages(currentLocale)
  applyTranslations()
  populateLocalePicker()
  updateDynamicTranslations()
})

async function initialize() {
  const loading = document.querySelector('#loading-state')
  if (window.location.protocol === 'file:') {
    loading.dataset.state = 'error'
    loading.textContent = 'Local preview requires a web server. From the repository root, run: python -m http.server 8011 — then open http://localhost:8011/examples/'
    delete document.documentElement.dataset.loading
    return
  }
  try {
    const [manifest] = await Promise.all([
      fetchJson('../testbench/manifest.json'),
      loadMessages(currentLocale),
    ])
    const firstByLanguage = new Map()
    manifest.cases.forEach((example) => {
      if (example.group.startsWith('random') && !firstByLanguage.has(example.language)) {
        firstByLanguage.set(example.language, example)
      }
    })
    examples = manifest.supported_languages_with_assets
      .map(({ language }) => firstByLanguage.get(language))
      .filter(Boolean)
    try { localStorage.setItem(LOCALE_STORAGE_KEY, currentLocale) } catch {}
    applyTranslations()
    populateLocalePicker()
    renderCards()
    renderFilters()
    setFilter(filterLanguage)
    loading.hidden = true
    delete document.documentElement.dataset.loading
  } catch (error) {
    console.error(error)
    loading.dataset.state = 'error'
    loading.textContent = t('examplesPage.loadFailed', {}, 'Examples could not be loaded. Open this page through the published GitHub Pages site.')
    delete document.documentElement.dataset.loading
  }
}

initialize()
