const canvas = document.querySelector('#shapes')
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

if (canvas && !reducedMotion) {
  const context = canvas.getContext('2d')
  const characters = 'QWEN3ASRSTT0123456789'
  const fontSize = 16
  const frameInterval = 100
  let columns = 0
  let drops = []
  let lastUpdate = 0

  function resetCanvas() {
    const scale = window.devicePixelRatio || 1
    canvas.width = Math.floor(window.innerWidth * scale)
    canvas.height = Math.floor(window.innerHeight * scale)
    canvas.style.width = `${window.innerWidth}px`
    canvas.style.height = `${window.innerHeight}px`
    context.setTransform(scale, 0, 0, scale, 0, 0)
    columns = Math.floor(window.innerWidth / fontSize)
    drops = Array.from({ length: columns }, () => [Math.random() * -100, Math.random() * -200])
  }

  function draw(timestamp) {
    if (timestamp - lastUpdate >= frameInterval) {
      lastUpdate = timestamp
      context.fillStyle = 'rgba(0, 0, 0, 0.1)'
      context.fillRect(0, 0, window.innerWidth, window.innerHeight)
      context.fillStyle = '#ff6b00'
      context.font = `${fontSize}px monospace`
      drops.forEach((column, columnIndex) => {
        column.forEach((drop, dropIndex) => {
          const character = characters[Math.floor(Math.random() * characters.length)]
          context.fillText(character, columnIndex * fontSize, drop * fontSize)
          column[dropIndex] = drop * fontSize > window.innerHeight && Math.random() > 0.975 ? 0 : drop + 1
        })
      })
    }
    requestAnimationFrame(draw)
  }

  window.addEventListener('resize', resetCanvas)
  resetCanvas()
  requestAnimationFrame(draw)
}
