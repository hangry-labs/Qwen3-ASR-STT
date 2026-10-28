---
layout: default
title: Transcription examples
permalink: /examples/
sidebar: false
wide: true
content_class: gallery-content asr-gallery page-shell
stylesheets:
  - /examples/styles.css
  - /examples/website.css
scripts:
  - /examples/translations.js
  - /examples/catalog.js
  - /examples/page.js
  - /examples/website.js
---
{% capture gallery %}{% include_relative index.html %}{% endcapture %}
{{ gallery | split: '<main class="page-shell">' | last | split: '</main>' | first | strip }}
