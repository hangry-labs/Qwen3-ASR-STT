// SPDX-License-Identifier: Apache-2.0. Keep gallery locales inside the shared shell.
const requestedGalleryLanguage = new URLSearchParams(location.search).get('lang');
const requestedGalleryFilter = new URLSearchParams(location.search).get('filter');
const initialGallery = document.querySelector('.gallery-content');
if (initialGallery) {
  initialGallery.lang = activeLanguage.code;
  initialGallery.dir = activeLanguage.direction;
  document.documentElement.dir = 'ltr';
}
document.addEventListener('DOMContentLoaded', () => {
  const gallery = document.querySelector('.gallery-content');
  const selector = document.querySelector('#site-language');
  if (!gallery || !selector) return;
  const reflectLanguage = () => {
    gallery.lang = activeLanguage.code;
    gallery.dir = activeLanguage.direction;
    document.documentElement.dir = 'ltr';
    // A separate speech filter survives reloads even outside the six site locales.
    const url = new URL(location.href);
    url.searchParams.set('filter', filterLanguage);
    history.replaceState(null, '', url);
    // The broader speech catalog owns its translations; Norwegian uses English.
    document.querySelector('#translation-notice').hidden = Boolean(LANGUAGE_BY_CODE.get(selector.value)) || activeLanguage.code !== 'en';
  };
  const selectLanguage = () => {
    const language = LANGUAGE_BY_CODE.get(selector.value) || LANGUAGE_BY_CODE.get('en');
    setFilter(language.language, false);
    reflectLanguage();
  };
  if (requestedGalleryFilter) reflectLanguage();
  else if (!requestedGalleryLanguage || [...selector.options].some(option => option.value === requestedGalleryLanguage)) selectLanguage();
  else reflectLanguage();
  selector.addEventListener('change', selectLanguage);
  // The player rebuilds its filter buttons, so delegate to their stable parent.
  gallery.querySelector('#language-filter').addEventListener('click', event => {
    if (event.target.closest('[data-language]')) reflectLanguage();
  });
});
