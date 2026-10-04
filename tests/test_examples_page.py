from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
EXAMPLES_DIR = ROOT / "examples"
LOCALES_DIR = ROOT / "qwen_asr" / "standalone_ui" / "static" / "locales"


class ExamplesPageTests(unittest.TestCase):
    def test_page_is_dependency_free_and_uses_repository_assets(self) -> None:
        html = (EXAMPLES_DIR / "index.html").read_text(encoding="utf-8")
        script = (EXAMPLES_DIR / "page.js").read_text(encoding="utf-8")
        stylesheet = (EXAMPLES_DIR / "styles.css").read_text(encoding="utf-8")
        translations = (EXAMPLES_DIR / "translations.js").read_text(encoding="utf-8")
        catalog = (EXAMPLES_DIR / "catalog.js").read_text(encoding="utf-8")

        self.assertIn('src="page.js"', html)
        self.assertIn('src="translations.js"', html)
        self.assertIn('src="catalog.js"', html)
        self.assertIn('href="https://hangrylabs.app/"', html)
        self.assertNotIn("nuggies.website", html)
        self.assertIn('src="background.js"', html)
        self.assertIn('href="styles.css"', html)
        self.assertIn('../assets/qwen3_asr_logo.webp', html)
        self.assertNotIn('../assets/qwen3_asr_logo_horizontal.webp', html)
        self.assertIn('../assets/hangrylabs_favicon.webp', html)
        self.assertNotIn("cdn.", html)
        self.assertNotIn("https://fonts.", html)
        self.assertNotIn("fetch(", script)
        self.assertNotIn("import.meta", script)
        self.assertIn("document.baseURI", script)
        self.assertIn("audio.preload = 'metadata'", script)
        self.assertIn("function pauseOthers(", script)
        self.assertIn("setPointerCapture", script)
        self.assertIn("localStorage.setItem(EXAMPLES_STORAGE_KEY", script)
        self.assertIn("window.QWEN_EXAMPLE_LANGUAGES", translations)
        self.assertIn("window.QWEN_EXAMPLE_CATALOG", catalog)
        self.assertIn('../assets/hero_background.webp', stylesheet)
        self.assertIn("@media (prefers-reduced-motion: reduce)", stylesheet)
        for asset in ("qwen3_asr_badge.webp", "qwen3_asr_logo.webp", "qwen3_asr_mascot.webp"):
            path = ROOT / "assets" / asset
            self.assertTrue(path.is_file(), asset)
            self.assertLess(path.stat().st_size, 300_000, asset)

    def test_showcase_has_one_existing_random_sample_per_supported_language(self) -> None:
        manifest = json.loads((ROOT / "testbench" / "manifest.json").read_text(encoding="utf-8"))
        supported = [item["language"] for item in manifest["supported_languages_with_assets"]]
        first_by_language: dict[str, dict[str, object]] = {}
        for example in manifest["cases"]:
            if str(example["group"]).startswith("random"):
                first_by_language.setdefault(example["language"], example)

        self.assertEqual(len(supported), 30)
        self.assertEqual(len(supported), len(set(supported)))
        self.assertEqual(set(supported), set(first_by_language))
        for language in supported:
            example = first_by_language[language]
            self.assertTrue((ROOT / "testbench" / str(example["audio"])).is_file(), language)
            self.assertTrue(str(example["expected_text"]).strip(), language)

        catalog_script = (EXAMPLES_DIR / "catalog.js").read_text(encoding="utf-8")
        catalog_entries = re.findall(r"\{ language: '([^']+)', audio: '([^']+)'", catalog_script)
        self.assertEqual(
            [(language, str(first_by_language[language]["audio"])) for language in supported],
            catalog_entries,
        )

    def test_examples_own_all_thirty_language_translations(self) -> None:
        translations = (EXAMPLES_DIR / "translations.js").read_text(encoding="utf-8")
        entries = re.findall(
            r"\{ code: '([^']+)', language: '([^']+)', nativeName: '([^']+)', direction: '(ltr|rtl)'",
            translations,
        )
        manifest = json.loads((ROOT / "testbench" / "manifest.json").read_text(encoding="utf-8"))
        supported = [item["language"] for item in manifest["supported_languages_with_assets"]]

        self.assertEqual(len(entries), 30)
        self.assertEqual(len({code for code, _, _, _ in entries}), 30)
        self.assertEqual([language for _, language, _, _ in entries], supported)
        self.assertEqual({language for _, language, _, direction in entries if direction == "rtl"}, {"Arabic", "Persian"})
        for key in ("headline", "intro", "pillLanguages", "pillSamples", "reference", "play", "pause", "seek"):
            self.assertIn(f"{key}:", translations)

    def test_examples_translations_do_not_expand_main_ui_catalogs(self) -> None:
        for locale in ("en", "pl", "ja", "zh", "es", "de"):
            catalog = json.loads((LOCALES_DIR / f"{locale}.json").read_text(encoding="utf-8"))
            self.assertFalse(any(key.startswith("examplesPage.") for key in catalog), locale)


if __name__ == "__main__":
    unittest.main()
