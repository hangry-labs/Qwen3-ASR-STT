from __future__ import annotations

import json
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

        self.assertIn('src="page.js"', html)
        self.assertIn('href="https://hangrylabs.app/"', html)
        self.assertNotIn("nuggies.website", html)
        self.assertIn('src="background.js"', html)
        self.assertIn('href="styles.css"', html)
        self.assertIn('../assets/qwen3_asr_logo_horizontal.webp', html)
        self.assertIn('../assets/hangrylabs_favicon.webp', html)
        self.assertNotIn("cdn.", html)
        self.assertNotIn("https://fonts.", html)
        self.assertIn("../testbench/manifest.json", script)
        self.assertIn("../qwen_asr/standalone_ui/static/locales/", script)
        self.assertIn("document.baseURI", script)
        self.assertIn("Local preview requires a web server", script)
        self.assertIn("audio.preload = 'metadata'", script)
        self.assertIn("function pauseOthers(", script)
        self.assertIn("setPointerCapture", script)
        self.assertIn("localStorage.setItem(LOCALE_STORAGE_KEY", script)
        self.assertIn('../assets/hero_background.webp', stylesheet)
        self.assertIn("@media (prefers-reduced-motion: reduce)", stylesheet)

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

    def test_examples_translations_exist_in_every_ui_catalog(self) -> None:
        english = json.loads((LOCALES_DIR / "en.json").read_text(encoding="utf-8"))
        example_keys = {key for key in english if key.startswith("examplesPage.")}

        self.assertGreaterEqual(len(example_keys), 25)
        for locale in ("en", "pl", "ja", "zh", "es", "de"):
            catalog = json.loads((LOCALES_DIR / f"{locale}.json").read_text(encoding="utf-8"))
            self.assertEqual(example_keys, {key for key in catalog if key.startswith("examplesPage.")}, locale)
            self.assertTrue(all(catalog[key].strip() for key in example_keys), locale)


if __name__ == "__main__":
    unittest.main()
