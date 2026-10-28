# Mixed-language transcription and timestamps

Qwen3-ASR can lose accuracy when a recording switches languages. A language that works correctly in isolation may be omitted, translated or written phonetically in another script when embedded in other speech. Choosing a larger model or forcing a language does not guarantee faithful multilingual transcription.

If recognition misses or changes spoken content, the forced aligner can place subsequent words several seconds early. Positive word durations and ordered timestamps do not establish acoustic accuracy. This also affects subtitles derived from those timestamps.

We independently reproduced [QwenLM/Qwen3-ASR#221](https://github.com/QwenLM/Qwen3-ASR/issues/221) using Chinese–Korean–Chinese recordings generated with another TTS solution:

- Native 1.7B omitted a Korean insertion and placed the following Chinese character 3.60 seconds early in three repeated transcription/alignment calls.
- The current 0.6B vLLM API, with Chinese forced, rendered Korean as Chinese text and placed a following character 3.20 seconds early.
- Complete supplied transcripts and equal-length silence controls aligned correctly in these fixtures.

## A context hint to try

Keep Language on **Auto** when testing a multilingual recording. In the Transcribe tab's context field, describe the languages present. The API accepts the same text through its existing `prompt` field.

For the Chinese/Korean fixtures, this hint recovered the complete transcript and correct following Chinese timestamps in the **native Transformers 1.7B diagnostic backend**:

```text
请逐字转写所有中文和韩语，不要翻译或省略。
```

Meaning: “Transcribe all Chinese and Korean speech verbatim; do not translate or omit it.” The following English hint also improved recognition in that diagnostic:

```text
The recording contains Mandarin Chinese and Korean speech.
```

These are optional experiments, not a general fix or a guarantee for production vLLM, other recordings or other language pairs. The production 0.6B API still made errors on some test cases with these hints. Context can also influence the reported language, and it can leak into output on silent or weak audio. Check the actual recording and transcript before trusting timestamps, especially for training data or precise subtitles.

Korean transcription remains supported. Korean-selected timestamp alignment separately requires an optional tokenizer that is not bundled; such requests return an actionable 422 error. That packaging boundary does not explain or resolve incorrectly aligned mixed speech labeled Chinese.

The runtime keeps its existing decoding and silence behavior. Shorter automatic segmentation was investigated but dropped other words and introduced errors, so it is not enabled as a workaround. We acknowledge this model limitation; no automatic downstream workaround is planned without a broadly validated solution.
