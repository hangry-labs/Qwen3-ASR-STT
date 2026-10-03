# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch
from transformers.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASRConfig
from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASRForTokenClassification
from transformers.models.qwen3_asr.processing_qwen3_asr import Qwen3ASRProcessor

from .compile_utils import compile_model_forward
from .utils import AudioLike, ensure_list, normalize_audios


def _repair_timestamps(values: Any) -> List[int]:
    """Repair timestamp bins while preserving the legacy stable anchors."""
    data = values.tolist() if hasattr(values, "tolist") else list(values)
    count = len(data)
    if count == 0:
        # Preserve the exception raised by the previous max(dp) implementation.
        raise ValueError("max() arg is an empty sequence")

    # Stable O(n log n) LNDS construction from Qwen3-ASR PR #215, based on
    # vLLM-Omni PR #8230. The level lists retain the legacy implementation's
    # first endpoint and first valid predecessor tie-breaking behavior.
    tails: List[Any] = []
    levels: List[List[int]] = []
    for index, value in enumerate(data):
        length = bisect_right(tails, value)
        if length == len(tails):
            tails.append(value)
            levels.append([])
        else:
            tails[length] = value
        levels[length].append(index)

    is_normal = [False] * count
    index = levels[-1][0]
    is_normal[index] = True
    for depth in range(len(levels) - 2, -1, -1):
        for predecessor in levels[depth]:
            if data[predecessor] <= data[index]:
                index = predecessor
                is_normal[index] = True
                break

    result = data.copy()
    block_start = 0
    while block_start < count:
        if is_normal[block_start]:
            block_start += 1
            continue

        block_end = block_start
        while block_end < count and not is_normal[block_end]:
            block_end += 1
        anomaly_count = block_end - block_start

        left_value = next(
            (result[pos] for pos in range(block_start - 1, -1, -1) if is_normal[pos]),
            None,
        )
        right_value = next(
            (result[pos] for pos in range(block_end, count) if is_normal[pos]),
            None,
        )

        if anomaly_count <= 2:
            for pos in range(block_start, block_end):
                if left_value is None:
                    result[pos] = right_value
                elif right_value is None:
                    result[pos] = left_value
                else:
                    distance_left = pos - (block_start - 1)
                    distance_right = block_end - pos
                    result[pos] = left_value if distance_left <= distance_right else right_value
        elif left_value is not None and right_value is not None:
            step = (right_value - left_value) / (anomaly_count + 1)
            for pos in range(block_start, block_end):
                result[pos] = left_value + step * (pos - block_start + 1)
        elif left_value is not None:
            for pos in range(block_start, block_end):
                result[pos] = left_value
        elif right_value is not None:
            for pos in range(block_start, block_end):
                result[pos] = right_value

        block_start = block_end

    return [int(value) for value in result]


def _decode_timestamp_bins(timestamp_logits: torch.Tensor) -> List[int]:
    """Decode monotonic timestamp bins, resolving zero-duration words by score."""
    if timestamp_logits.ndim != 2:
        raise ValueError("timestamp_logits must have shape [timestamp_tokens, timestamp_bins]")

    timestamp_count, bin_count = timestamp_logits.shape
    if timestamp_count % 2:
        raise ValueError("timestamp_logits must contain paired start and end tokens")

    raw_bins = timestamp_logits.argmax(dim=-1).detach().cpu().tolist()
    repaired_bins = _repair_timestamps(raw_bins)
    if all(
        repaired_bins[position] < repaired_bins[position + 1]
        for position in range(0, timestamp_count, 2)
    ):
        return repaired_bins

    # The checkpoint predicts each boundary independently, so a short word can
    # receive the same highest-scoring 80 ms bin for both start and end. Only in
    # that case, find the maximum-score path whose boundaries remain monotonic
    # and whose word ends are at least one bin after their starts. This uses the
    # model's alternative scores rather than inventing a fixed offset.
    if bin_count < 2:
        return repaired_bins

    scores = timestamp_logits.float()
    previous_scores = scores[0]
    backpointers: List[torch.Tensor] = []
    negative_infinity = torch.full(
        (1,),
        float("-inf"),
        dtype=scores.dtype,
        device=scores.device,
    )

    for position in range(1, timestamp_count):
        prefix_scores, prefix_indices = torch.cummax(previous_scores, dim=0)
        if position % 2:
            # End tokens must follow their paired start by at least one bin.
            best_scores = torch.cat((negative_infinity, prefix_scores[:-1]))
            best_indices = torch.cat(
                (
                    torch.zeros(1, dtype=prefix_indices.dtype, device=scores.device),
                    prefix_indices[:-1],
                )
            )
        else:
            # A word may start where the preceding word ends.
            best_scores = prefix_scores
            best_indices = prefix_indices
        previous_scores = scores[position] + best_scores
        backpointers.append(best_indices)

    if not torch.isfinite(previous_scores).any():
        return repaired_bins

    cursor = int(previous_scores.argmax().item())
    constrained_bins = [cursor]
    for pointers in reversed(backpointers):
        cursor = int(pointers[cursor].item())
        constrained_bins.append(cursor)
    constrained_bins.reverse()

    if any(
        constrained_bins[position] >= constrained_bins[position + 1]
        for position in range(0, timestamp_count, 2)
    ):
        return repaired_bins
    return constrained_bins


@dataclass(frozen=True)
class ForcedAlignItem:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class ForcedAlignResult:
    items: List[ForcedAlignItem]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> ForcedAlignItem:
        return self.items[idx]


class Qwen3ForcedAligner:
    """Native Transformers wrapper for Qwen3-ForcedAligner checkpoints."""

    def __init__(self, model: Any, processor: Any, torch_compile_enabled: bool = False):
        self.model = model
        self.processor = processor
        self.torch_compile_enabled = bool(torch_compile_enabled)
        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        self.timestamp_token_id = int(model.config.timestamp_token_id)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_compile: bool = False,
        torch_compile_backend: str = "inductor",
        torch_compile_mode: str = "default",
        torch_compile_fullgraph: bool = False,
        torch_compile_dynamic: bool | None = None,
        **kwargs: Any,
    ) -> "Qwen3ForcedAligner":
        # vLLM registers its generation-oriented Qwen3ASRConfig under the same
        # AutoConfig model type as Transformers. Once a vLLM ASR engine has
        # started, the Auto* classes can therefore resolve an aligner checkpoint
        # to vLLM's config, which has thinker_config instead of audio_config.
        # Pin the native Transformers classes explicitly so lazy aligner loading
        # remains independent of vLLM's process-wide registry changes.
        config = kwargs.pop("config", None)
        hub_keys = {
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "token",
            "trust_remote_code",
        }
        hub_kwargs = {key: value for key, value in kwargs.items() if key in hub_keys}
        if config is None:
            config = Qwen3ASRConfig.from_pretrained(
                pretrained_model_name_or_path,
                **hub_kwargs,
            )

        model = Qwen3ASRForTokenClassification.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            **kwargs,
        )
        model.eval()
        compile_enabled = compile_model_forward(
            model,
            enabled=torch_compile,
            backend=torch_compile_backend,
            mode=torch_compile_mode,
            fullgraph=torch_compile_fullgraph,
            dynamic=torch_compile_dynamic,
            label="forced aligner",
        )
        processor = Qwen3ASRProcessor.from_pretrained(
            pretrained_model_name_or_path,
            fix_mistral_regex=True,
            **hub_kwargs,
        )
        return cls(
            model=model,
            processor=processor,
            torch_compile_enabled=compile_enabled,
        )

    def warm_up(
        self,
        *,
        audio: AudioLike,
        text: str,
        language: str,
    ) -> None:
        self.align(audio=audio, text=text, language=language)

    @staticmethod
    def _to_structured_items(timestamp_output: List[Any]) -> ForcedAlignResult:
        items: List[ForcedAlignItem] = []
        for item in timestamp_output:
            if isinstance(item, dict):
                text = item.get("text", "")
                start_time = item.get("start_time", 0)
                end_time = item.get("end_time", 0)
            else:
                text = getattr(item, "text", "")
                start_time = getattr(item, "start_time", 0)
                end_time = getattr(item, "end_time", 0)
            items.append(
                ForcedAlignItem(
                    text=str(text),
                    start_time=float(start_time),
                    end_time=float(end_time),
                )
            )
        return ForcedAlignResult(items=items)

    def _decode_forced_alignment(
        self,
        *,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        word_lists: List[List[str]],
    ) -> List[List[dict[str, Any]]]:
        """Decode aligner logits using the stable subquadratic repair path."""
        timestamp_segment_time = float(self.processor.timestamp_segment_time)
        decoded: List[List[dict[str, Any]]] = []

        for sample_index, words in enumerate(word_lists):
            timestamp_mask = input_ids[sample_index] == self.timestamp_token_id
            timestamp_logits = logits[sample_index][timestamp_mask]
            timestamp_bins = _decode_timestamp_bins(timestamp_logits)
            fixed_milliseconds = [value * timestamp_segment_time for value in timestamp_bins]
            decoded.append(
                [
                    {
                        "text": word,
                        "start_time": round(fixed_milliseconds[word_index * 2] / 1000.0, 3),
                        "end_time": round(fixed_milliseconds[word_index * 2 + 1] / 1000.0, 3),
                    }
                    for word_index, word in enumerate(words)
                ]
            )

        return decoded

    @torch.inference_mode()
    def align(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        text: Union[str, List[str]],
        language: Union[str, List[str]],
    ) -> List[ForcedAlignResult]:
        texts = ensure_list(text)
        languages = ensure_list(language)
        audios = normalize_audios(audio)

        if len(texts) == 1 and len(audios) > 1:
            texts = texts * len(audios)
        if len(languages) == 1 and len(audios) > 1:
            languages = languages * len(audios)
        if not (len(audios) == len(texts) == len(languages)):
            raise ValueError(
                f"Batch size mismatch: audio={len(audios)}, text={len(texts)}, "
                f"language={len(languages)}"
            )

        inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
            audio=audios,
            transcript=texts,
            language=languages,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        outputs = self.model(**inputs)
        decoded = self._decode_forced_alignment(
            logits=outputs.logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
        )
        return [self._to_structured_items(items) for items in decoded]

    def get_supported_languages(self) -> Optional[List[str]]:
        fn = getattr(self.model, "get_support_languages", None)
        if not callable(fn):
            return None
        languages = fn()
        if languages is None:
            return None
        return sorted({str(language).lower() for language in languages})
