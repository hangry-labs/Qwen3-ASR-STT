# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch
from transformers import AutoModelForTokenClassification, AutoProcessor

from .compile_utils import compile_model_forward
from .utils import AudioLike, ensure_list, normalize_audios


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
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path,
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
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            fix_mistral_regex=True,
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
        decoded = self.processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self.timestamp_token_id,
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
