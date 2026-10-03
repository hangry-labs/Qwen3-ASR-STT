from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner
from qwen_asr.inference.compile_utils import compile_model_forward


class _Tokenizer:
    @staticmethod
    def encode(text: str) -> list[int]:
        return list(range(len(text.split())))

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return "prefix " * len(token_ids)


class _ASRProcessor:
    tokenizer = _Tokenizer()

    @staticmethod
    def apply_chat_template(*_args, **_kwargs) -> str:
        return "prompt:"


class _CharacterTokenizer:
    @staticmethod
    def encode(text: str, **_kwargs) -> list[int]:
        return [ord(character) for character in text]

    @staticmethod
    def decode(token_ids: list[int], **_kwargs) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class _CharacterASRProcessor(_ASRProcessor):
    tokenizer = _CharacterTokenizer()


class _ASRModel:
    device = torch.device("cpu")
    dtype = torch.float32

    @staticmethod
    def parameters():
        return iter(())


class _VLLMModel:
    def __init__(self):
        self.generate = Mock(
            return_value=[SimpleNamespace(outputs=[SimpleNamespace(text="language English<asr_text>Hello")])]
        )


class _SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Batch(dict):
    def to(self, *_args, **_kwargs):
        return self


class _AlignerProcessor:
    timestamp_segment_time = 80

    def __init__(self):
        self.prepared = None

    def prepare_forced_aligner_inputs(self, *, audio, transcript, language):
        self.prepared = (audio, transcript, language)
        return _Batch(input_ids=torch.tensor([[42, 42], [42, 42]])), [["one"], ["two"]]


class _AlignerModel:
    device = torch.device("cpu")
    dtype = torch.float32
    config = SimpleNamespace(timestamp_token_id=42)

    @staticmethod
    def parameters():
        return iter(())

    @staticmethod
    def __call__(**_inputs):
        logits = torch.full((2, 2, 5), -1.0)
        logits[0, 0, 1] = 1.0
        logits[0, 1, 2] = 1.0
        logits[1, 0, 3] = 1.0
        logits[1, 1, 4] = 1.0
        return SimpleNamespace(logits=logits)


class NativeInferenceTests(unittest.TestCase):
    def test_aligner_loader_uses_native_qwen_classes_after_vllm_registration(self):
        config = object()
        model = Mock(
            device=torch.device("cpu"),
            config=SimpleNamespace(timestamp_token_id=42),
        )
        processor = object()

        with (
            patch(
                "qwen_asr.inference.qwen3_forced_aligner.Qwen3ASRConfig.from_pretrained",
                return_value=config,
            ) as load_config,
            patch(
                "qwen_asr.inference.qwen3_forced_aligner.Qwen3ASRForTokenClassification.from_pretrained",
                return_value=model,
            ) as load_model,
            patch(
                "qwen_asr.inference.qwen3_forced_aligner.Qwen3ASRProcessor.from_pretrained",
                return_value=processor,
            ) as load_processor,
            patch(
                "qwen_asr.inference.qwen3_forced_aligner.compile_model_forward",
                return_value=False,
            ),
        ):
            aligner = Qwen3ForcedAligner.from_pretrained(
                "Qwen/test-aligner",
                dtype="bfloat16",
                device_map="cuda:0",
                revision="test-revision",
            )

        load_config.assert_called_once_with(
            "Qwen/test-aligner",
            revision="test-revision",
        )
        load_model.assert_called_once_with(
            "Qwen/test-aligner",
            config=config,
            dtype="bfloat16",
            device_map="cuda:0",
            revision="test-revision",
        )
        load_processor.assert_called_once_with(
            "Qwen/test-aligner",
            fix_mistral_regex=True,
            revision="test-revision",
        )
        model.eval.assert_called_once_with()
        self.assertIs(aligner.model, model)
        self.assertIs(aligner.processor, processor)

    def test_compile_model_forward_uses_requested_native_profile(self):
        model = SimpleNamespace(forward=Mock(name="forward"))
        compiled_forward = Mock(name="compiled_forward")

        with patch("qwen_asr.inference.compile_utils.torch.compile", return_value=compiled_forward) as compile_mock:
            enabled = compile_model_forward(
                model,
                enabled=True,
                backend="inductor",
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=True,
                label="test model",
            )

        self.assertTrue(enabled)
        model.forward(input_ids=torch.tensor([[1]]))
        compiled_forward.assert_called_once()
        compile_mock.assert_called_once_with(
            compile_mock.call_args.args[0],
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=True,
        )

    def test_reduce_overhead_marks_each_cudagraph_step(self):
        model = SimpleNamespace(forward=Mock(name="forward"))
        compiled_forward = Mock(name="compiled_forward")

        with (
            patch("qwen_asr.inference.compile_utils.torch.compile", return_value=compiled_forward),
            patch("qwen_asr.inference.compile_utils.torch.compiler.cudagraph_mark_step_begin") as mark_step,
        ):
            compile_model_forward(
                model,
                enabled=True,
                mode="reduce-overhead",
                label="test model",
            )
            model.forward(input_ids=torch.tensor([[1]]))
            model.forward(input_ids=torch.tensor([[2]]))

        self.assertEqual(mark_step.call_count, 2)
        self.assertEqual(compiled_forward.call_count, 2)

    def test_warmup_runs_requested_iterations_and_aligner(self):
        aligner = Mock()
        asr = Qwen3ASRModel(model=_ASRModel(), processor=_ASRProcessor(), forced_aligner=aligner)
        asr.transcribe = Mock()
        audio = (np.zeros(16_000, dtype=np.float32), 16_000)

        asr.warm_up(
            max_new_tokens=64,
            iterations=3,
            audio=audio,
            aligner_text="warmup transcript",
        )

        self.assertEqual(asr.transcribe.call_count, 3)
        self.assertEqual(
            [call.kwargs["language"] for call in asr.transcribe.call_args_list],
            [None, "English", None],
        )
        self.assertEqual(asr.max_new_tokens, 512)
        aligner.warm_up.assert_called_once_with(
            audio=audio,
            text="warmup transcript",
            language="English",
        )

    def test_generation_cache_implementation_is_forwarded(self):
        model = _ASRModel()
        model.generate = Mock(return_value=torch.tensor([[1]]))
        asr = Qwen3ASRModel(
            model=model,
            processor=_ASRProcessor(),
            generation_cache_implementation="static",
        )

        asr._generate(input_ids=torch.tensor([[1]]), use_cache=True)

        self.assertEqual(model.generate.call_args.kwargs["cache_implementation"], "static")
        self.assertTrue(model.generate.call_args.kwargs["use_cache"])

    def test_vllm_inference_uses_builtin_multimodal_prompt(self):
        model = _VLLMModel()
        sampling_params = object()
        asr = Qwen3ASRModel(
            backend="vllm",
            model=model,
            processor=_ASRProcessor(),
            sampling_params=sampling_params,
            max_inference_batch_size=2,
        )
        audio = np.zeros(16_000, dtype=np.float32)

        outputs = asr._infer_asr_vllm([""], [audio], ["English"])

        self.assertEqual(outputs, ["language English<asr_text>Hello"])
        request = model.generate.call_args.args[0][0]
        self.assertEqual(request["prompt"], "prompt:language English<asr_text>")
        self.assertIs(request["multi_modal_data"]["audio"][0], audio)
        self.assertIs(model.generate.call_args.kwargs["sampling_params"], sampling_params)
        self.assertFalse(model.generate.call_args.kwargs["use_tqdm"])

    def test_vllm_warmup_temporarily_uses_requested_token_limit(self):
        original_sampling_params = _SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop_token_ids=[151643, 151645],
        )
        asr = Qwen3ASRModel(
            backend="vllm",
            model=_VLLMModel(),
            processor=_ASRProcessor(),
            sampling_params=original_sampling_params,
        )
        observed_sampling_params = []

        def record_transcribe(**_kwargs):
            observed_sampling_params.append(asr.sampling_params)

        asr.transcribe = Mock(side_effect=record_transcribe)
        asr.warm_up(max_new_tokens=64, iterations=2)

        self.assertEqual([params.max_tokens for params in observed_sampling_params], [64, 64])
        self.assertEqual(observed_sampling_params[0].stop_token_ids, [151643, 151645])
        self.assertIs(asr.sampling_params, original_sampling_params)

    def test_streaming_waits_for_a_full_chunk_before_inference(self):
        asr = Qwen3ASRModel(model=_ASRModel(), processor=_ASRProcessor())
        asr._generate_streaming_text = Mock(return_value="language English<asr_text>Hello")
        state = asr.init_streaming_state(chunk_size_sec=2.0)
        one_second = np.zeros(16_000, dtype=np.float32)

        asr.streaming_transcribe(one_second, state)
        self.assertEqual(state.chunk_id, 0)
        asr._generate_streaming_text.assert_not_called()

        asr.streaming_transcribe(one_second, state)
        self.assertEqual(state.chunk_id, 1)
        self.assertEqual(state.language, "English")
        self.assertEqual(state.text, "Hello")
        asr._generate_streaming_text.assert_called_once()

    def test_streaming_caps_audio_window_and_stitches_stable_transcript(self):
        asr = Qwen3ASRModel(model=_ASRModel(), processor=_CharacterASRProcessor())
        asr._generate_streaming_text = Mock(
            side_effect=[
                "language English<asr_text>alpha beta",
                "language English<asr_text>alpha beta gamma delta",
                "delta epsilon zeta",
                " zeta eta theta",
            ]
        )
        state = asr.init_streaming_state(chunk_size_sec=2.0, max_window_sec=4.0)
        two_seconds = np.zeros(32_000, dtype=np.float32)

        asr.streaming_transcribe(two_seconds, state)
        asr.streaming_transcribe(two_seconds, state)
        asr.streaming_transcribe(two_seconds, state)
        asr.streaming_transcribe(two_seconds, state)

        self.assertEqual(state.audio_samples_seen, 128_000)
        self.assertEqual(len(state.audio_accum), 64_000)
        self.assertEqual(state.text, "alpha beta gamma delta epsilon zeta eta theta")
        self.assertEqual(state.language, "English")
        third_prompt, third_audio = asr._generate_streaming_text.call_args_list[2].args
        self.assertEqual(third_prompt, "prompt:language English<asr_text>gamma ")
        self.assertEqual(len(third_audio), 64_000)
        fourth_prompt, fourth_audio = asr._generate_streaming_text.call_args_list[3].args
        self.assertEqual(fourth_prompt, "prompt:language English<asr_text>epsilon")
        self.assertEqual(len(fourth_audio), 64_000)

    def test_streaming_window_must_not_be_shorter_than_chunk(self):
        asr = Qwen3ASRModel(model=_ASRModel(), processor=_ASRProcessor())

        with self.assertRaisesRegex(ValueError, "max_window_sec"):
            asr.init_streaming_state(chunk_size_sec=2.0, max_window_sec=1.0)

    def test_native_aligner_broadcasts_text_and_language(self):
        processor = _AlignerProcessor()
        aligner = Qwen3ForcedAligner(model=_AlignerModel(), processor=processor)
        audio = [
            (np.zeros(16_000, dtype=np.float32), 16_000),
            (np.zeros(16_000, dtype=np.float32), 16_000),
        ]

        results = aligner.align(audio=audio, text="sample", language="English")

        self.assertEqual(processor.prepared[1], ["sample", "sample"])
        self.assertEqual(processor.prepared[2], ["English", "English"])
        self.assertEqual(results[0][0].text, "one")
        self.assertEqual(results[0][0].start_time, 0.08)
        self.assertEqual(results[1][0].start_time, 0.24)


if __name__ == "__main__":
    unittest.main()
