from __future__ import annotations

from typing import Any, Protocol


class ASRRuntime(Protocol):
    """Inference operations required by the product server."""

    forced_aligner: Any

    def transcribe(self, **kwargs: Any) -> Any:
        pass

    def init_streaming_state(self, **kwargs: Any) -> Any:
        pass

    def streaming_transcribe(self, audio: Any, state: Any) -> Any:
        pass

    def finish_streaming_transcribe(self, state: Any) -> Any:
        pass

    def warm_up(self, **kwargs: Any) -> None:
        pass
