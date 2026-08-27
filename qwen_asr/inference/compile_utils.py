from __future__ import annotations

from functools import wraps
from typing import Any

import torch

from qwen_asr.startup_logging import StartupTimer, log_startup


def compile_model_forward(
    model: Any,
    *,
    enabled: bool,
    backend: str = "inductor",
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool | None = None,
    label: str,
) -> bool:
    """Compile a model forward pass using the native PyTorch compiler."""
    if not enabled:
        log_startup(f"{label} torch.compile disabled")
        return False

    compile_kwargs: dict[str, Any] = {
        "backend": backend,
        "mode": mode,
        "fullgraph": fullgraph,
    }
    if dynamic is not None:
        compile_kwargs["dynamic"] = dynamic

    log_startup(f"{label} torch.compile configuration: {compile_kwargs}")
    with StartupTimer(f"wrap {label} forward with torch.compile"):
        compiled_forward = torch.compile(model.forward, **compile_kwargs)
        if mode in {"reduce-overhead", "max-autotune"}:
            @torch.compiler.disable(recursive=False)
            @wraps(compiled_forward)
            def forward_with_cudagraph_step(*args: Any, **kwargs: Any) -> Any:
                torch.compiler.cudagraph_mark_step_begin()
                return compiled_forward(*args, **kwargs)

            model.forward = forward_with_cudagraph_step
        else:
            model.forward = compiled_forward
    return True
