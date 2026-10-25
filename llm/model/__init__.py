"""Model architectures. Importing this package registers every architecture;
callers use `build_model(arch, ...)` / `architecture(arch)` and the
`LanguageModel` contract, never a concrete class.

To add one: a module here with a `ModelConfig` subclass and a `LanguageModel`
subclass decorated `@register("name")`, an import below, and a
`train.model.<name>` section in params.yaml for its own knobs.
"""

from llm.model import gpt as _gpt, rwkv as _rwkv, ssm as _ssm  # noqa: F401  (registration side effect)
from llm.model.base import LanguageModel, ModelConfig
from llm.model.registry import ARCHITECTURES, architecture, build_model

__all__ = ["ARCHITECTURES", "LanguageModel", "ModelConfig", "architecture", "build_model"]
