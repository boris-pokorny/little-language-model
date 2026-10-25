"""Name -> architecture lookup. Architectures register themselves with
`@register("name")`; callers build models by name and never enumerate classes."""

from typing import Any, Callable

from llm.model.base import LanguageModel

ARCHITECTURES: dict[str, type[LanguageModel]] = {}


def register(name: str) -> Callable[[type[LanguageModel]], type[LanguageModel]]:
    def wrap(cls: type[LanguageModel]) -> type[LanguageModel]:
        if name in ARCHITECTURES:
            raise ValueError(f"architecture {name!r} is already registered")
        ARCHITECTURES[name] = cls
        return cls

    return wrap


def architecture(name: str) -> type[LanguageModel]:
    try:
        return ARCHITECTURES[name]
    except KeyError:
        raise ValueError(f"unknown architecture {name!r}; known: {sorted(ARCHITECTURES)}") from None


def build_model(
    arch: str, vocab_size: int, common: dict[str, Any], arch_params: dict[str, Any]
) -> LanguageModel:
    """`common` holds ModelConfig fields, `arch_params` the architecture's own.
    Unknown keys fail loudly in the config dataclass constructor."""
    cls = architecture(arch)
    return cls(cls.config_cls(vocab_size=vocab_size, **common, **arch_params))
