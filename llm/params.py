"""Pipeline parameters: one params.yaml, read per stage.

`load_params("train")` returns that section as a dict. Each stage keeps its CLI
flags; `argparse` defaults are filled from params so a flag given on the command
line still wins (handy for `--steps 200` smoke tests), while DVC only sees the
file. Missing file or section falls back to the defaults the stage passes in.
"""

from pathlib import Path
from typing import Any

import yaml

PARAMS_PATH = Path("params.yaml")


def load_params(section: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = dict(defaults or {})
    if PARAMS_PATH.exists():
        loaded = yaml.safe_load(PARAMS_PATH.read_text()) or {}
        params.update(loaded.get(section) or {})
    return params
