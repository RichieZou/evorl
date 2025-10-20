from collections.abc import Mapping
from typing import Any

import jax.tree_util as jtu
import pandas as pd
import swanlab

from .recorder import Recorder


class SwanlabRecorder(Recorder):
    """Recorder for Swanlab (decoupled from WandB)."""

    def __init__(self, *, project, name, config, tags, path, **swanlab_kwargs):
        self.swanlab_kwargs = {
            "project": project,
            "name": name,
            "config": config,
            "tags": tags,
            "dir": path,
            **swanlab_kwargs,
        }

    def init(self) -> None:
        try:
            swanlab.init(**self.swanlab_kwargs)
        except Exception:
            pass

    def write(self, data: Mapping[str, Any], step: int | None = None) -> None:
        data = jtu.tree_map(lambda x: _convert_data(x), data)
        try:
            swanlab.log(data, step=step)
        except Exception:
            pass

    def close(self) -> None:
        try:
            swanlab.finish()
        except Exception:
            pass


def _convert_data(val: Any):
    if isinstance(val, pd.Series):
        # Swanlab can accept pandas Series as list; keep as-is
        return val
    elif isinstance(val, pd.DataFrame):
        # Convert DataFrame to dict records for Swanlab
        return val.to_dict(orient="list")
    else:
        return val


