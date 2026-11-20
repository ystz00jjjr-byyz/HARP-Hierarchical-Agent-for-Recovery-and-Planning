"""JSONL metadata with NPZ array sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class JsonlNpzSink:
    def __init__(self, output_dir: str | Path, *, compress_arrays: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.array_dir = self.output_dir / "arrays"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.array_dir.mkdir(parents=True, exist_ok=True)
        self._stream = (self.output_dir / "trajectory.jsonl").open("w", encoding="utf-8")
        self.compress_arrays = compress_arrays
        self.closed = False

    def write(self, entry: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        if self.closed:
            return
        if arrays:
            array_file = f"arrays/record_{entry['record_index']:08d}.npz"
            save = np.savez_compressed if self.compress_arrays else np.savez
            save(self.output_dir / array_file, **arrays)
            entry["array_file"] = array_file
        else:
            entry["array_file"] = None
        self._stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if not self.closed:
            self._stream.close()
            self.closed = True
