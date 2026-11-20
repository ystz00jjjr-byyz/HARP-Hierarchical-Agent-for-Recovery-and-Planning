"""Deterministic marker layouts for the RSI spill-wipe benchmark."""

from __future__ import annotations

import numpy as np


SCENARIOS = {
    "fixed_linear_wipe",
    "wipe_front_regions",
    "wipe_back_right_region",
    "wipe_strip_and_patch",
    "wipe_held_out_l_shape",
}


def marker_xy(scenario_id: str, seed: int, count: int = 100) -> np.ndarray:
    """Return deterministic table-local marker positions in meters."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown spill-wipe scenario: {scenario_id}")
    if count < 2:
        raise ValueError("count must be >= 2")
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0.0, 0.002, size=(count, 2))

    if scenario_id == "fixed_linear_wipe":
        xy = _line(count, (-0.12, 0.0), (0.12, 0.0))
    elif scenario_id == "wipe_front_regions":
        center = (-0.05, -0.16) if seed % 2 else (0.05, -0.16)
        xy = _line(count, (center[0] - 0.10, center[1]), (center[0] + 0.10, center[1]))
    elif scenario_id == "wipe_back_right_region":
        xy = _line(count, (0.02, 0.16), (0.20, 0.16))
    elif scenario_id == "wipe_strip_and_patch":
        if seed % 2:
            xy = _line(count, (-0.10, 0.0), (0.10, 0.0))
        else:
            xy = _patch(count, (0.0, 0.0))
    else:
        first = count // 2
        xy = np.vstack([
            _line(first, (-0.10, 0.10), (0.10, 0.10)),
            _line(count - first, (-0.10, 0.10), (-0.10, -0.10)),
        ])
    return np.clip(xy + jitter, (-0.22, -0.34), (0.22, 0.34))


def _line(count: int, start: tuple[float, float], end: tuple[float, float]) -> np.ndarray:
    return np.column_stack([
        np.linspace(start[0], end[0], count),
        np.linspace(start[1], end[1], count),
    ])


def _patch(count: int, center: tuple[float, float]) -> np.ndarray:
    side = int(np.ceil(np.sqrt(count)))
    axis = np.linspace(-0.045, 0.045, side)
    grid = np.array([(center[0] + x, center[1] + y) for x in axis for y in axis])
    return grid[:count]


__all__ = ["SCENARIOS", "marker_xy"]
