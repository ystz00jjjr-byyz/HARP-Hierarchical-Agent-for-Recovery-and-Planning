"""Rule-based post-condition checks for common CaP-X primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from capx.recovery.schema import PlannedAction, ValidationResult

Rule = Callable[[Mapping[str, Any], PlannedAction], tuple[bool, str, tuple[str, ...]]]


def _value(state: Mapping[str, Any], key: str) -> Any:
    current: Any = state
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first(state: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[str | None, Any]:
    for key in keys:
        value = _value(state, key)
        if value is not None:
            return key, value
    return None, None


def _bool_rule(keys: tuple[str, ...], label: str) -> Rule:
    def check(state: Mapping[str, Any], _action: PlannedAction) -> tuple[bool, str, tuple[str, ...]]:
        key, value = _first(state, keys)
        if key is None:
            return False, f"missing world-state signal for {label}", keys
        if bool(value):
            return True, f"{key}=true", ()
        return False, f"{key} is not true", ()
    return check


def _numeric_rule(keys: tuple[str, ...], label: str, default_limit: float) -> Rule:
    def check(state: Mapping[str, Any], action: PlannedAction) -> tuple[bool, str, tuple[str, ...]]:
        key, value = _first(state, keys)
        if key is None:
            return False, f"missing world-state signal for {label}", keys
        try:
            observed = float(value)
            limit = float(action.expected.get(f"{label}_threshold", default_limit))
        except (TypeError, ValueError):
            return False, f"{key} is not numeric: {value!r}", ()
        ok = observed <= limit
        return ok, f"{key}={observed:g} <= {limit:g}", ()
    return check


def _alignment_rule(state: Mapping[str, Any], action: PlannedAction):
    key, value = _first(state, ("eef_aligned", "target_aligned"))
    if key is not None:
        return bool(value), f"{key}={value!r}", ()
    return _numeric_rule(("alignment_error", "eef_alignment_error"), "alignment", 0.02)(state, action)


def _orientation_rule(state: Mapping[str, Any], action: PlannedAction):
    key, value = _first(state, ("orientation_aligned",))
    if key is not None:
        return bool(value), f"{key}={value!r}", ()
    return _numeric_rule(("orientation_error",), "orientation", 0.15)(state, action)


def _check_expected(expected: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[bool, str, tuple[str, ...]]:
    for key, target in expected.items():
        if key.endswith("_threshold"):
            continue
        actual = _value(state, key)
        if actual is None:
            return False, f"missing expected signal '{key}'", (key,)
        if isinstance(target, Mapping):
            if "lte" in target and float(actual) > float(target["lte"]):
                return False, f"{key}={actual!r} exceeds lte {target['lte']!r}", ()
            if "gte" in target and float(actual) < float(target["gte"]):
                return False, f"{key}={actual!r} is below gte {target['gte']!r}", ()
            if "eq" in target and actual != target["eq"]:
                return False, f"{key}={actual!r} != {target['eq']!r}", ()
        elif actual != target:
            return False, f"{key}={actual!r} != {target!r}", ()
    return True, "explicit expectations satisfied", ()


def default_rules() -> dict[str, Rule]:
    return {
        "move_to": _alignment_rule,
        "goto_pose": _alignment_rule,
        "align": _alignment_rule,
        "grasp": _bool_rule(("target_grasped", "object_grasped"), "grasp"),
        "close_gripper": _bool_rule(("gripper_closed",), "close_gripper"),
        "open_gripper": _bool_rule(("gripper_open",), "open_gripper"),
        "place": _bool_rule(("object_at_target", "placed"), "place"),
        "release": _bool_rule(("gripper_open",), "release"),
        "rotate": _orientation_rule,
    }


class RuleBasedValidator:
    def __init__(self, rules: Mapping[str, Rule] | None = None) -> None:
        self.rules = dict(rules or default_rules())

    def validate(self, action: PlannedAction, state: Mapping[str, Any]) -> ValidationResult:
        primitive = action.primitive.strip().lower()
        rule = self.rules.get(primitive)
        if rule is None:
            return ValidationResult(False, action.primitive, action.action_id, "unknown_primitive",
                                    f"no rule configured for primitive '{action.primitive}'")
        ok, message, missing = rule(state, action)
        expected_ok, expected_message, expected_missing = _check_expected(action.expected, state)
        if ok and not expected_ok:
            ok, message, missing = False, expected_message, expected_missing
        return ValidationResult(ok, action.primitive, action.action_id, primitive, message,
                                tuple(dict.fromkeys((*missing, *expected_missing))), dict(state))


default_validator = RuleBasedValidator()
