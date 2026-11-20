"""Small, text-first recovery harnesses for CaP-X multi-turn execution."""

from capx.recovery.harness import ThreeStepRecoveryHarness, build_recovery_prompt
from capx.recovery.planner import ThreeStepWindowPlanner
from capx.recovery.scoring import PrimitiveStepScorer, ScoreWeights
from capx.recovery.stability import (
    StateWindowAggregator,
    StateWindowConfig,
    StateWindowSummary,
)
from capx.recovery.schema import (
    ExecutionOutcome,
    HarnessEvent,
    HarnessResult,
    PlannedAction,
    StepScore,
    ValidationResult,
)
from capx.recovery.validators import RuleBasedValidator, default_validator
from capx.recovery.execution_guard import (
    ExecutionGuardConfig,
    GuardResult,
    assign_stage_indices,
    format_guard_feedback,
    task_completion_confirmed,
    validate_code_block,
    validate_decision,
    validate_stage,
)
from capx.recovery.offline import (
    OfflineAction,
    OfflineExecution,
    OfflineEvent,
    OfflineRunResult,
    OfflineTaskGraph,
    OfflineWindowController,
    RecoveryLibrary,
    StateParameterError,
    bind_state_parameters,
)

__all__ = [
    "ExecutionOutcome", "HarnessEvent", "HarnessResult", "PlannedAction", "PrimitiveStepScorer",
    "RuleBasedValidator", "ScoreWeights", "StateWindowAggregator", "StateWindowConfig",
    "StateWindowSummary", "StepScore", "ThreeStepRecoveryHarness",
    "ThreeStepWindowPlanner", "ValidationResult",
    "ExecutionGuardConfig", "GuardResult", "assign_stage_indices",
    "build_recovery_prompt", "default_validator", "format_guard_feedback",
    "task_completion_confirmed", "validate_code_block", "validate_decision", "validate_stage",
    "OfflineAction", "OfflineExecution", "OfflineEvent", "OfflineRunResult",
    "OfflineTaskGraph", "OfflineWindowController", "RecoveryLibrary",
    "StateParameterError", "bind_state_parameters",
]
