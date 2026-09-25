"""Versioned reinforcement-learning experiment contracts.

Frontierwright uses "reinforcement learning" only for experiments with an explicit
environment, reward source, rollout policy, and policy-optimization algorithm.
Preference optimization such as DPO is intentionally separate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from frontierwright.errors import FrontierwrightError

RL_SPEC_SCHEMA_VERSION = 1
REFERENCE_RL_ALGORITHM = "reinforce"
REFERENCE_RL_ENVIRONMENT_KIND = "VERIFIABLE_MULTIPLE_CHOICE"
REFERENCE_RL_REWARD_KIND = "EXACT_CORRECT_CHOICE"
DIRECT_OBSERVATION_REWARD_KINDS = frozenset(
    {"USAGE_OBSERVATION", "OPERATIONAL_OUTCOME", "USAGE_SUCCESS_RATE"}
)


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise FrontierwrightError(
            "INVALID_RL_SPEC",
            f"{label} must be nonempty NUL-free text.",
            2,
        )
    return value.strip()


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrontierwrightError(
            "INVALID_RL_SPEC",
            f"{label} must be a positive finite number.",
            2,
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise FrontierwrightError(
            "INVALID_RL_SPEC",
            f"{label} must be a positive finite number.",
            2,
        )
    return result


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrontierwrightError(
            "INVALID_RL_SPEC",
            f"{label} must be a nonnegative finite number.",
            2,
        )
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise FrontierwrightError(
            "INVALID_RL_SPEC",
            f"{label} must be a nonnegative finite number.",
            2,
        )
    return result


def _json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FrontierwrightError(
            "INVALID_RL_SPEC",
            f"{label} must be a JSON object.",
            2,
        )
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        canonical = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise FrontierwrightError(
            "INVALID_RL_SPEC",
            f"{label} must contain only finite JSON-compatible values.",
            2,
        ) from exc
    assert isinstance(canonical, dict)
    return canonical


@dataclass(frozen=True)
class RLIdentity:
    """Immutable identity for an environment or reward provider."""

    identity_id: str
    version: str
    kind: str
    config: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_id", _required_text(self.identity_id, "identity_id"))
        object.__setattr__(self, "version", _required_text(self.version, "version"))
        object.__setattr__(self, "kind", _required_text(self.kind, "kind"))
        object.__setattr__(self, "config", _json_object(self.config, "identity config"))

    def to_payload(self) -> dict[str, object]:
        return {
            "id": self.identity_id,
            "version": self.version,
            "kind": self.kind,
            "config": dict(self.config),
        }


@dataclass(frozen=True)
class RLExperimentSpec:
    """Pinned semantics for one real RL policy-optimization experiment."""

    algorithm_id: str
    environment: RLIdentity
    reward: RLIdentity
    rollout_temperature: float = 1.0
    entropy_coefficient: float = 0.0
    algorithm_config: dict[str, object] | None = None
    schema_version: int = RL_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RL_SPEC_SCHEMA_VERSION:
            raise FrontierwrightError(
                "INVALID_RL_SPEC",
                f"RL spec schema_version must be {RL_SPEC_SCHEMA_VERSION}.",
                2,
            )
        object.__setattr__(
            self,
            "algorithm_id",
            _required_text(self.algorithm_id, "algorithm_id").lower(),
        )
        object.__setattr__(
            self,
            "rollout_temperature",
            _finite_positive(self.rollout_temperature, "rollout_temperature"),
        )
        object.__setattr__(
            self,
            "entropy_coefficient",
            _finite_nonnegative(self.entropy_coefficient, "entropy_coefficient"),
        )
        object.__setattr__(
            self,
            "algorithm_config",
            _json_object(self.algorithm_config or {}, "algorithm_config"),
        )
        reward_kind = self.reward.kind.upper()
        source_kind = self.reward.config.get("source_kind")
        normalized_source = source_kind.upper() if isinstance(source_kind, str) else None
        if (
            reward_kind in DIRECT_OBSERVATION_REWARD_KINDS
            or normalized_source in DIRECT_OBSERVATION_REWARD_KINDS
        ):
            raise FrontierwrightError(
                "RL_OBSERVATION_REWARD_REQUIRES_TRANSFORM",
                (
                    "Operational usage observations cannot be used directly as RL rewards. "
                    "Create and validate a separate versioned reward/verifier transformation "
                    "with explicit provenance first."
                ),
                2,
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "algorithm_id": self.algorithm_id,
            "environment": self.environment.to_payload(),
            "reward": self.reward.to_payload(),
            "rollout_temperature": self.rollout_temperature,
            "entropy_coefficient": self.entropy_coefficient,
            "algorithm_config": dict(self.algorithm_config or {}),
        }

    @property
    def spec_hash(self) -> str:
        raw = json.dumps(
            self.to_payload(),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(raw).hexdigest()


def _identity_from_payload(raw: object, label: str) -> RLIdentity:
    if not isinstance(raw, dict):
        raise FrontierwrightError(
            "INVALID_RL_SPEC",
            f"{label} must be an object.",
            2,
        )
    return RLIdentity(
        identity_id=_required_text(raw.get("id"), f"{label}.id"),
        version=_required_text(raw.get("version"), f"{label}.version"),
        kind=_required_text(raw.get("kind"), f"{label}.kind"),
        config=_json_object(raw.get("config", {}), f"{label}.config"),
    )


def rl_spec_from_config(config: dict[str, object]) -> RLExperimentSpec:
    """Parse the nested rl object pinned inside a TrainingPlan config."""

    raw = config.get("rl")
    if not isinstance(raw, dict):
        raise FrontierwrightError(
            "INVALID_RL_SPEC",
            "RL_POLICY_OPTIMIZATION requires a config.rl object.",
            2,
        )
    allowed = {
        "schema_version",
        "algorithm_id",
        "environment",
        "reward",
        "rollout_temperature",
        "entropy_coefficient",
        "algorithm_config",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise FrontierwrightError(
            "INVALID_RL_SPEC",
            "Unsupported config.rl keys: " + ", ".join(unknown),
            2,
        )
    schema_version = raw.get("schema_version", RL_SPEC_SCHEMA_VERSION)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise FrontierwrightError(
            "INVALID_RL_SPEC",
            "config.rl.schema_version must be an integer.",
            2,
        )
    return RLExperimentSpec(
        schema_version=schema_version,
        algorithm_id=_required_text(raw.get("algorithm_id"), "config.rl.algorithm_id"),
        environment=_identity_from_payload(raw.get("environment"), "config.rl.environment"),
        reward=_identity_from_payload(raw.get("reward"), "config.rl.reward"),
        rollout_temperature=_finite_positive(
            raw.get("rollout_temperature", 1.0),
            "config.rl.rollout_temperature",
        ),
        entropy_coefficient=_finite_nonnegative(
            raw.get("entropy_coefficient", 0.0),
            "config.rl.entropy_coefficient",
        ),
        algorithm_config=_json_object(
            raw.get("algorithm_config", {}),
            "config.rl.algorithm_config",
        ),
    )


def validate_reference_rl_spec(spec: RLExperimentSpec) -> None:
    """Enforce the narrow built-in RL environment without restricting Lab adapters."""

    if spec.algorithm_id != REFERENCE_RL_ALGORITHM:
        raise FrontierwrightError(
            "REFERENCE_RL_UNSUPPORTED",
            "The built-in reference backend supports algorithm_id=reinforce only.",
            2,
        )
    if spec.environment.kind != REFERENCE_RL_ENVIRONMENT_KIND:
        raise FrontierwrightError(
            "REFERENCE_RL_UNSUPPORTED",
            "The built-in reference backend requires VERIFIABLE_MULTIPLE_CHOICE environment.",
            2,
        )
    if spec.reward.kind != REFERENCE_RL_REWARD_KIND:
        raise FrontierwrightError(
            "REFERENCE_RL_UNSUPPORTED",
            "The built-in reference backend requires EXACT_CORRECT_CHOICE reward.",
            2,
        )
    algorithm_config = spec.algorithm_config or {}
    unknown = sorted(set(algorithm_config) - {"reward_baseline"})
    if unknown:
        raise FrontierwrightError(
            "REFERENCE_RL_UNSUPPORTED",
            "Reference REINFORCE does not support algorithm_config keys: " + ", ".join(unknown),
            2,
        )
    baseline = algorithm_config.get("reward_baseline", 0.5)
    if (
        isinstance(baseline, bool)
        or not isinstance(baseline, (int, float))
        or not math.isfinite(float(baseline))
        or not 0.0 <= float(baseline) <= 1.0
    ):
        raise FrontierwrightError(
            "REFERENCE_RL_UNSUPPORTED",
            "Reference reward_baseline must be a finite number in [0, 1].",
            2,
        )
