"""Private Lab adapter manifests and trust-boundary metadata."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from frontierwright.errors import FrontierwrightError
from frontierwright.execution import BackendDataBoundary


class LabAdapterKind(StrEnum):
    MODEL_SOURCE = "MODEL_SOURCE"
    DATASET_SOURCE = "DATASET_SOURCE"
    TRAINER = "TRAINER"
    EXECUTOR = "EXECUTOR"
    CLUSTER_EXECUTOR = "CLUSTER_EXECUTOR"
    EVALUATOR = "EVALUATOR"
    ROLLOUT_ENGINE = "ROLLOUT_ENGINE"
    REWARD_PROVIDER = "REWARD_PROVIDER"
    ARTIFACT_STORE = "ARTIFACT_STORE"


class NetworkScope(StrEnum):
    OFFLINE = "OFFLINE"
    PRIVATE_ONLY = "PRIVATE_ONLY"
    EXTERNAL_ALLOWED = "EXTERNAL_ALLOWED"


@dataclass(frozen=True)
class LabAdapterManifest:
    adapter_id: str
    adapter_version: str
    display_name: str
    kinds: tuple[LabAdapterKind, ...]
    data_boundary: BackendDataBoundary
    network_scope: NetworkScope
    capabilities: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Lab adapter schema_version must be 1")
        for label, value in (
            ("adapter_id", self.adapter_id),
            ("adapter_version", self.adapter_version),
            ("display_name", self.display_name),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be nonempty")
            if "\x00" in value:
                raise ValueError(f"{label} must not contain NUL")
        if not self.kinds:
            raise ValueError("Lab adapter must declare at least one kind")
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError("Lab adapter kinds must be unique")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("Lab adapter capabilities must be unique")
        for capability in self.capabilities:
            if not capability.strip() or "\x00" in capability:
                raise ValueError("Lab adapter capabilities must be nonempty and NUL-free")
        if self.data_boundary is BackendDataBoundary.UNKNOWN:
            raise ValueError("Lab adapter data boundary cannot be UNKNOWN")

    @property
    def adapter_ref(self) -> str:
        return f"{self.adapter_id}@{self.adapter_version}"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "display_name": self.display_name,
            "kinds": sorted(item.value for item in self.kinds),
            "data_boundary": self.data_boundary.value,
            "network_scope": self.network_scope.value,
            "capabilities": sorted(self.capabilities),
        }

    @property
    def manifest_hash(self) -> str:
        data = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _strict_string_list(raw: object, label: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"{label} must be a list of strings")
    return tuple(raw)


def load_lab_adapter_manifest(path: Path) -> LabAdapterManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrontierwrightError(
            "LAB_ADAPTER_MANIFEST_INVALID",
            f"Lab adapter manifest is unreadable: {path}",
            2,
        ) from exc

    if not isinstance(raw, dict):
        raise FrontierwrightError(
            "LAB_ADAPTER_MANIFEST_INVALID",
            "Lab adapter manifest must be a JSON object.",
            2,
        )

    forbidden = {
        "token",
        "password",
        "secret",
        "api_key",
        "apiKey",
        "credential",
        "credentials",
    }
    if forbidden.intersection(raw):
        raise FrontierwrightError(
            "LAB_ADAPTER_SECRET_FORBIDDEN",
            "Lab adapter manifests must not contain credentials or secrets.",
            2,
        )

    try:
        if raw.get("schema_version") != 1:
            raise ValueError("schema_version must be 1")
        kinds = tuple(
            LabAdapterKind(item) for item in _strict_string_list(raw.get("kinds"), "kinds")
        )
        capabilities = _strict_string_list(
            raw.get("capabilities", []),
            "capabilities",
        )
        manifest = LabAdapterManifest(
            adapter_id=str(raw["adapter_id"]),
            adapter_version=str(raw["adapter_version"]),
            display_name=str(raw["display_name"]),
            kinds=kinds,
            data_boundary=BackendDataBoundary(str(raw["data_boundary"])),
            network_scope=NetworkScope(str(raw["network_scope"])),
            capabilities=capabilities,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FrontierwrightError(
            "LAB_ADAPTER_MANIFEST_INVALID",
            f"Invalid Lab adapter manifest: {exc}",
            2,
        ) from exc

    return manifest


def validate_manifest_runtime_metadata(value: object) -> None:
    """Reject non-finite JSON-like metadata before it can become audit evidence."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("runtime metadata must contain only finite numbers")
        return
    if isinstance(value, list):
        for item in value:
            validate_manifest_runtime_metadata(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("runtime metadata keys must be strings")
            validate_manifest_runtime_metadata(item)
        return
    raise ValueError("runtime metadata must be JSON-compatible")
