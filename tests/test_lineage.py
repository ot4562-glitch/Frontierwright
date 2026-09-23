from pathlib import Path

import pytest

from frontierwright.domain import LineageRelation, ModelOrigin, ModelState
from frontierwright.errors import FrontierwrightError
from frontierwright.registry import Registry


def model(
    registry: Registry,
    *,
    model_id: str,
    parent_model_id: str | None = None,
) -> ModelState:
    state = registry.read()
    return ModelState(
        model_id=model_id,
        identity_id=state.project["identity_id"],
        origin=ModelOrigin.ZERO,
        checkpoint=f"checkpoint:{model_id}",
        fingerprint=f"sha256:{model_id}",
        parent_model_id=parent_model_id,
    )


def test_primary_parent_creates_derived_lineage_edge(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    parent = model(registry, model_id="parent")
    registry.register_candidate(parent)

    child = model(
        registry,
        model_id="child",
        parent_model_id=parent.model_id,
    )
    registry.register_candidate(child)

    lineage = registry.get_model_lineage(child.model_id)
    assert len(lineage) == 1
    assert lineage[0]["parent_model_id"] == parent.model_id
    assert lineage[0]["relation"] == "DERIVED_FROM"
    assert lineage[0]["ordinal"] == 0
    assert lineage[0]["details"] == {"source": "model.parent_model_id"}


def test_model_can_record_multiple_merge_parents_idempotently(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    primary = model(registry, model_id="primary")
    other = model(registry, model_id="other")
    registry.register_candidate(primary)
    registry.register_candidate(other)

    merged = model(
        registry,
        model_id="merged",
        parent_model_id=primary.model_id,
    )
    registry.register_candidate(merged)

    registry.record_model_lineage_edge(
        child_model_id=merged.model_id,
        parent_model_id=primary.model_id,
        relation=LineageRelation.MERGED_FROM,
        ordinal=0,
        details={"weight": 0.75},
    )
    registry.record_model_lineage_edge(
        child_model_id=merged.model_id,
        parent_model_id=other.model_id,
        relation=LineageRelation.MERGED_FROM,
        ordinal=1,
        details={"weight": 0.25},
    )
    # Identical replay is idempotent.
    registry.record_model_lineage_edge(
        child_model_id=merged.model_id,
        parent_model_id=other.model_id,
        relation=LineageRelation.MERGED_FROM,
        ordinal=1,
        details={"weight": 0.25},
    )

    lineage = registry.get_model_lineage(merged.model_id)
    merge_edges = [item for item in lineage if item["relation"] == "MERGED_FROM"]
    assert [(item["parent_model_id"], item["ordinal"]) for item in merge_edges] == [
        (primary.model_id, 0),
        (other.model_id, 1),
    ]
    assert merge_edges[0]["details"] == {"weight": 0.75}
    assert merge_edges[1]["details"] == {"weight": 0.25}


def test_conflicting_lineage_replay_is_rejected(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    parent = model(registry, model_id="parent")
    child = model(registry, model_id="child")
    registry.register_candidate(parent)
    registry.register_candidate(child)

    registry.record_model_lineage_edge(
        child_model_id=child.model_id,
        parent_model_id=parent.model_id,
        relation=LineageRelation.TRANSFORMED_FROM,
        ordinal=0,
        details={"method": "first"},
    )

    with pytest.raises(FrontierwrightError, match="incompatible lineage edge"):
        registry.record_model_lineage_edge(
            child_model_id=child.model_id,
            parent_model_id=parent.model_id,
            relation=LineageRelation.TRANSFORMED_FROM,
            ordinal=1,
            details={"method": "changed"},
        )


def test_lineage_self_reference_is_rejected(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize("NOVA", ModelOrigin.ZERO)

    current = model(registry, model_id="model")
    registry.register_candidate(current)

    with pytest.raises(FrontierwrightError, match="cannot be its own lineage parent"):
        registry.record_model_lineage_edge(
            child_model_id=current.model_id,
            parent_model_id=current.model_id,
            relation=LineageRelation.MERGED_FROM,
            ordinal=0,
        )
