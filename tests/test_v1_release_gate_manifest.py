from pathlib import Path

from tools.v1_release_gate import CANONICAL_CRITERIA


def test_canonical_v1_gate_has_twenty_unique_evidenced_criteria() -> None:
    assert len(CANONICAL_CRITERIA) == 20
    ids = [item.criterion_id for item in CANONICAL_CRITERIA]
    assert ids == [f"V1-{index:02d}" for index in range(1, 21)]
    assert len(set(ids)) == 20

    root = Path(__file__).resolve().parents[1]
    virtual_evidence = {"pytest"}
    for item in CANONICAL_CRITERIA:
        assert item.requirement.strip()
        assert item.evidence
        assert item.to_dict()["status"] == "PASS"
        for value in item.evidence:
            assert value.strip()
            path_text = value.split("::", 1)[0]
            if path_text in virtual_evidence:
                continue
            assert (root / path_text).exists(), (
                f"{item.criterion_id} references missing evidence: {value}"
            )
