from pathlib import Path

from tools.rc2_release_gate import RC2_CRITERIA


def test_rc2_gate_manifest_is_complete_and_grounded() -> None:
    assert len(RC2_CRITERIA) == 12
    assert [item.criterion_id for item in RC2_CRITERIA] == [
        f"RC2-{index:02d}" for index in range(1, 13)
    ]
    assert len({item.criterion_id for item in RC2_CRITERIA}) == len(RC2_CRITERIA)

    root = Path(__file__).resolve().parents[1]
    for criterion in RC2_CRITERIA:
        assert criterion.requirement.strip()
        assert criterion.evidence
        payload = criterion.to_dict()
        assert payload["status"] == "PASS"
        for evidence in criterion.evidence:
            if evidence == "pytest":
                continue
            path = evidence.split("::", 1)[0]
            assert (root / path).is_file(), f"{criterion.criterion_id}: missing {path}"
