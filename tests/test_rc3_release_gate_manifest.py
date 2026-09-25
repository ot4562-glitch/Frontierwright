from pathlib import Path

from tools.rc3_release_gate import RC3_CRITERIA


def test_rc3_gate_manifest_is_complete_and_grounded() -> None:
    assert len(RC3_CRITERIA) == 16
    assert [item.criterion_id for item in RC3_CRITERIA] == [
        f"RC3-{index:02d}" for index in range(1, 17)
    ]
    assert len({item.criterion_id for item in RC3_CRITERIA}) == len(RC3_CRITERIA)

    root = Path(__file__).resolve().parents[1]
    for criterion in RC3_CRITERIA:
        assert criterion.requirement.strip()
        assert criterion.evidence
        payload = criterion.to_dict()
        assert payload["status"] == "PASS"
        for evidence in criterion.evidence:
            if evidence == "pytest":
                continue
            path = evidence.split("::", 1)[0]
            assert (root / path).is_file(), f"{criterion.criterion_id}: missing {path}"
