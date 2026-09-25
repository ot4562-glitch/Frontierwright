from pathlib import Path

from tools.rc4_release_gate import RC4_CRITERIA


def test_rc4_gate_manifest_is_complete_and_grounded() -> None:
    assert len(RC4_CRITERIA) == 23
    assert [item.criterion_id for item in RC4_CRITERIA] == [
        f"RC4-{index:02d}" for index in range(1, 24)
    ]
    assert len({item.criterion_id for item in RC4_CRITERIA}) == len(RC4_CRITERIA)

    root = Path(__file__).resolve().parents[1]
    for criterion in RC4_CRITERIA:
        assert criterion.requirement.strip()
        assert criterion.evidence
        payload = criterion.to_dict()
        assert payload["status"] == "PASS"
        for evidence in criterion.evidence:
            if evidence == "pytest":
                continue
            path = evidence.split("::", 1)[0]
            assert (root / path).is_file(), f"{criterion.criterion_id}: missing {path}"
