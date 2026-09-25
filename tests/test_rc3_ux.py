from dataclasses import replace

from frontierwright.service import DataView, ResourceView, StatusView
from frontierwright.tui import FrontierwrightApp
from frontierwright.tui.app import tab_order_for_edition


def _academy_view() -> StatusView:
    return StatusView(
        initialized=True,
        project_id="fw-rc3",
        project_name="NOVA",
        language="en",
        nickname="NOVA",
        edition_profile="ACADEMY",
        edition_name="Frontierwright Academy",
        edition_tagline="Understand AI by building a real model from birth.",
        edition_starting_point="BIRTH",
        origin="ZERO",
        history_confidence="COMPLETE",
        measurement_state="NOT_READY",
        build_mode="INTENT",
        strong_recommendation_allowed=True,
        stats={"general": None, "reasoning": None, "math": None, "coding": None},
    )


def test_rc3_tab_order_is_edition_specific() -> None:
    assert tab_order_for_edition("ACADEMY") == (
        "character",
        "data",
        "resources",
        "build",
        "paths",
        "workload",
        "candidates",
        "history",
    )
    assert tab_order_for_edition("STUDIO") == (
        "character",
        "workload",
        "resources",
        "build",
        "paths",
        "data",
        "candidates",
        "history",
    )
    assert tab_order_for_edition("LAB") == (
        "character",
        "workload",
        "data",
        "resources",
        "paths",
        "build",
        "candidates",
        "history",
    )


def test_rc3_academy_resource_text_turns_probe_gap_into_action() -> None:
    resources = ResourceView(
        available=True,
        provenance="DETECTED",
        detected_at="2026-09-25T06:37:21+00:00",
        snapshot={
            "cpu_model": "Example CPU",
            "cpu_logical_count": 12,
            "ram_total_bytes": 32 * 1024**3,
            "ram_available_bytes": 18 * 1024**3,
            "disk_total_bytes": 500 * 1024**3,
            "disk_free_bytes": 50 * 1024**3,
            "gpus": [],
            "diagnostics": [
                {
                    "probe_id": "gpu.nvidia",
                    "status": "MEASUREMENT_NEEDED",
                    "reason_code": "NVIDIA_SMI_NOT_FOUND",
                    "detail": "NVIDIA probe executable was not found.",
                    "next_action": "Install or repair the NVIDIA driver, then refresh resources.",
                    "physical_absence_proven": False,
                }
            ],
        },
        headroom={
            "ram_total_bytes": 32 * 1024**3,
            "ram_available_bytes": 18 * 1024**3,
            "disk_total_bytes": 500 * 1024**3,
            "disk_available_bytes": 50 * 1024**3,
            "gpus": [],
        },
    )
    app = FrontierwrightApp(view=_academy_view(), resources=resources)
    rendered = app._resources_text()

    assert "Measured: 2026-09-25T06:37:21+00:00" in rendered
    assert "GPU availability: MEASUREMENT NEEDED" in rendered
    assert "does not prove that no GPU is installed" in rendered
    assert "Install or repair the NVIDIA driver" in rendered


def test_rc3_fresh_academy_state_uses_plain_language() -> None:
    app = FrontierwrightApp(view=_academy_view())
    text = app._character_text()

    assert "Project history: fully recorded" in text
    assert "Model: not created yet" in text
    assert "Capability: not measured yet" in text
    assert "History evidence: COMPLETE" not in text


def test_rc3_studio_and_lab_character_focus_are_distinct() -> None:
    studio = FrontierwrightApp(
        view=replace(
            _academy_view(),
            edition_profile="STUDIO",
            edition_name="Frontierwright Studio",
            edition_starting_point="IMPORT_OR_CONTINUE",
            origin="IMPORTED_LOCAL",
        )
    )
    lab = FrontierwrightApp(
        view=replace(
            _academy_view(),
            edition_profile="LAB",
            edition_name="Frontierwright Lab",
            edition_starting_point="CONNECT_INFRASTRUCTURE",
            origin="INTERNAL_LAB",
        )
    )

    assert "YOUR MODEL / YOUR MACHINE / YOUR WORKLOAD" in studio._character_text()
    assert "CONTROLLED MODEL / EVIDENCE / INFRASTRUCTURE" in lab._character_text()


def test_rc3_academy_data_explains_private_and_unmeasured_tokens() -> None:
    data = DataView(
        datasets=[
            {
                "name": "academy-corpus",
                "role": "PRETRAIN",
                "total_bytes": 733,
                "provenance": "LOCAL_USER",
                "classification": "PRIVATE",
                "fingerprint": "sha256:fixture",
                "license": "synthetic-playtest",
                "token_count": None,
                "managed": False,
            }
        ]
    )
    app = FrontierwrightApp(view=_academy_view(), data=data)
    rendered = app._data_text()

    assert "Tokens: not measured yet" in rendered
    assert "tokenizer" in rendered.lower()
    assert "PRIVATE" in rendered
    assert "compatible data boundary" in rendered


def test_rc3_edition_help_has_a_first_run_sequence() -> None:
    from frontierwright.tui.app import _edition_help_text

    academy = _edition_help_text("en", "ACADEMY")
    studio = _edition_help_text("en", "STUDIO")
    lab = _edition_help_text("en", "LAB")

    assert "START HERE" in academy
    assert "1. Add learning data" in academy
    assert "START HERE" in studio
    assert "1. Import or continue a model you control" in studio
    assert "START HERE" in lab
    assert "1. Connect controlled private infrastructure" in lab
