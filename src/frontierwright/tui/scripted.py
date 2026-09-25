"""Deterministic scripted TUI play for non-PTY QA environments such as CodexPro."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from textual.css.query import NoMatches
from textual.widgets import Input, Static, TabbedContent

from frontierwright.tui.app import FrontierwrightApp


def load_play_script(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read scripted-play JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Scripted-play JSON must be an object.")
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("Scripted-play JSON must contain a steps list.")

    steps: list[dict[str, object]] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise ValueError(f"Step {index} must be an object.")
        kinds = [key for key in ("press", "set_input", "pause") if key in raw]
        if len(kinds) != 1:
            raise ValueError(
                f"Step {index} must contain exactly one of press, set_input, or pause."
            )
        kind = kinds[0]
        if kind == "press":
            keys = raw["press"]
            if (
                not isinstance(keys, list)
                or not keys
                or not all(isinstance(key, str) and key for key in keys)
            ):
                raise ValueError(f"Step {index} press must be a non-empty string list.")
        elif kind == "set_input":
            value = raw["set_input"]
            if not isinstance(value, dict):
                raise ValueError(f"Step {index} set_input must be an object.")
            widget_id = value.get("id")
            text_value = value.get("value")
            if not isinstance(widget_id, str) or not widget_id:
                raise ValueError(f"Step {index} set_input.id must be a non-empty string.")
            if not isinstance(text_value, str):
                raise ValueError(f"Step {index} set_input.value must be a string.")
        else:
            seconds = raw["pause"]
            if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds < 0:
                raise ValueError(f"Step {index} pause must be a non-negative number.")
        steps.append(dict(raw))
    return steps


def _snapshot(app: FrontierwrightApp, *, step_index: int, action: str) -> dict[str, object]:
    active_tab: str | None = None
    try:
        active_tab = app.query_one("#main-tabs", TabbedContent).active
    except NoMatches:
        pass

    focused = app.focused
    focused_widget = focused.id if focused is not None else None

    static_text: list[str] = []
    for widget in app.screen.query(Static):
        content = str(widget.content).strip()
        if content:
            static_text.append(content)

    inputs: dict[str, str] = {}
    for input_widget in app.screen.query(Input):
        if input_widget.id:
            inputs[input_widget.id] = input_widget.value

    return {
        "step_index": step_index,
        "action": action,
        "screen": type(app.screen).__name__,
        "active_tab": active_tab,
        "focused_widget": focused_widget,
        "inputs": inputs,
        "text": "\n\n".join(static_text),
    }


async def run_scripted_play(
    app: FrontierwrightApp,
    steps: list[dict[str, object]],
    *,
    width: int = 120,
    height: int = 44,
) -> dict[str, object]:
    if width < 80 or height < 24:
        raise ValueError("Scripted-play terminal size must be at least 80x24.")

    snapshots: list[dict[str, object]] = []
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()
        snapshots.append(_snapshot(app, step_index=0, action="initial"))

        for index, step in enumerate(steps, start=1):
            if "press" in step:
                keys = step["press"]
                assert isinstance(keys, list)
                await pilot.press(*[str(key) for key in keys])
                action = "press:" + ",".join(str(key) for key in keys)
            elif "set_input" in step:
                payload = step["set_input"]
                assert isinstance(payload, dict)
                widget_id = str(payload["id"])
                value = str(payload["value"])
                try:
                    widget = app.screen.query_one(f"#{widget_id}", Input)
                except NoMatches as exc:
                    raise ValueError(
                        f"Step {index} cannot find visible Input #{widget_id} on "
                        f"{type(app.screen).__name__}."
                    ) from exc
                widget.value = value
                widget.focus()
                action = f"set_input:{widget_id}"
            else:
                pause_value = step["pause"]
                assert isinstance(pause_value, (int, float)) and not isinstance(pause_value, bool)
                seconds = float(pause_value)
                await pilot.pause(delay=seconds)
                action = f"pause:{seconds:g}"

            await pilot.pause()
            snapshots.append(_snapshot(app, step_index=index, action=action))

    return {
        "schema_version": 1,
        "ok": True,
        "mode": "SCRIPTED_TUI_QA",
        "terminal": {"width": width, "height": height},
        "snapshots": snapshots,
    }


def scripted_play_payload(
    app: FrontierwrightApp,
    script_path: Path,
    *,
    width: int = 120,
    height: int = 44,
) -> dict[str, Any]:
    """Synchronous wrapper used by the CLI command."""

    import asyncio

    steps = load_play_script(script_path)
    return asyncio.run(run_scripted_play(app, steps, width=width, height=height))
