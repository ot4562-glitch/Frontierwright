from __future__ import annotations

Language = str

_MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "app.title": "FRONTIERWRIGHT",
        "character": "CHARACTER",
        "build": "BUILD",
        "paths": "PATHS",
        "resources": "RESOURCES",
        "data": "DATA",
        "history": "HISTORY",
        "candidates": "CANDIDATES",
        "not_ready": "NOT READY",
        "unknown": "UNKNOWN",
        "no_project": "No Frontierwright project is initialized here.",
        "help": "Arrows/hjkl move · Enter select · Esc back · 1-7 tabs · ? help · q quit",
        "recommendation_locked": (
            "History-aware recommendation unavailable until history is COMPLETE or VERIFIED."
        ),
    },
    "ko": {
        "app.title": "FRONTIERWRIGHT",
        "character": "캐릭터",
        "build": "빌드",
        "paths": "육성 경로",
        "resources": "자원",
        "data": "데이터",
        "history": "히스토리",
        "candidates": "후보",
        "not_ready": "아직 측정 불가",
        "unknown": "알 수 없음",
        "no_project": "이 위치에는 Frontierwright 프로젝트가 없습니다.",
        "help": "방향키/hjkl 이동 · Enter 선택 · Esc 뒤로 · 1-7 탭 · ? 도움말 · q 종료",
        "recommendation_locked": (
            "이력이 COMPLETE 또는 VERIFIED가 될 때까지 이력 기반 추천을 제공하지 않습니다."
        ),
    },
}


def tr(language: Language, key: str) -> str:
    catalog = _MESSAGES.get(language, _MESSAGES["en"])
    return catalog.get(key, _MESSAGES["en"].get(key, key))
