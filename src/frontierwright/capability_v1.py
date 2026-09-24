"""Frozen Frontierwright Capability v1 task bundle and presentation scale.

Capability v1 is a transparent local microbenchmark for supported Frontierwright
reference models. It is not a claim of general intelligence or a moving leaderboard.
Every task has four choices. The display mapping is frozen so that 0% accuracy maps
to 0, 25% chance maps to 50, the 50% reference anchor maps to 100, and 100% maps
to 200.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from frontierwright.domain import Axis
from frontierwright.evaluations import AxisScale, CapabilityScale, ScaleTask

CAPABILITY_V1_BUNDLE_ID = "frontierwright.capability.v1"
CAPABILITY_V1_BUNDLE_VERSION = "1"
CAPABILITY_V1_EVALUATOR_ID = "frontierwright.capability-v1-reference-evaluator"
CAPABILITY_V1_EVALUATOR_VERSION = "2"
CAPABILITY_V1_SCORING = "mean-conditional-logprob-v1"
CAPABILITY_V1_UNCERTAINTY_METHOD = "wilson-score-95-v1"
CAPABILITY_V1_CONFIDENCE_LEVEL = 0.95
_CAPABILITY_V1_WILSON_Z = 1.959963984540054
CAPABILITY_V1_FROZEN_BUNDLE_SHA256 = (
    "52ae4fc38dca0bda9e7d8d2845d985e38afeb9530317d9f716f88cabd25261c8"
)
CAPABILITY_V1_FROZEN_SCALE_SHA256 = (
    "34ca954fc3dd6fae6f9e322517518e628491a17700ac1aa4d74c3d0b16073ce5"
)


@dataclass(frozen=True)
class CapabilityTask:
    item_id: str
    axis: Axis
    prompt: str
    choices: tuple[str, str, str, str]
    correct_index: int

    def __post_init__(self) -> None:
        if not self.item_id.strip() or not self.prompt.strip():
            raise ValueError("capability task identity and prompt must be nonempty")
        if len(self.choices) != 4 or len(set(self.choices)) != 4:
            raise ValueError("capability task must contain four unique choices")
        if any(not choice.strip() for choice in self.choices):
            raise ValueError("capability choices must be nonempty")
        if self.correct_index not in range(4):
            raise ValueError("capability correct_index must be 0..3")

    def canonical_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["axis"] = self.axis.value
        payload["choices"] = list(self.choices)
        return payload


def _t(
    item_id: str,
    axis: Axis,
    prompt: str,
    choices: tuple[str, str, str, str],
    correct_index: int,
) -> CapabilityTask:
    return CapabilityTask(item_id, axis, prompt, choices, correct_index)


CAPABILITY_V1_TASKS: tuple[CapabilityTask, ...] = (
    _t(
        "general-01",
        Axis.GENERAL,
        "What is the capital of France?",
        ("Berlin", "Paris", "Rome", "Madrid"),
        1,
    ),
    _t(
        "general-02",
        Axis.GENERAL,
        "At standard pressure, water freezes at what Celsius temperature?",
        ("0", "10", "32", "100"),
        0,
    ),
    _t(
        "general-03",
        Axis.GENERAL,
        "Which word is the best opposite of 'ancient'?",
        ("old", "modern", "historic", "former"),
        1,
    ),
    _t(
        "general-04",
        Axis.GENERAL,
        "Which animal is a mammal?",
        ("trout", "whale", "lizard", "sparrow"),
        1,
    ),
    _t(
        "general-05",
        Axis.GENERAL,
        "Plants primarily use which energy source for photosynthesis?",
        ("sunlight", "sound", "magnetism", "gravity"),
        0,
    ),
    _t(
        "general-06",
        Axis.GENERAL,
        "What is the standard plural of 'child'?",
        ("childs", "childes", "children", "childrens"),
        2,
    ),
    _t(
        "general-07",
        Axis.GENERAL,
        "Which season normally follows spring in the four-season cycle?",
        ("winter", "autumn", "summer", "spring"),
        2,
    ),
    _t(
        "general-08",
        Axis.GENERAL,
        "What is Earth's natural satellite called?",
        ("Mars", "Moon", "Venus", "Sun"),
        1,
    ),
    _t(
        "general-09",
        Axis.GENERAL,
        "How many sides does a triangle have?",
        ("2", "3", "4", "5"),
        1,
    ),
    _t(
        "general-10",
        Axis.GENERAL,
        "Which gas makes up the largest share of Earth's atmosphere?",
        ("oxygen", "carbon dioxide", "nitrogen", "hydrogen"),
        2,
    ),
    _t(
        "general-11",
        Axis.GENERAL,
        "Which word is closest in meaning to 'rapid'?",
        ("slow", "fast", "quiet", "heavy"),
        1,
    ),
    _t(
        "general-12",
        Axis.GENERAL,
        "Which instrument commonly has black and white keys?",
        ("violin", "flute", "piano", "drum"),
        2,
    ),
    _t(
        "general-13",
        Axis.GENERAL,
        "At sea level, pure water boils at about what Celsius temperature?",
        ("50", "75", "100", "150"),
        2,
    ),
    _t(
        "general-14",
        Axis.GENERAL,
        "Japan is part of which continent?",
        ("Africa", "Asia", "Europe", "South America"),
        1,
    ),
    _t(
        "general-15",
        Axis.GENERAL,
        "What is the currency of Japan?",
        ("won", "yuan", "yen", "euro"),
        2,
    ),
    _t(
        "general-16",
        Axis.GENERAL,
        "Complete the sentence: She ___ to school every day.",
        ("go", "goes", "going", "gone"),
        1,
    ),
    _t(
        "reasoning-01",
        Axis.REASONING,
        "Continue the sequence: 2, 4, 8, 16, ?",
        ("18", "24", "30", "32"),
        3,
    ),
    _t(
        "reasoning-02",
        Axis.REASONING,
        "All glips are nops. All nops are blue. What must be true?",
        (
            "All glips are blue",
            "No glips are blue",
            "All blue things are glips",
            "Some nops are not blue",
        ),
        0,
    ),
    _t(
        "reasoning-03",
        Axis.REASONING,
        "A is taller than B. B is taller than C. Which must be true?",
        (
            "C is taller than A",
            "A is taller than C",
            "A and C are equal",
            "Nothing follows",
        ),
        1,
    ),
    _t(
        "reasoning-04",
        Axis.REASONING,
        "Which item is different because it is three-dimensional?",
        ("square", "circle", "triangle", "cube"),
        3,
    ),
    _t(
        "reasoning-05",
        Axis.REASONING,
        "Continue the sequence: 1, 1, 2, 3, 5, ?",
        ("6", "7", "8", "10"),
        2,
    ),
    _t(
        "reasoning-06",
        Axis.REASONING,
        "If the switch is on, the lamp is lit. The lamp is not lit. What follows?",
        (
            "The switch is on",
            "The switch is not on",
            "The lamp is broken",
            "Nothing can follow",
        ),
        1,
    ),
    _t(
        "reasoning-07",
        Axis.REASONING,
        "Continue the letter pattern: A, C, E, G, ?",
        ("H", "I", "J", "K"),
        1,
    ),
    _t(
        "reasoning-08",
        Axis.REASONING,
        "Every red object is warm. Some cube is red. What must be true?",
        (
            "Every cube is warm",
            "Some cube is warm",
            "No cube is warm",
            "Every warm object is red",
        ),
        1,
    ),
    _t(
        "reasoning-09",
        Axis.REASONING,
        "Continue the sequence: 10, 7, 4, 1, ?",
        ("-4", "-3", "-2", "0"),
        2,
    ),
    _t(
        "reasoning-10",
        Axis.REASONING,
        "If today is Monday, what day is three days later?",
        ("Tuesday", "Wednesday", "Thursday", "Friday"),
        2,
    ),
    _t(
        "reasoning-11",
        Axis.REASONING,
        "Ana is before Ben. Ben is before Cara. Who is earliest among the three?",
        ("Ana", "Ben", "Cara", "Cannot tell"),
        0,
    ),
    _t(
        "reasoning-12",
        Axis.REASONING,
        "Continue the repeating pattern: 0, 1, 1, 0, 1, 1, ?",
        ("0", "1", "2", "-1"),
        0,
    ),
    _t(
        "reasoning-13",
        Axis.REASONING,
        "If P implies Q and P is true, what follows?",
        ("Q is true", "Q is false", "P is false", "No conclusion"),
        0,
    ),
    _t(
        "reasoning-14",
        Axis.REASONING,
        "Exactly one of A or B is true. A is false. What follows?",
        ("B is false", "B is true", "Both are true", "Neither is true"),
        1,
    ),
    _t(
        "reasoning-15",
        Axis.REASONING,
        "You move one step north, then one step east. Where are you from the start?",
        ("northwest", "northeast", "southeast", "southwest"),
        1,
    ),
    _t(
        "reasoning-16",
        Axis.REASONING,
        "A book is left of a cup. The cup is left of a lamp. Which is rightmost?",
        ("book", "cup", "lamp", "They are aligned vertically"),
        2,
    ),
    _t("math-01", Axis.MATH, "What is 7 + 5?", ("10", "11", "12", "13"), 2),
    _t("math-02", Axis.MATH, "What is 9 × 6?", ("45", "54", "56", "64"), 1),
    _t("math-03", Axis.MATH, "What is 81 ÷ 9?", ("7", "8", "9", "10"), 2),
    _t(
        "math-04",
        Axis.MATH,
        "What is 15% of 200?",
        ("15", "20", "30", "40"),
        2,
    ),
    _t("math-05", Axis.MATH, "What is 3^4?", ("12", "27", "64", "81"), 3),
    _t("math-06", Axis.MATH, "Solve x + 7 = 19.", ("10", "11", "12", "13"), 2),
    _t(
        "math-07",
        Axis.MATH,
        "What is the arithmetic mean of 4, 8, and 12?",
        ("6", "8", "10", "12"),
        1,
    ),
    _t(
        "math-08",
        Axis.MATH,
        "Which fraction equals 0.25?",
        ("1/2", "1/3", "1/4", "3/4"),
        2,
    ),
    _t(
        "math-09",
        Axis.MATH,
        "A rectangle is 5 units by 3 units. What is its area?",
        ("8", "15", "16", "30"),
        1,
    ),
    _t("math-10", Axis.MATH, "Solve 2x = 18.", ("6", "8", "9", "16"), 2),
    _t(
        "math-11",
        Axis.MATH,
        "What is 5! (five factorial)?",
        ("25", "60", "120", "125"),
        2,
    ),
    _t(
        "math-12",
        Axis.MATH,
        "What is the greatest common divisor of 18 and 24?",
        ("3", "6", "9", "12"),
        1,
    ),
    _t(
        "math-13",
        Axis.MATH,
        "What is 2/3 + 1/6?",
        ("1/2", "2/3", "5/6", "1"),
        2,
    ),
    _t(
        "math-14",
        Axis.MATH,
        "What is the positive square root of 144?",
        ("10", "11", "12", "14"),
        2,
    ),
    _t(
        "math-15",
        Axis.MATH,
        "Seven items cost  each. What is the total cost?",
        ("0", "8", "1", "4"),
        2,
    ),
    _t(
        "math-16",
        Axis.MATH,
        "What is the slope between points (0,0) and (2,6)?",
        ("2", "3", "4", "6"),
        1,
    ),
    _t(
        "coding-01",
        Axis.CODING,
        "In Python, what is len([1, 2, 3])?",
        ("2", "3", "4", "error"),
        1,
    ),
    _t(
        "coding-02",
        Axis.CODING,
        "In Python, what is 2 ** 3?",
        ("5", "6", "8", "9"),
        2,
    ),
    _t(
        "coding-03",
        Axis.CODING,
        "In Python, which values are produced by list(range(3))?",
        ("[1, 2, 3]", "[0, 1, 2]", "[0, 1, 2, 3]", "[3]"),
        1,
    ),
    _t(
        "coding-04",
        Axis.CODING,
        "In Python, what is True and False?",
        ("True", "False", "None", "error"),
        1,
    ),
    _t(
        "coding-05",
        Axis.CODING,
        "In Python, what is ['a', 'b', 'c'][1]?",
        ("a", "b", "c", "1"),
        1,
    ),
    _t(
        "coding-06",
        Axis.CODING,
        "What does this Python function return for f(5)? def f(x): return x * 2",
        ("5", "7", "10", "25"),
        2,
    ),
    _t(
        "coding-07",
        Axis.CODING,
        "In Python, what is sum(range(4))?",
        ("3", "6", "10", "error"),
        1,
    ),
    _t(
        "coding-08",
        Axis.CODING,
        "In Python, what is 'abc'.upper()?",
        ("abc", "ABC", "Abc", "error"),
        1,
    ),
    _t(
        "coding-09",
        Axis.CODING,
        "In Python, what is 7 % 4?",
        ("1", "2", "3", "4"),
        2,
    ),
    _t(
        "coding-10",
        Axis.CODING,
        "In JavaScript, what is [1, 2].length?",
        ("1", "2", "3", "undefined"),
        1,
    ),
    _t(
        "coding-11",
        Axis.CODING,
        "If x = -1, which branch runs in: if x > 0: positive else: nonpositive?",
        ("positive", "nonpositive", "both", "neither"),
        1,
    ),
    _t(
        "coding-12",
        Axis.CODING,
        "In Python, what is sorted([3, 1, 2])?",
        ("[3, 2, 1]", "[1, 2, 3]", "[2, 1, 3]", "None"),
        1,
    ),
    _t(
        "coding-13",
        Axis.CODING,
        "In Python, what is 'hi' * 3?",
        ("hihihi", "hi3", "hhhiii", "error"),
        0,
    ),
    _t(
        "coding-14",
        Axis.CODING,
        "What values does Python print for: for i in range(2): print(i)",
        ("1 then 2", "0 then 1", "0 then 1 then 2", "2 only"),
        1,
    ),
    _t(
        "coding-15",
        Axis.CODING,
        "What is the usual time complexity of binary search on a sorted array?",
        ("O(1)", "O(log n)", "O(n)", "O(n^2)"),
        1,
    ),
    _t(
        "coding-16",
        Axis.CODING,
        "Which Python keyword defines a function?",
        ("class", "func", "def", "return"),
        2,
    ),
)


def capability_v1_uncertainty(correct: int, total: int) -> dict[str, object]:
    """Return a 95% Wilson interval for raw accuracy and the frozen v1 display scale.

    Capability v1 has only 16 items per axis. Returning a point estimate without an
    uncertainty interval would imply more precision than the evidence supports. The
    frozen v1 scale is linear (display = 200 * raw accuracy), so the raw Wilson bounds
    can be mapped directly without changing the frozen benchmark or scale identity.
    """

    if isinstance(correct, bool) or not isinstance(correct, int):
        raise ValueError("correct must be an integer")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("total must be a positive integer")
    if not 0 <= correct <= total:
        raise ValueError("correct must be between zero and total")

    p = correct / total
    z = _CAPABILITY_V1_WILSON_Z
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    return {
        "method": CAPABILITY_V1_UNCERTAINTY_METHOD,
        "confidence_level": CAPABILITY_V1_CONFIDENCE_LEVEL,
        "sample_size": total,
        "correct_count": correct,
        "accuracy_lower": lower,
        "accuracy_upper": upper,
        "stat_lower": 200.0 * lower,
        "stat_upper": 200.0 * upper,
    }


def exact_mcnemar_paired_binary(
    champion: tuple[bool, ...], candidate: tuple[bool, ...]
) -> dict[str, object]:
    """Exact two-sided McNemar evidence for paired binary item outcomes.

    This is descriptive/statistical evidence, not an automatic promotion rule. For the
    small Capability v1 axes, the exact binomial form avoids an asymptotic chi-square
    approximation.
    """

    if not champion or len(champion) != len(candidate):
        raise ValueError("paired binary samples must be nonempty and equal length")
    both_correct = 0
    regressions = 0
    improvements = 0
    both_wrong = 0
    for before, after in zip(champion, candidate, strict=True):
        if not isinstance(before, bool) or not isinstance(after, bool):
            raise ValueError("paired binary samples must contain booleans")
        if before and after:
            both_correct += 1
        elif before and not after:
            regressions += 1
        elif not before and after:
            improvements += 1
        else:
            both_wrong += 1

    discordant = regressions + improvements
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(regressions, improvements)
        tail_numerator = sum(math.comb(discordant, k) for k in range(lower + 1))
        p_value = min(1.0, 2.0 * tail_numerator / (2**discordant))

    if improvements > regressions:
        direction = "MORE_IMPROVEMENTS"
    elif regressions > improvements:
        direction = "MORE_REGRESSIONS"
    else:
        direction = "BALANCED"

    total = len(champion)
    return {
        "method": "mcnemar-exact-binomial-two-sided-v1",
        "sample_size": total,
        "both_correct": both_correct,
        "regressions": regressions,
        "improvements": improvements,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "net_accuracy_delta": (improvements - regressions) / total,
        "direction": direction,
        "p_value_two_sided": p_value,
        "statistically_detectable_at_0_05": p_value < 0.05,
        "note": (
            "Paired exact evidence on the same frozen items. A non-significant result is "
            "inconclusive, not proof that the models are equivalent."
        ),
    }


def capability_v1_bundle_payload() -> dict[str, object]:
    return {
        "bundle_id": CAPABILITY_V1_BUNDLE_ID,
        "bundle_version": CAPABILITY_V1_BUNDLE_VERSION,
        "scoring": CAPABILITY_V1_SCORING,
        "choice_count": 4,
        "tasks": [task.canonical_payload() for task in CAPABILITY_V1_TASKS],
    }


def capability_v1_bundle_hash() -> str:
    encoded = json.dumps(
        capability_v1_bundle_payload(),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capability_v1_task_counts() -> dict[str, int]:
    counts = {axis.value.lower(): 0 for axis in Axis}
    for task in CAPABILITY_V1_TASKS:
        counts[task.axis.value.lower()] += 1
    return counts


def capability_v1_scale() -> CapabilityScale:
    axes: list[AxisScale] = []
    for axis in Axis:
        task_id = f"{CAPABILITY_V1_BUNDLE_ID}.{axis.value.lower()}"
        axes.append(
            AxisScale(
                axis=axis,
                display_anchor=100.0,
                tasks=(
                    ScaleTask(
                        task_id=task_id,
                        task_version=CAPABILITY_V1_BUNDLE_VERSION,
                        metric="accuracy",
                        weight=1.0,
                        raw_anchor=0.5,
                        raw_unit=0.5,
                        display_per_unit=100.0,
                        higher_is_better=True,
                    ),
                ),
            )
        )
    return CapabilityScale(
        scale_id=CAPABILITY_V1_BUNDLE_ID,
        scale_version=CAPABILITY_V1_BUNDLE_VERSION,
        axes=tuple(axes),
        frozen=True,
    )


CAPABILITY_V1_SCALE = capability_v1_scale()

if capability_v1_bundle_hash() != CAPABILITY_V1_FROZEN_BUNDLE_SHA256:
    raise RuntimeError("Capability v1 task bundle changed without an explicit version/hash update")
if CAPABILITY_V1_SCALE.sha256 != CAPABILITY_V1_FROZEN_SCALE_SHA256:
    raise RuntimeError("Capability v1 scale changed without an explicit version/hash update")
