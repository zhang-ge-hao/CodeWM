from __future__ import annotations

import pytest

from rw_obfuscator import RandomWalkObfuscator
from rw_obfuscator.rules.advanced import DispatcherRule, OpaquePredicateRule

from tests.helpers import first_action, matching_actions


@pytest.mark.parametrize(
    "source",
    [
        # The second factor must be the first factor plus one.
        "def f():\n    if (17*19)%2==0:\n        x=1\n    else:\n        0\n",
        # The constant must come from the finite configured catalog.
        "def f():\n    if (19*20)%2==0:\n        x=1\n    else:\n        0\n",
        # The dead branch must be the canonical single expression `0`.
        "def f():\n    if (17*18)%2==0:\n        x=1\n    else:\n        1\n",
        # A ==1 guard must put the live payload in the else branch.
        "def f():\n    if (17*18)%2==1:\n        x=1\n    else:\n        0\n",
        # No additional statement is allowed in either branch.
        "def f():\n    if (17*18)%2==0:\n        x=1\n        y=2\n    else:\n        0\n",
        # Canonical spelling is part of the recognizer grammar.
        "def f():\n    if (17 * 18) % 2 == 0:\n        x=1\n    else:\n        0\n",
    ],
)
def test_malformed_opaque_guards_are_never_unwrapped(source: str) -> None:
    actions = RandomWalkObfuscator(
        source, rules=(OpaquePredicateRule(),)
    ).enumerate_actions(source)
    assert not [action for action in actions if action.rule.startswith("remove_")]


def test_only_the_exact_canonical_opaque_guard_is_unwrapped() -> None:
    source = "def f():\n    x=1\n    return x\n"
    rules = (OpaquePredicateRule(),)
    wrapped = first_action(
        source,
        "insert_true_opaque_guard",
        rules=rules,
        predicate=lambda action: dict(action.parameters)["constant"] == "17",
    ).apply(source)
    inverse = first_action(wrapped, "remove_true_opaque_guard", rules=rules)
    assert inverse.apply(wrapped) == source


def test_malformed_dispatchers_are_never_restored() -> None:
    source = "def f(x):\n    y=x+1\n    z=y*2\n    return z\n"
    rules = (DispatcherRule(),)
    engine = RandomWalkObfuscator(source, rules=rules)
    flatten = next(
        action
        for action in engine.enumerate_actions(source)
        if action.rule == "flatten_straight_line"
    )
    canonical = flatten.apply(source)
    parameters = dict(flatten.parameters)
    pc = parameters["pc"]
    labels = tuple(map(int, parameters["labels"].split(",")))
    order = tuple(map(int, parameters["order"].split(",")))

    restores = tuple(
        action
        for action in engine.enumerate_actions(canonical)
        if action.rule == "restore_straight_line"
    )
    assert any(action.apply(canonical) == source for action in restores)

    malformed = (
        canonical.replace(f"{pc} = {labels[1]}\n", f"{pc} = 999\n", 1),
        canonical.replace(f"del {pc}\n", f"{pc} = 0\n", 1),
        canonical.replace(pc, "__rw_pc_99"),
        canonical.replace(
            f"{pc} == {labels[order[0]]}",
            f"{pc} == 999",
            1,
        ),
    )
    assert len(set(malformed)) == len(malformed)
    for candidate in malformed:
        assert not [
            action
            for action in engine.enumerate_actions(candidate)
            if action.rule == "restore_straight_line"
        ]


def test_malformed_conditional_dispatchers_are_never_restored() -> None:
    source = (
        "def f(x):\n"
        "    if x>0:\n"
        "        y=x\n"
        "    else:\n"
        "        y=-x\n"
        "    return y\n"
    )
    rules = (DispatcherRule(),)
    engine = RandomWalkObfuscator(source, rules=rules)
    flatten = next(
        action
        for action in engine.enumerate_actions(source)
        if action.rule == "flatten_simple_if"
    )
    canonical = flatten.apply(source)
    parameters = dict(flatten.parameters)
    pc = parameters["pc"]
    labels = tuple(map(int, parameters["labels"].split(",")))

    restores = tuple(
        action
        for action in engine.enumerate_actions(canonical)
        if action.rule == "restore_simple_if"
    )
    assert any(action.apply(canonical) == source for action in restores)

    malformed = (
        canonical.replace(
            f"{pc} = {labels[3]}\n",
            f"{pc} = 999\n",
            1,
        ),
        canonical.replace(f"del {pc}\n", f"{pc} = 0\n", 1),
        canonical.replace(pc, "__rw_pc_99"),
    )
    assert len(set(malformed)) == len(malformed)
    for candidate in malformed:
        assert not [
            action
            for action in engine.enumerate_actions(candidate)
            if action.rule == "restore_simple_if"
        ]
