from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from rw_obfuscator import RandomWalkObfuscator
from rw_obfuscator.model import Action, ObfuscatorConfig


def matching_actions(
    source: str,
    rule: str,
    *,
    predicate: Callable[[Action], bool] | None = None,
    rules: Sequence[object] | None = None,
    config: ObfuscatorConfig | None = None,
) -> tuple[Action, ...]:
    engine = RandomWalkObfuscator(source, rules=rules, config=config)
    return tuple(
        action
        for action in engine.enumerate_actions(source)
        if action.rule == rule and (predicate is None or predicate(action))
    )


def first_action(
    source: str,
    rule: str,
    *,
    predicate: Callable[[Action], bool] | None = None,
    rules: Sequence[object] | None = None,
    config: ObfuscatorConfig | None = None,
) -> Action:
    actions = matching_actions(
        source,
        rule,
        predicate=predicate,
        rules=rules,
        config=config,
    )
    assert actions, f"no {rule} action for:\n{source}"
    return actions[0]


def observe(source: str, expression: str) -> tuple[Any, ...]:
    namespace: dict[str, object] = {}
    try:
        exec(source, namespace, namespace)
        return ("value", eval(expression, namespace, namespace))
    except BaseException as error:
        return ("exception", type(error).__name__, str(error))
