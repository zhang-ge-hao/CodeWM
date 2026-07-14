"""Rule protocol and action construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from ..context import AnalysisContext
from ..model import Action, RuntimeState, TextEdit


class Rule(Protocol):
    name: str

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]: ...


def replacement_action(
    *,
    rule: str,
    inverse_rule: str,
    site: str,
    start: int,
    end: int,
    expected: str,
    replacement: str,
    parameters: tuple[tuple[str, str], ...] = (),
) -> Action:
    return Action(
        rule=rule,
        inverse_rule=inverse_rule,
        site=site,
        edits=(TextEdit(start, end, expected, replacement),),
        parameters=parameters,
    )
