from __future__ import annotations

from collections import Counter

import pytest

from rw_obfuscator import RandomWalkObfuscator
from rw_obfuscator.model import Action, IDENTITY, ObfuscatorConfig, TextEdit


class BoundaryRule:
    name = "boundary_test"

    def enumerate_actions(self, context, runtime):
        yield Action(
            rule="grow",
            inverse_rule="shrink",
            site="eof",
            edits=(TextEdit(len(context.source), len(context.source), "", "#"),),
        )
        yield Action(
            rule="shrink",
            inverse_rule="grow",
            site="space",
            edits=(TextEdit(1, 2, " ", ""),),
        )


def test_actions_crossing_either_bucket_boundary_are_not_enumerated() -> None:
    source = "x = 1\n"  # six UTF-8 bytes

    upper_engine = RandomWalkObfuscator(
        source,
        config=ObfuscatorConfig(bucket_width=7),
        rules=(BoundaryRule(),),
    )
    upper_rules = {action.rule for action in upper_engine.enumerate_actions(source)}
    assert upper_rules == {"identity", "shrink"}

    lower_engine = RandomWalkObfuscator(
        source,
        config=ObfuscatorConfig(bucket_width=6),
        rules=(BoundaryRule(),),
    )
    lower_rules = {action.rule for action in lower_engine.enumerate_actions(source)}
    assert lower_rules == {"identity", "grow"}


def test_flat_action_sampler_is_uniform_and_identity_is_not_special() -> None:
    engine = RandomWalkObfuscator("x = 1\n", seed=918273, rules=())
    actions = (IDENTITY,) + tuple(
        Action(rule=f"action_{index}", site=str(index)) for index in range(4)
    )
    draws = 50_000
    counts = Counter(engine.sample_action(actions).rule for _ in range(draws))
    expected = draws / len(actions)
    chi_square = sum(
        (counts[action.rule] - expected) ** 2 / expected for action in actions
    )

    assert set(counts) == {action.rule for action in actions}
    # df=4; 18.47 is the p=0.001 upper critical value.
    assert chi_square < 18.47


def test_flat_action_sampler_rejects_only_an_empty_input() -> None:
    engine = RandomWalkObfuscator("x = 1\n", rules=())
    with pytest.raises(ValueError, match="empty action set"):
        engine.sample_action(())
