from __future__ import annotations

import builtins
import ast
import keyword

from hypothesis import assume, given, settings, strategies as st
import libcst as cst

from rw_obfuscator import RandomWalkObfuscator
from rw_obfuscator.model import LengthBucket, ObfuscatorConfig
from rw_obfuscator.rules.lexical import NumericSpellingRule
from rw_obfuscator.rules.pyminifier import IntegerTemplateRule
from rw_obfuscator.rules.variable import VariableRenameRule

from tests.helpers import first_action, matching_actions


PROPERTY_SETTINGS = settings(max_examples=40, deadline=None)


@PROPERTY_SETTINGS
@given(text=st.text(max_size=100), width=st.integers(min_value=1, max_value=64))
def test_length_bucket_always_contains_its_source(text: str, width: int) -> None:
    bucket = LengthBucket.for_source(text, width)
    byte_length = len(text.encode("utf-8"))
    assert bucket.lower % width == 0
    assert bucket.upper - bucket.lower == width
    assert bucket.contains(byte_length)
    assert not bucket.contains(bucket.lower - 1)
    assert not bucket.contains(bucket.upper)


@PROPERTY_SETTINGS
@given(digits=st.text(alphabet="0123456789", min_size=1, max_size=12))
def test_fraction_leading_zero_property(digits: str) -> None:
    source = f"def f():\n    return 0.{digits}\n"
    rules = (NumericSpellingRule(),)
    target = first_action(
        source,
        "numeric_delete_leading_zero",
        rules=rules,
    ).apply(source)
    assert target == f"def f():\n    return .{digits}\n"
    assert any(
        action.apply(target) == source
        for action in matching_actions(
            target,
            "numeric_insert_leading_zero",
            rules=rules,
        )
    )


@PROPERTY_SETTINGS
@given(
    value=st.integers(min_value=0, max_value=10**12),
    key=st.sampled_from((1, 2, 3, 5, 7)),
    template=st.sampled_from(("add_sub", "xor")),
)
def test_integer_template_round_trip_property(
    value: int,
    key: int,
    template: str,
) -> None:
    source = f"def f():\n    return {value}\n"
    config = ObfuscatorConfig(integer_template_keys=(key,))
    rules = (IntegerTemplateRule(),)
    expand = f"expand_integer_{template}"
    fold = f"fold_integer_{template}"
    target = first_action(
        source,
        expand,
        rules=rules,
        config=config,
    ).apply(source)
    inverse = matching_actions(
        target,
        fold,
        rules=rules,
        config=config,
    )
    assert any(action.apply(target) == source for action in inverse)

    before: dict[str, object] = {}
    after: dict[str, object] = {}
    exec(source, before, before)
    exec(target, after, after)
    assert before["f"]() == after["f"]()


IDENTIFIERS = st.from_regex(r"[a-z][a-z0-9]{0,7}", fullmatch=True)


@PROPERTY_SETTINGS
@given(name=IDENTIFIERS, value=st.integers(min_value=-1000, max_value=1000))
def test_one_variable_rename_round_trip_property(name: str, value: int) -> None:
    assume(not keyword.iskeyword(name))
    assume(name not in {"f", "value"})
    assume(name not in dir(builtins))
    source = (
        f"def f(value):\n"
        f"    {name}=value+1\n"
        f"    return {name}*2\n"
    )
    rules = (VariableRenameRule(),)
    engine = RandomWalkObfuscator(source, rules=rules)
    pool = engine.runtime.variable_pool(name)
    assert pool is not None
    replacement = pool[1]
    candidates = [
        action
        for action in engine.enumerate_actions(source)
        if action.rule == "rename_variable"
        and dict(action.parameters) == {"old": name, "new": replacement}
    ]
    assert len(candidates) == 1
    rename = candidates[0]
    target = rename.apply(source)
    inverse = [
        action
        for action in engine.enumerate_actions(target)
        if action.rule == "rename_variable"
    ]
    assert any(action.apply(target) == source for action in inverse)

    before: dict[str, object] = {}
    after: dict[str, object] = {}
    exec(source, before, before)
    exec(target, after, after)
    assert before["f"](value) == after["f"](value)


WALK_SOURCE = """\
def score(values):
    total = 0
    for value in values:
        if value >= 0:
            total += value
        else:
            total -= value
    return total
"""


@settings(max_examples=20, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    steps=st.integers(min_value=0, max_value=12),
)
def test_every_walk_state_parses_and_stays_in_the_initial_bucket(
    seed: int,
    steps: int,
) -> None:
    engine = RandomWalkObfuscator(WALK_SOURCE, seed=seed)
    current = WALK_SOURCE
    for index in range(steps):
        current, _ = engine.step(current, step_index=index)
        ast.parse(current)
        cst.parse_module(current)
        assert engine.runtime.bucket.contains(len(current.encode("utf-8")))
