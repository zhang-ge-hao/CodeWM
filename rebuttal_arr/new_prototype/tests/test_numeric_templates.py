from __future__ import annotations

import pytest

from rw_obfuscator import RandomWalkObfuscator
from rw_obfuscator.model import ObfuscatorConfig
from rw_obfuscator.rules.lexical import NumericSpellingRule
from rw_obfuscator.rules.pyminifier import IntegerTemplateRule

from tests.helpers import first_action


@pytest.mark.parametrize(
    ("literal", "rule", "replacement", "inverse_rule"),
    [
        ("0.5", "numeric_delete_leading_zero", ".5", "numeric_insert_leading_zero"),
        (".5", "numeric_insert_leading_zero", "0.5", "numeric_delete_leading_zero"),
        ("1.0", "numeric_delete_trailing_zero", "1.", "numeric_insert_trailing_zero"),
        ("1.", "numeric_insert_trailing_zero", "1.0", "numeric_delete_trailing_zero"),
        ("1e+10", "numeric_delete_exponent_plus", "1e10", "numeric_insert_exponent_plus"),
        ("1e10", "numeric_insert_exponent_plus", "1e+10", "numeric_delete_exponent_plus"),
        ("256", "numeric_decimal_to_hex", "0x100", "numeric_hex_to_decimal"),
        ("0x100", "numeric_hex_to_decimal", "256", "numeric_decimal_to_hex"),
    ],
)
def test_every_numeric_spelling_pair_is_exactly_reversible(
    literal: str,
    rule: str,
    replacement: str,
    inverse_rule: str,
) -> None:
    source = f"def f():\n    return {literal}\n"
    expected = f"def f():\n    return {replacement}\n"
    rules = (NumericSpellingRule(),)

    action = first_action(source, rule, rules=rules)
    target = action.apply(source)
    assert target == expected

    inverse = first_action(target, inverse_rule, rules=rules)
    assert inverse.apply(target) == source

    before: dict[str, object] = {}
    after: dict[str, object] = {}
    exec(source, before, before)
    exec(target, after, after)
    assert before["f"]() == after["f"]()


def test_hexadecimal_with_e_digit_is_not_treated_as_scientific_notation() -> None:
    source = "def f():\n    return 0x3e8\n"
    engine = RandomWalkObfuscator(
        source,
        seed=0,
        rules=(NumericSpellingRule(),),
    )
    actions = [
        action for action in engine.enumerate_actions(source) if not action.is_identity
    ]
    assert {action.rule for action in actions} == {"numeric_hex_to_decimal"}
    assert actions[0].apply(source) == "def f():\n    return 1000\n"


def test_decimal_to_hex_cannot_then_insert_an_exponent_plus() -> None:
    source = "def f():\n    return 1000\n"
    engine = RandomWalkObfuscator(
        source,
        seed=0,
        rules=(NumericSpellingRule(),),
    )
    to_hex = next(
        action
        for action in engine.enumerate_actions(source)
        if action.rule == "numeric_decimal_to_hex"
    )
    hexadecimal = to_hex.apply(source)
    assert hexadecimal == "def f():\n    return 0x3e8\n"
    next_rules = {action.rule for action in engine.enumerate_actions(hexadecimal)}
    assert "numeric_insert_exponent_plus" not in next_rules
    assert "numeric_hex_to_decimal" in next_rules


@pytest.mark.parametrize(("key", "kind"), [
    (key, kind)
    for key in (1, 2, 3, 5, 7)
    for kind in ("add_sub", "xor")
])
def test_integer_templates_include_add_sub_and_xor(
    key: int,
    kind: str,
) -> None:
    source = "def f():\n    return 37\n"
    config = ObfuscatorConfig(integer_template_keys=(key,))
    rules = (IntegerTemplateRule(),)
    engine = RandomWalkObfuscator(source, config=config, rules=rules)
    expand = f"expand_integer_{kind}"
    fold = f"fold_integer_{kind}"
    candidates = [
        action for action in engine.enumerate_actions(source) if action.rule == expand
    ]
    assert len(candidates) == 1
    target = candidates[0].apply(source)
    template = (
        f"((37+{key})-{key})"
        if kind == "add_sub"
        else f"((37^{key})^{key})"
    )
    assert target == f"def f():\n    return {template}\n"

    inverse_candidates = [
        action for action in engine.enumerate_actions(target) if action.rule == fold
    ]
    assert len(inverse_candidates) == 1
    assert inverse_candidates[0].apply(target) == source


@pytest.mark.parametrize("source", [
    "def f():\n    return ((7+2)-3)\n",
    "def f():\n    return ((7^2)^3)\n",
])
def test_mismatched_integer_template_keys_are_not_folded(source: str) -> None:
    actions = RandomWalkObfuscator(
        source, rules=(IntegerTemplateRule(),)
    ).enumerate_actions(source)
    assert not [action for action in actions if action.rule.startswith("fold_integer_")]
