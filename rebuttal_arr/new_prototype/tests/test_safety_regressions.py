from __future__ import annotations

import ast

from rw_obfuscator import RandomWalkObfuscator
from rw_obfuscator.rules.advanced import DispatcherRule, OpaquePredicateRule
from rw_obfuscator.rules.lexical import TrailingCommaRule
from rw_obfuscator.rules.pyminifier import (
    BuiltinRaiseParenthesesRule,
    ObjectBaseRule,
)
from rw_obfuscator.rules.structural import (
    IndentationUnitRule,
    SimpleStatementBoundaryRule,
    SimpleSuiteRule,
)
from rw_obfuscator.rules.variable import VariableRenameRule

from tests.helpers import first_action, matching_actions


def test_three_statement_semicolon_line_has_no_one_step_split() -> None:
    source = "def f():\n    x=1;y=2;z=3\n    return x+y+z\n"
    assert not matching_actions(
        source,
        "split_simple_statements",
        rules=(SimpleStatementBoundaryRule(),),
    )


def test_every_tuple_trailing_comma_action_has_an_exact_inverse() -> None:
    source = "def f():\n    return (1,2,)\n"
    rules = (TrailingCommaRule(),)
    actions = matching_actions(source, "delete_trailing_comma", rules=rules)
    assert actions
    for action in actions:
        target = action.apply(source)
        inverse = matching_actions(target, "insert_trailing_comma", rules=rules)
        assert any(candidate.apply(target) == source for candidate in inverse)


def test_trailing_comma_skips_unparenthesized_generator_argument() -> None:
    source = "def f(xs):\n    return sum(x for x in xs)\n"
    rules = (TrailingCommaRule(),)
    actions = matching_actions(source, "insert_trailing_comma", rules=rules)
    assert not actions

    parenthesized = "def f(xs):\n    return sum((x for x in xs))\n"
    actions = matching_actions(
        parenthesized,
        "insert_trailing_comma",
        rules=rules,
    )
    assert actions
    for action in actions:
        target = action.apply(parenthesized)
        ast.parse(target)
        inverse = matching_actions(target, "delete_trailing_comma", rules=rules)
        assert any(candidate.apply(target) == parenthesized for candidate in inverse)


def test_rename_does_not_capture_a_nested_free_variable() -> None:
    source = (
        "z=10\n"
        "def f():\n"
        "    x=1\n"
        "    def g():\n"
        "        return z\n"
        "    return x+g()\n"
    )
    actions = matching_actions(
        source,
        "rename_variable",
        rules=(VariableRenameRule(),),
    )
    assert not [
        action
        for action in actions
        if dict(action.parameters) == {"old": "x", "new": "z"}
    ]


def test_imported_names_disable_builtin_dependent_toggles() -> None:
    object_source = "from somewhere import object\nclass C:\n    pass\n"
    assert not matching_actions(
        object_source,
        "insert_builtin_object_base",
        rules=(ObjectBaseRule(),),
    )

    exception_source = (
        "from somewhere import ValueError\n"
        "def f():\n"
        "    raise ValueError\n"
    )
    assert not matching_actions(
        exception_source,
        "insert_builtin_raise_parentheses",
        rules=(BuiltinRaiseParenthesesRule(),),
    )


def test_noncanonical_dispatcher_spelling_is_not_restored() -> None:
    source = "def f(x):\n    y=x+1\n    z=y*2\n    return z\n"
    rules = (DispatcherRule(),)
    flatten = first_action(source, "flatten_straight_line", rules=rules)
    canonical = flatten.apply(source)
    pc = dict(flatten.parameters)["pc"]
    noncanonical = canonical.replace(f"{pc} = ", f"{pc}  = ", 1)
    assert noncanonical != canonical
    assert not matching_actions(
        noncanonical,
        "restore_straight_line",
        rules=rules,
    )


def test_multiline_simple_statement_is_not_used_as_advanced_payload() -> None:
    source = (
        "def f():\n"
        "    x=(\n"
        "        1+2\n"
        "    )\n"
        "    return x\n"
    )
    opaque = RandomWalkObfuscator(
        source, rules=(OpaquePredicateRule(),)
    ).enumerate_actions(source)
    assert not [action for action in opaque if action.rule.startswith("insert_")]

    dispatch = RandomWalkObfuscator(
        source, rules=(DispatcherRule(),)
    ).enumerate_actions(source)
    assert not [action for action in dispatch if action.rule.startswith("flatten_")]


def test_decorated_compound_is_not_rewritten_by_suite_toggle() -> None:
    source = (
        "def decorate(function):\n"
        "    return function\n"
        "@decorate\n"
        "def f():\n"
        "    return 1\n"
    )
    actions = matching_actions(
        source,
        "inline_simple_suite",
        rules=(SimpleSuiteRule(),),
    )
    assert not [action for action in actions if action.site == "suite@4"]


def test_elif_and_terminal_semicolon_are_not_inlined() -> None:
    elif_source = (
        "def f(x):\n"
        "    if x > 1:\n"
        "        return 1\n"
        "    elif x > 0:\n"
        "        return 0\n"
        "    else:\n"
        "        return -1\n"
    )
    actions = matching_actions(
        elif_source,
        "inline_simple_suite",
        rules=(SimpleSuiteRule(),),
    )
    assert not [action for action in actions if action.site == "suite@4"]

    semicolon_source = (
        "def f(x):\n"
        "    if x:\n"
        "        x += 1; \n"
        "    return x\n"
    )
    actions = matching_actions(
        semicolon_source,
        "inline_simple_suite",
        rules=(SimpleSuiteRule(),),
    )
    assert not [action for action in actions if action.site == "suite@2"]


def test_indentation_toggle_uses_the_owners_actual_indent_unit() -> None:
    two_space_source = (
        "def f(x):\n"
        "  if x:\n"
        "    return 1\n"
        "  return 0\n"
    )
    actions = RandomWalkObfuscator(
        two_space_source,
        rules=(IndentationUnitRule(),),
    ).enumerate_actions(two_space_source)
    assert not [action for action in actions if action.rule != "identity"]

    four_space_source = (
        "def f(x):\n"
        "    if x:\n"
        "        return 1\n"
        "    return 0\n"
    )
    engine = RandomWalkObfuscator(
        four_space_source,
        rules=(IndentationUnitRule(),),
    )
    actions = [
        action
        for action in engine.enumerate_actions(four_space_source)
        if action.rule == "four_spaces_to_tab"
    ]
    assert actions
    for action in actions:
        target = action.apply(four_space_source)
        ast.parse(target)
        inverse = [
            candidate
            for candidate in engine.enumerate_actions(target)
            if candidate.rule == "tab_to_four_spaces"
        ]
        assert any(candidate.apply(target) == four_space_source for candidate in inverse)
