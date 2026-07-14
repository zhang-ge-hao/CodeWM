from __future__ import annotations

from collections.abc import Callable

from rw_obfuscator import RandomWalkObfuscator


def round_trip(
    source: str,
    forward_rule: str,
    inverse_rule: str,
    predicate: Callable[[object], bool] | None = None,
) -> str:
    engine = RandomWalkObfuscator(source, seed=0)
    candidates = [
        action
        for action in engine.enumerate_actions(source)
        if action.rule == forward_rule and (predicate is None or predicate(action))
    ]
    assert candidates, f"no {forward_rule} action for:\n{source}"
    target = candidates[0].apply(source)
    inverse_candidates = [
        action
        for action in engine.enumerate_actions(target)
        if action.rule == inverse_rule
    ]
    assert any(action.apply(target) == source for action in inverse_candidates)
    return target


def test_variable_rename_round_trip() -> None:
    source = (
        "def total_values(values):\n"
        "    total = 0\n"
        "    for value in values:\n"
        "        total += value\n"
        "    return total\n"
    )
    round_trip(
        source,
        "rename_variable",
        "rename_variable",
    )


def test_space_and_numeric_round_trip() -> None:
    source = "def f():\n    x = 0.5\n    return x\n"
    round_trip(source, "delete_optional_space", "insert_optional_space")
    round_trip(source, "numeric_delete_leading_zero", "numeric_insert_leading_zero")


def test_statement_boundary_round_trip() -> None:
    source = "def f():\n    x=1\n    y=2\n    return x+y\n"
    round_trip(source, "join_simple_statements", "split_simple_statements")


def test_suite_indent_parentheses_and_comma_round_trip() -> None:
    round_trip(
        "def f():\n    return 1\n",
        "inline_simple_suite",
        "expand_simple_suite",
    )
    round_trip(
        "def f(x):\n    y=x+1\n    return y\n",
        "four_spaces_to_tab",
        "tab_to_four_spaces",
    )
    round_trip(
        "def f():\n    x=(1+2)\n    return x\n",
        "delete_grouping_parentheses",
        "insert_grouping_parentheses",
    )
    round_trip(
        "def f():\n    return (sum([1, 2]))\n",
        "delete_grouping_parentheses",
        "insert_grouping_parentheses",
    )
    round_trip(
        "def f(g):\n    return g(1,)\n",
        "delete_trailing_comma",
        "insert_trailing_comma",
    )


def test_adjacent_return_parentheses_are_not_deleted() -> None:
    source = "def f():\n    return(sum([1, 2]))\n"
    engine = RandomWalkObfuscator(source, seed=0)
    targets = [
        action.apply(source)
        for action in engine.enumerate_actions(source)
        if action.rule == "delete_grouping_parentheses"
    ]
    assert not [target for target in targets if "returnsum" in target]


def test_pass_return_and_object_round_trip() -> None:
    round_trip(
        "def f():\n    x=1\n    pass\n    return x\n",
        "delete_redundant_pass",
        "insert_redundant_pass",
    )
    round_trip(
        "def f():\n    return None\n",
        "return_none_to_bare",
        "bare_return_to_none",
    )
    round_trip(
        "class C(object):\n    pass\n",
        "delete_builtin_object_base",
        "insert_builtin_object_base",
    )


def test_remaining_pyminifier_pairs_round_trip() -> None:
    round_trip(
        "import os\nimport sys\n",
        "merge_adjacent_plain_imports",
        "split_plain_import",
    )
    round_trip(
        "def f():\n    pass\n",
        "sole_pass_to_zero",
        "sole_zero_to_pass",
    )
    round_trip(
        "def f():\n    x=1\n    return\n",
        "delete_final_bare_return",
        "append_final_bare_return",
    )
    round_trip(
        "def f():\n    raise ValueError()\n",
        "delete_builtin_raise_parentheses",
        "insert_builtin_raise_parentheses",
    )
    round_trip(
        "def f():\n    x=7\n    return x\n",
        "expand_integer_add_sub",
        "fold_integer_add_sub",
    )


def test_opaque_round_trip() -> None:
    source = "def f(x):\n    y=x+1\n    return y\n"
    round_trip(source, "insert_true_opaque_guard", "remove_true_opaque_guard")
    round_trip(source, "insert_false_opaque_guard", "remove_false_opaque_guard")


def test_sequential_dispatcher_round_trip() -> None:
    source = "def f(x):\n    y=x+1\n    z=y*2\n    print(z)\n"
    round_trip(source, "flatten_straight_line", "restore_straight_line")


def test_simple_if_dispatcher_round_trip() -> None:
    source = (
        "def f(x):\n"
        "    if x>0:\n"
        "        y=x\n"
        "    else:\n"
        "        y=-x\n"
        "    print(y)\n"
    )
    round_trip(source, "flatten_simple_if", "restore_simple_if")


def test_identity_is_exactly_one_option() -> None:
    source = "def f(x):\n    y=x+1\n    return y\n"
    actions = RandomWalkObfuscator(source).enumerate_actions(source)
    assert sum(action.rule == "identity" for action in actions) == 1
