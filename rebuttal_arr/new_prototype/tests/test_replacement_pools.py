from __future__ import annotations

import re

from rw_obfuscator import RandomWalkObfuscator
from rw_obfuscator.rules.advanced import DispatcherRule


ALIAS_RE = re.compile(r"[A-Za-z]{16}")


def _actions(engine, source: str, rule: str):
    return tuple(
        action
        for action in engine.enumerate_actions(source)
        if action.rule == rule
    )


def test_each_original_variable_has_one_disjoint_ten_member_pool() -> None:
    source = (
        "def f():\n"
        "    var1=1\n"
        "    var2=2\n"
        "    return var1+var2\n"
    )
    engine = RandomWalkObfuscator(source, seed=7)
    pool1 = engine.runtime.variable_pool("var1")
    pool2 = engine.runtime.variable_pool("var2")
    assert pool1 is not None and pool2 is not None
    assert len(pool1) == len(pool2) == 10
    assert pool1[0] == "var1" and pool2[0] == "var2"
    assert set(pool1).isdisjoint(pool2)
    assert all(ALIAS_RE.fullmatch(name) for name in pool1[1:] + pool2[1:])

    actions = _actions(engine, source, "rename_variable")
    by_old = {}
    for action in actions:
        parameters = dict(action.parameters)
        by_old.setdefault(parameters["old"], set()).add(parameters["new"])
    assert by_old["var1"] == set(pool1[1:])
    assert by_old["var2"] == set(pool2[1:])

    selected = next(
        action
        for action in actions
        if dict(action.parameters) == {"old": "var1", "new": pool1[1]}
    )
    target = selected.apply(source)
    next_actions = _actions(engine, target, "rename_variable")
    next_by_old = {}
    for action in next_actions:
        parameters = dict(action.parameters)
        next_by_old.setdefault(parameters["old"], set()).add(parameters["new"])
    assert next_by_old[pool1[1]] == {"var1"}
    assert next_by_old["var2"] == set(pool2[1:])
    assert "var1" not in next_by_old["var2"]


def test_single_comment_has_a_directional_ten_state_pool() -> None:
    source = "def f():\n    return 1  # beta\n"
    engine = RandomWalkObfuscator(source, seed=11)
    actions = _actions(engine, source, "replace_comment_line")
    assert len(actions) == 9
    assert all(
        re.fullmatch(r"# [A-Za-z]{16}", dict(action.parameters)["new"])
        for action in actions
    )
    target = actions[0].apply(source)
    inverse = _actions(engine, target, "replace_comment_line")
    assert len(inverse) == 1
    assert inverse[0].apply(target) == source


def test_multiline_comments_and_standalone_strings_are_whole_100_alias_units() -> None:
    source = (
        "def f():\n"
        "    # alpha\n"
        "    # beta\n"
        '    \"\"\"first\n'
        '    second\"\"\"\n'
        "    return 1\n"
    )
    engine = RandomWalkObfuscator(source, seed=11)
    actions = _actions(engine, source, "replace_comment_block")
    comment_actions = [action for action in actions if action.site.startswith("comment-block")]
    string_actions = [action for action in actions if action.site.startswith("standalone-string")]
    assert len(comment_actions) == len(string_actions) == 100

    comment_target = comment_actions[0].apply(source)
    assert re.search(r"    # [A-Za-z]{16}\n    #\n", comment_target)
    comment_inverse = [
        action
        for action in _actions(engine, comment_target, "replace_comment_block")
        if action.site.startswith("comment-block")
    ]
    assert len(comment_inverse) == 1
    assert comment_inverse[0].apply(comment_target) == source

    string_target = string_actions[0].apply(source)
    assert re.search(r'    \"\"\"[A-Za-z]{16}\n\"\"\"\n', string_target)
    string_inverse = [
        action
        for action in _actions(engine, string_target, "replace_comment_block")
        if action.site.startswith("standalone-string")
    ]
    assert len(string_inverse) == 1
    assert string_inverse[0].apply(string_target) == source


def test_replaced_comment_after_semicolon_remains_parseable_and_reversible() -> None:
    source = "def f():\n    C=[0]; # nC0 is 1\n    return C\n"
    engine = RandomWalkObfuscator(source, seed=12)
    action = next(
        action
        for action in _actions(engine, source, "replace_comment_line")
        if action.site.startswith("comment@")
    )
    target = action.apply(source)
    assert re.search(r"; # [A-Za-z]{16}\n", target)
    # Re-enumeration reparses with both AST and LibCST.
    inverse = _actions(engine, target, "replace_comment_line")
    assert any(candidate.apply(target) == source for candidate in inverse)


def test_mbpp_592_comment_seed_regression() -> None:
    source = (
        "\n"
        "def binomial_Coeff(n, k): \n"
        "    C = [0] * (k + 1); \n"
        "    C[0] = 1; # nC0 is 1 \n"
        "    for i in range(1,n + 1):  \n"
        "        for j in range(min(i, k),0,-1): \n"
        "            C[j] = C[j] + C[j - 1]; \n"
        "    return C[k]; \n"
        "def sum_Of_product(n): \n"
        "    return binomial_Coeff(2 * n, n - 1); \n"
    )
    engine = RandomWalkObfuscator(source, seed=9369153796502753434)
    current = source
    records = []
    for step in range(20):
        try:
            current, record = engine.step(current, step_index=step)
            records.append(record.to_dict())
        except Exception as error:
            raise AssertionError(
                f"failed at step {step}: {type(error).__name__}: {error}\n"
                f"records={records}\n{current}"
            ) from error


def test_assigned_string_pool_is_value_preserving_and_reversible() -> None:
    source = "def f():\n    value='héllo\\n世界'\n    return value\n"
    engine = RandomWalkObfuscator(source, seed=13)
    actions = _actions(engine, source, "replace_assigned_string")
    assert len(actions) == 9
    for action in actions:
        target = action.apply(source)
        before = {}
        after = {}
        exec(source, before, before)
        exec(target, after, after)
        assert before["f"]() == after["f"]()
        inverse = _actions(engine, target, "replace_assigned_string")
        assert any(candidate.apply(target) == source for candidate in inverse)


def test_each_dispatcher_shape_has_twenty_helper_name_actions() -> None:
    source = "def f(x):\n    y=x+1\n    z=y*2\n    return z\n"
    engine = RandomWalkObfuscator(source, seed=17, rules=(DispatcherRule(),))
    actions = _actions(engine, source, "flatten_straight_line")
    groups = {}
    for action in actions:
        parameters = dict(action.parameters)
        key = (action.site, parameters["labels"], parameters["order"])
        groups.setdefault(key, []).append(parameters["pc"])
    assert groups
    assert all(len(names) == 20 and len(set(names)) == 20 for names in groups.values())
    assert all(
        name.startswith("__rw_pc_") and not ALIAS_RE.fullmatch(name)
        for names in groups.values()
        for name in names
    )


def test_new_dispatcher_variable_gets_a_pool_and_must_rename_back_to_restore() -> None:
    source = "def f(x):\n    y=x+1\n    z=y*2\n    return z\n"
    engine = RandomWalkObfuscator(source, seed=19)
    flatten = _actions(engine, source, "flatten_straight_line")[0]
    flattened = flatten.apply(source)
    pc = dict(flatten.parameters)["pc"]
    rename = next(
        action
        for action in _actions(engine, flattened, "rename_variable")
        if dict(action.parameters)["old"] == pc
    )
    renamed = rename.apply(flattened)
    alias = dict(rename.parameters)["new"]
    pool = engine.runtime.variable_pool(alias)
    assert pool is not None and len(pool) == 10 and pool[0] == pc
    assert not _actions(engine, renamed, "restore_straight_line")
    inverse_renames = _actions(engine, renamed, "rename_variable")
    restored_name = next(
        action for action in inverse_renames if action.apply(renamed) == flattened
    ).apply(renamed)
    assert restored_name == flattened
    restores = _actions(engine, restored_name, "restore_straight_line")
    assert any(action.apply(restored_name) == source for action in restores)
