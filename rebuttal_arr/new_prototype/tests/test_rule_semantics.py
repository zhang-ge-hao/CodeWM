from __future__ import annotations

from dataclasses import dataclass
import ast

import pytest

from rw_obfuscator.rules.advanced import DispatcherRule, OpaquePredicateRule
from rw_obfuscator.rules.lexical import (
    GroupingParenthesesRule,
    OptionalSpaceRule,
    TrailingCommaRule,
)
from rw_obfuscator.rules.pyminifier import (
    BuiltinRaiseParenthesesRule,
    FinalBareReturnRule,
    IntegerTemplateRule,
    ObjectBaseRule,
    PlainImportBoundaryRule,
    RedundantPassRule,
    ReturnNoneRule,
    SolePassZeroRule,
)
from rw_obfuscator.rules.structural import (
    IndentationUnitRule,
    SimpleStatementBoundaryRule,
    SimpleSuiteRule,
)
from rw_obfuscator.rules.variable import VariableRenameRule

from tests.helpers import matching_actions, observe


@dataclass(frozen=True)
class SemanticCase:
    name: str
    enumerator: object
    action_rule: str
    source: str
    observation: str


CASES = (
    SemanticCase(
        "rename-variable",
        VariableRenameRule(),
        "rename_variable",
        "def f(x):\n    y=x+1\n    return y*2\n",
        "(f(0), f(5))",
    ),
    SemanticCase(
        "delete-space",
        OptionalSpaceRule(),
        "delete_optional_space",
        "def f(x):\n    y = x + 1\n    return y\n",
        "(f(-1), f(4))",
    ),
    SemanticCase(
        "insert-space",
        OptionalSpaceRule(),
        "insert_optional_space",
        "def f(x):\n    y=x+1\n    return y\n",
        "(f(-1), f(4))",
    ),
    SemanticCase(
        "delete-grouping-parentheses",
        GroupingParenthesesRule(),
        "delete_grouping_parentheses",
        "def f():\n    x=(1+2)\n    return x\n",
        "f()",
    ),
    SemanticCase(
        "insert-grouping-parentheses",
        GroupingParenthesesRule(),
        "insert_grouping_parentheses",
        "def f():\n    x=1+2\n    return x\n",
        "f()",
    ),
    SemanticCase(
        "delete-trailing-comma",
        TrailingCommaRule(),
        "delete_trailing_comma",
        "def f():\n    return list((1,2,))\n",
        "f()",
    ),
    SemanticCase(
        "insert-trailing-comma",
        TrailingCommaRule(),
        "insert_trailing_comma",
        "def f():\n    return list((1,2))\n",
        "f()",
    ),
    SemanticCase(
        "join-statements",
        SimpleStatementBoundaryRule(),
        "join_simple_statements",
        "def f():\n    x=1\n    y=2\n    return x+y\n",
        "f()",
    ),
    SemanticCase(
        "split-statements",
        SimpleStatementBoundaryRule(),
        "split_simple_statements",
        "def f():\n    x=1;y=2\n    return x+y\n",
        "f()",
    ),
    SemanticCase(
        "inline-suite",
        SimpleSuiteRule(),
        "inline_simple_suite",
        "def f():\n    return 3\n",
        "f()",
    ),
    SemanticCase(
        "expand-suite",
        SimpleSuiteRule(),
        "expand_simple_suite",
        "def f():return 3\n",
        "f()",
    ),
    SemanticCase(
        "spaces-to-tab",
        IndentationUnitRule(),
        "four_spaces_to_tab",
        "def f(x):\n    y=x+1\n    return y\n",
        "f(4)",
    ),
    SemanticCase(
        "tab-to-spaces",
        IndentationUnitRule(),
        "tab_to_four_spaces",
        "def f(x):\n\ty=x+1\n\treturn y\n",
        "f(4)",
    ),
    SemanticCase(
        "merge-imports",
        PlainImportBoundaryRule(),
        "merge_adjacent_plain_imports",
        "import math\nimport sys\ndef f():\n    return math.isqrt(81)+int(sys.version_info.major>0)\n",
        "f()",
    ),
    SemanticCase(
        "split-import",
        PlainImportBoundaryRule(),
        "split_plain_import",
        "import math,sys\ndef f():\n    return math.isqrt(81)+int(sys.version_info.major>0)\n",
        "f()",
    ),
    SemanticCase(
        "delete-pass",
        RedundantPassRule(),
        "delete_redundant_pass",
        "def f():\n    x=3\n    pass\n    return x\n",
        "f()",
    ),
    SemanticCase(
        "insert-pass",
        RedundantPassRule(),
        "insert_redundant_pass",
        "def f():\n    x=3\n    return x\n",
        "f()",
    ),
    SemanticCase(
        "sole-pass-to-zero",
        SolePassZeroRule(),
        "sole_pass_to_zero",
        "def f():\n    pass\n",
        "f()",
    ),
    SemanticCase(
        "sole-zero-to-pass",
        SolePassZeroRule(),
        "sole_zero_to_pass",
        "def f():\n    0\n",
        "f()",
    ),
    SemanticCase(
        "return-none-to-bare",
        ReturnNoneRule(),
        "return_none_to_bare",
        "def f():\n    return None\n",
        "f()",
    ),
    SemanticCase(
        "bare-return-to-none",
        ReturnNoneRule(),
        "bare_return_to_none",
        "def f():\n    return\n",
        "f()",
    ),
    SemanticCase(
        "delete-final-return",
        FinalBareReturnRule(),
        "delete_final_bare_return",
        "def f():\n    x=3\n    return\n",
        "f()",
    ),
    SemanticCase(
        "append-final-return",
        FinalBareReturnRule(),
        "append_final_bare_return",
        "def f():\n    x=3\n",
        "f()",
    ),
    SemanticCase(
        "delete-object-base",
        ObjectBaseRule(),
        "delete_builtin_object_base",
        "class C(object):\n    def value(self):\n        return 3\n",
        "(C.__bases__[0] is object,C().value())",
    ),
    SemanticCase(
        "insert-object-base",
        ObjectBaseRule(),
        "insert_builtin_object_base",
        "class C:\n    def value(self):\n        return 3\n",
        "(C.__bases__[0] is object,C().value())",
    ),
    SemanticCase(
        "delete-raise-parentheses",
        BuiltinRaiseParenthesesRule(),
        "delete_builtin_raise_parentheses",
        "def f():\n    raise ValueError()\n",
        "f()",
    ),
    SemanticCase(
        "insert-raise-parentheses",
        BuiltinRaiseParenthesesRule(),
        "insert_builtin_raise_parentheses",
        "def f():\n    raise ValueError\n",
        "f()",
    ),
    SemanticCase(
        "expand-add-sub",
        IntegerTemplateRule(),
        "expand_integer_add_sub",
        "def f():\n    return 37\n",
        "f()",
    ),
    SemanticCase(
        "fold-add-sub",
        IntegerTemplateRule(),
        "fold_integer_add_sub",
        "def f():\n    return ((37+5)-5)\n",
        "f()",
    ),
    SemanticCase(
        "expand-xor",
        IntegerTemplateRule(),
        "expand_integer_xor",
        "def f():\n    return 37\n",
        "f()",
    ),
    SemanticCase(
        "fold-xor",
        IntegerTemplateRule(),
        "fold_integer_xor",
        "def f():\n    return ((37^5)^5)\n",
        "f()",
    ),
    SemanticCase(
        "true-opaque",
        OpaquePredicateRule(),
        "insert_true_opaque_guard",
        "def f(x):\n    y=x+1\n    return y\n",
        "(f(-1),f(4))",
    ),
    SemanticCase(
        "false-opaque",
        OpaquePredicateRule(),
        "insert_false_opaque_guard",
        "def f(x):\n    y=x+1\n    return y\n",
        "(f(-1),f(4))",
    ),
    SemanticCase(
        "straight-line-dispatcher",
        DispatcherRule(),
        "flatten_straight_line",
        "def f(x):\n    y=x+1\n    z=y*2\n    w=z-3\n    return w\n",
        "(f(-1),f(4))",
    ),
    SemanticCase(
        "conditional-dispatcher",
        DispatcherRule(),
        "flatten_simple_if",
        "def f(x):\n    if x>0:\n        y=x+1\n    else:\n        y=-x\n    return y\n",
        "(f(-2),f(4))",
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_every_rule_family_preserves_behavior(case: SemanticCase) -> None:
    baseline = observe(case.source, case.observation)
    assert baseline[0] in {"value", "exception"}
    actions = matching_actions(
        case.source,
        case.action_rule,
        rules=(case.enumerator,),
    )
    assert actions, f"no {case.action_rule} action"

    for action in actions:
        target = action.apply(case.source)
        ast.parse(target)
        assert observe(target, case.observation) == baseline, (
            f"semantic change from {action.rule} at {action.site}:\n{target}"
        )
