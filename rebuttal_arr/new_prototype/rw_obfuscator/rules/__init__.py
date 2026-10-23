"""Default rule registry."""

from .advanced import DispatcherRule, OpaquePredicateRule
from .content import AssignedStringRule, CommentLineRule
from .lexical import (
    GroupingParenthesesRule,
    NumericSpellingRule,
    OptionalSpaceRule,
    TrailingCommaRule,
)
from .pyminifier import (
    BuiltinRaiseParenthesesRule,
    FinalBareReturnRule,
    IntegerTemplateRule,
    ObjectBaseRule,
    PlainImportBoundaryRule,
    RedundantPassRule,
    ReturnNoneRule,
    SolePassZeroRule,
)
from .structural import IndentationUnitRule, SimpleStatementBoundaryRule, SimpleSuiteRule
from .variable import VariableRenameRule


def default_rules() -> tuple[object, ...]:
    """Return stateless enumerators in a stable order."""

    return (
        VariableRenameRule(),
        CommentLineRule(),
        AssignedStringRule(),
        OptionalSpaceRule(),
        NumericSpellingRule(),
        GroupingParenthesesRule(),
        TrailingCommaRule(),
        SimpleStatementBoundaryRule(),
        SimpleSuiteRule(),
        IndentationUnitRule(),
        PlainImportBoundaryRule(),
        RedundantPassRule(),
        SolePassZeroRule(),
        ReturnNoneRule(),
        FinalBareReturnRule(),
        ObjectBaseRule(),
        BuiltinRaiseParenthesesRule(),
        IntegerTemplateRule(),
        OpaquePredicateRule(),
        DispatcherRule(),
    )


__all__ = ["default_rules"]
