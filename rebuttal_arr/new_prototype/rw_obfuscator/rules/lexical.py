"""Reversible source-spelling rules adapted from Python-Minifier's printer."""

from __future__ import annotations

import re
import token
from typing import Iterable, Iterator

import libcst as cst
from libcst.metadata import CodePosition

from ..context import AnalysisContext, iter_nodes
from ..model import Action, RuntimeState, TextEdit
from .base import replacement_action


SAFE_SPACE_OPERATORS = frozenset(
    {
        "=",
        "+",
        "-",
        "*",
        "/",
        "//",
        "%",
        "@",
        "<<",
        ">>",
        "&",
        "|",
        "^",
        "~",
        "<",
        ">",
        "<=",
        ">=",
        "==",
        "!=",
        "+=",
        "-=",
        "*=",
        "/=",
        "//=",
        "%=",
        "@=",
        "&=",
        "|=",
        "^=",
        ">>=",
        "<<=",
        "**=",
        ":=",
        ",",
        ":",
    }
)


class OptionalSpaceRule:
    name = "optional_space"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        tokens = context.tokens
        for left, right in zip(tokens, tokens[1:]):
            if left.end[0] != right.start[0]:
                continue
            if left.type in {token.INDENT, token.DEDENT, token.NEWLINE, token.NL}:
                continue
            if right.type in {token.INDENT, token.DEDENT, token.NEWLINE, token.NL}:
                continue
            if left.type == token.COMMENT or right.type == token.COMMENT:
                continue
            left_is_op = left.type == token.OP and left.string in SAFE_SPACE_OPERATORS
            right_is_op = right.type == token.OP and right.string in SAFE_SPACE_OPERATORS
            if left_is_op == right_is_op:
                # Exactly one side must be a whitelisted symbolic operator.
                continue
            if "__rw_" in context.index.line_text(left.end[0]):
                continue

            start = context.index.offset(
                CodePosition(left.end[0], left.end[1])
            )
            end = context.index.offset(
                CodePosition(right.start[0], right.start[1])
            )
            gap = context.source[start:end]
            if gap == " ":
                yield replacement_action(
                    rule="delete_optional_space",
                    inverse_rule="insert_optional_space",
                    site=f"gap@{left.end[0]}:{left.end[1]}",
                    start=start,
                    end=end,
                    expected=" ",
                    replacement="",
                )
            elif gap == "":
                yield replacement_action(
                    rule="insert_optional_space",
                    inverse_rule="delete_optional_space",
                    site=f"gap@{left.end[0]}:{left.end[1]}",
                    start=start,
                    end=end,
                    expected="",
                    replacement=" ",
                )


class NumericSpellingRule:
    name = "numeric_spelling"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        significant = [
            item
            for item in context.tokens
            if item.type not in {token.INDENT, token.DEDENT, token.NL, token.NEWLINE}
        ]
        for index, item in enumerate(significant):
            if item.type != token.NUMBER:
                continue
            if "__rw_" in context.index.line_text(item.start[0]):
                continue
            if index + 1 < len(significant):
                following = significant[index + 1]
                if following.type == token.OP and following.string == ".":
                    continue
            start = context.index.offset(
                CodePosition(item.start[0], item.start[1])
            )
            end = context.index.offset(
                CodePosition(item.end[0], item.end[1])
            )
            for pair_name, replacement in self._pairs(item.string):
                yield replacement_action(
                    rule=f"numeric_{pair_name}",
                    inverse_rule=f"numeric_{self._inverse_name(pair_name)}",
                    site=f"number@{item.start[0]}:{item.start[1]}",
                    start=start,
                    end=end,
                    expected=item.string,
                    replacement=replacement,
                    parameters=(("from", item.string), ("to", replacement)),
                )

    @staticmethod
    def _pairs(value: str) -> Iterator[tuple[str, str]]:
        match = re.fullmatch(r"0\.(\d+)", value)
        if match:
            yield "delete_leading_zero", f".{match.group(1)}"
        match = re.fullmatch(r"\.(\d+)", value)
        if match:
            yield "insert_leading_zero", f"0.{match.group(1)}"

        match = re.fullmatch(r"(\d+)\.0", value)
        if match:
            yield "delete_trailing_zero", f"{match.group(1)}."
        match = re.fullmatch(r"(\d+)\.", value)
        if match:
            yield "insert_trailing_zero", f"{match.group(1)}.0"

        exponent_prefix = r"((?:\d+(?:\.\d*)?|\.\d+)[eE])"
        match = re.fullmatch(exponent_prefix + r"\+(\d+)", value)
        if match:
            yield "delete_exponent_plus", f"{match.group(1)}{match.group(2)}"
        match = re.fullmatch(exponent_prefix + r"(\d+)", value)
        if match:
            yield "insert_exponent_plus", f"{match.group(1)}+{match.group(2)}"

        if re.fullmatch(r"(?:0|[1-9]\d*)", value):
            integer = int(value)
            if integer >= 256:
                yield "decimal_to_hex", hex(integer)
        elif re.fullmatch(r"0x[0-9a-f]+", value):
            yield "hex_to_decimal", str(int(value, 16))

    @staticmethod
    def _inverse_name(name: str) -> str:
        pairs = {
            "delete_leading_zero": "insert_leading_zero",
            "insert_leading_zero": "delete_leading_zero",
            "delete_trailing_zero": "insert_trailing_zero",
            "insert_trailing_zero": "delete_trailing_zero",
            "delete_exponent_plus": "insert_exponent_plus",
            "insert_exponent_plus": "delete_exponent_plus",
            "decimal_to_hex": "hex_to_decimal",
            "hex_to_decimal": "decimal_to_hex",
        }
        return pairs[name]


SAFE_PAREN_PARENTS = (cst.Assign, cst.AnnAssign, cst.AugAssign, cst.Return)
UNSAFE_PAREN_EXPRESSIONS = (
    cst.Tuple,
    cst.GeneratorExp,
    cst.Yield,
    cst.Lambda,
    cst.NamedExpr,
)


class GroupingParenthesesRule:
    name = "grouping_parentheses"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for node in iter_nodes(context.module):
            if not isinstance(node, cst.BaseExpression):
                continue
            if isinstance(node, UNSAFE_PAREN_EXPRESSIONS) or context.is_protected(node):
                continue
            parent = context.parents.get(node)
            if not isinstance(parent, SAFE_PAREN_PARENTS):
                continue
            if not self._is_entire_value(parent, node):
                continue
            text = context.text(node)
            if "\n" in text or "#" in text:
                continue
            lpar = getattr(node, "lpar", ())
            rpar = getattr(node, "rpar", ())
            if len(lpar) == 0 and len(rpar) == 0:
                start, end = context.span(node)
                yield Action(
                    rule="insert_grouping_parentheses",
                    inverse_rule="delete_grouping_parentheses",
                    site=f"expr@{context.positions[node].start.line}:{context.positions[node].start.column}",
                    edits=(
                        TextEdit(start, start, "", "("),
                        TextEdit(end, end, "", ")"),
                    ),
                )
            elif len(lpar) == 1 and len(rpar) == 1:
                left_start, left_end = context.span(lpar[0])
                right_start, right_end = context.span(rpar[0])
                # ``return(expr)`` is valid Python, but deleting its only
                # grouping parentheses would concatenate the keyword and the
                # first token (for example, ``returnsum(expr)``).  Replacing
                # the parenthesis with whitespace would not have the exact
                # insertion action as its one-step inverse, so this spelling
                # is intentionally not a deletion candidate.  ``return
                # (expr)`` remains eligible and round-trips byte-for-byte.
                adjacent_return = (
                    isinstance(parent, cst.Return)
                    and left_start > 0
                    and not context.source[left_start - 1].isspace()
                )
                if (
                    not adjacent_return
                    and context.source[left_start:left_end] == "("
                    and context.source[right_start:right_end] == ")"
                ):
                    yield Action(
                        rule="delete_grouping_parentheses",
                        inverse_rule="insert_grouping_parentheses",
                        site=f"expr@{context.positions[node].start.line}:{context.positions[node].start.column}",
                        edits=(
                            TextEdit(left_start, left_end, "(", ""),
                            TextEdit(right_start, right_end, ")", ""),
                        ),
                    )

    @staticmethod
    def _is_entire_value(parent: cst.CSTNode, node: cst.BaseExpression) -> bool:
        if isinstance(parent, (cst.Assign, cst.AnnAssign, cst.AugAssign)):
            return parent.value is node
        if isinstance(parent, cst.Return):
            return parent.value is node
        return False


class TrailingCommaRule:
    name = "trailing_comma"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for node in iter_nodes(context.module):
            items, closing = self._items(node)
            if not items or context.is_protected(node):
                continue
            if isinstance(node, cst.Tuple) and len(items) < 2:
                continue
            if context.positions[node].start.line != context.positions[node].end.line:
                continue
            if "#" in context.text(node):
                continue
            last = items[-1]
            comma = getattr(last, "comma", cst.MaybeSentinel.DEFAULT)
            if isinstance(comma, cst.Comma):
                start, end = context.span(comma)
                if context.source[start:end] == ",":
                    yield replacement_action(
                        rule="delete_trailing_comma",
                        inverse_rule="insert_trailing_comma",
                        site=f"comma@{context.positions[comma].start.line}:{context.positions[comma].start.column}",
                        start=start,
                        end=end,
                        expected=",",
                        replacement="",
                    )
            elif comma is cst.MaybeSentinel.DEFAULT:
                if self._unparenthesized_generator_argument(node, last):
                    # ``f(x for x in xs)`` is legal only while the generator
                    # is the sole, comma-free argument.  ``f(x for x in xs,)``
                    # is a syntax error, whereas an explicitly parenthesized
                    # generator remains safe and reversible.
                    continue
                _, item_end = context.span(last)
                if isinstance(node, cst.Tuple):
                    if not node.rpar:
                        continue
                    closing_start = context.span(node.rpar[0])[0]
                else:
                    _, node_end = context.span(node)
                    closing_start = node_end - len(closing)
                tail = context.source[item_end:closing_start]
                if tail == "":
                    yield replacement_action(
                        rule="insert_trailing_comma",
                        inverse_rule="delete_trailing_comma",
                        site=f"closing@{context.positions[node].end.line}:{context.positions[node].end.column}",
                        start=item_end,
                        end=item_end,
                        expected="",
                        replacement=",",
                    )

    @staticmethod
    def _unparenthesized_generator_argument(
        node: cst.CSTNode,
        item: cst.CSTNode,
    ) -> bool:
        return (
            isinstance(node, cst.Call)
            and isinstance(item, cst.Arg)
            and isinstance(item.value, cst.GeneratorExp)
            and not item.value.lpar
            and not item.value.rpar
        )

    @staticmethod
    def _items(
        node: cst.CSTNode,
    ) -> tuple[tuple[cst.CSTNode, ...], str]:
        if isinstance(node, cst.Call):
            return tuple(node.args), ")"
        if isinstance(node, cst.List):
            return tuple(node.elements), "]"
        if isinstance(node, cst.Set):
            return tuple(node.elements), "}"
        if isinstance(node, cst.Dict):
            return tuple(node.elements), "}"
        if isinstance(node, cst.Tuple) and node.lpar and node.rpar:
            return tuple(node.elements), ")"
        return (), ""
