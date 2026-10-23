"""Reversible statement and indentation layout rules."""

from __future__ import annotations

import token
from typing import Iterable

import libcst as cst

from ..context import (
    AnalysisContext,
    canonical_line,
    canonical_suite,
    iter_nodes,
)
from ..model import Action, RuntimeState, TextEdit
from .base import replacement_action


def _canonical_trailing() -> cst.TrailingWhitespace:
    return cst.TrailingWhitespace(
        whitespace=cst.SimpleWhitespace(""),
        comment=None,
        newline=cst.Newline(),
    )


def _outer_indented(fragment: str, indent: str) -> str:
    """Add the surrounding block's indentation to continuation lines."""

    if not indent or "\n" not in fragment:
        return fragment
    pieces = fragment.splitlines(keepends=True)
    return "".join(
        piece if index == 0 else indent + piece
        for index, piece in enumerate(pieces)
    )


class SimpleStatementBoundaryRule:
    name = "simple_statement_boundary"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        yield from self._joins(context)
        yield from self._splits(context)

    def _joins(self, context: AnalysisContext) -> Iterable[Action]:
        for _, body in context.suite_bodies():
            for left, right in zip(body, body[1:]):
                if not isinstance(left, cst.SimpleStatementLine) or not isinstance(
                    right, cst.SimpleStatementLine
                ):
                    continue
                if (
                    len(left.body) != 1
                    or len(right.body) != 1
                    or not canonical_line(left)
                    or not canonical_line(right)
                    or context.is_protected(left)
                    or context.is_protected(right)
                ):
                    continue
                left_pos = context.positions[left]
                right_pos = context.positions[right]
                if right_pos.start.line != left_pos.start.line + 1:
                    continue
                left_indent = context.index.leading_indent(left_pos.start.line)
                right_indent = context.index.leading_indent(right_pos.start.line)
                if left_indent != right_indent:
                    continue
                if not context.index.line_text(left_pos.start.line).endswith("\n"):
                    continue
                if not context.index.line_text(right_pos.start.line).endswith("\n"):
                    continue

                left_small = left.body[0]
                if not hasattr(left_small, "semicolon"):
                    continue
                if left_small.semicolon is not cst.MaybeSentinel.DEFAULT:
                    continue
                left_small = left_small.with_changes(
                    semicolon=cst.Semicolon(
                        whitespace_before=cst.SimpleWhitespace(""),
                        whitespace_after=cst.SimpleWhitespace(""),
                    )
                )
                joined = cst.SimpleStatementLine(
                    body=(left_small, right.body[0]),
                    leading_lines=(),
                    trailing_whitespace=right.trailing_whitespace,
                )
                replacement = context.module.code_for_node(joined)
                start = context.span(left)[0]
                end = context.index.line_end(right_pos.end.line)
                expected = context.source[start:end]
                yield replacement_action(
                    rule="join_simple_statements",
                    inverse_rule="split_simple_statements",
                    site=f"lines@{left_pos.start.line}-{right_pos.start.line}",
                    start=start,
                    end=end,
                    expected=expected,
                    replacement=replacement,
                    parameters=(("indent", left_indent),),
                )

    def _splits(self, context: AnalysisContext) -> Iterable[Action]:
        for node in iter_nodes(context.module):
            if not isinstance(node, cst.SimpleStatementLine):
                continue
            if (
                len(node.body) != 2
                or not canonical_line(node)
                or context.is_protected(node)
                or context.positions[node].start.line != context.positions[node].end.line
            ):
                continue
            line = context.positions[node].start.line
            indent = context.index.leading_indent(line)
            for cut in range(1, len(node.body)):
                boundary = node.body[cut - 1]
                comma = getattr(boundary, "semicolon", cst.MaybeSentinel.DEFAULT)
                if not isinstance(comma, cst.Semicolon):
                    continue
                if (
                    comma.whitespace_before.value != ""
                    or comma.whitespace_after.value != ""
                ):
                    continue
                first_last = boundary.with_changes(semicolon=cst.MaybeSentinel.DEFAULT)
                first_body = tuple(node.body[: cut - 1]) + (first_last,)
                second_body = tuple(node.body[cut:])
                first_line = cst.SimpleStatementLine(
                    body=first_body,
                    leading_lines=(),
                    trailing_whitespace=_canonical_trailing(),
                )
                second_line = cst.SimpleStatementLine(
                    body=second_body,
                    leading_lines=(),
                    trailing_whitespace=node.trailing_whitespace,
                )
                replacement = (
                    context.module.code_for_node(first_line)
                    + indent
                    + context.module.code_for_node(second_line)
                )
                start = context.span(node)[0]
                end = context.index.line_end(context.positions[node].end.line)
                expected = context.source[start:end]
                yield replacement_action(
                    rule="split_simple_statements",
                    inverse_rule="join_simple_statements",
                    site=f"semicolon@{line}:{cut}",
                    start=start,
                    end=end,
                    expected=expected,
                    replacement=replacement,
                    parameters=(("indent", indent), ("cut", str(cut))),
                )


SUPPORTED_COMPOUNDS = (
    cst.FunctionDef,
    cst.ClassDef,
    cst.If,
    cst.For,
    cst.While,
    cst.With,
)


class SimpleSuiteRule:
    name = "simple_suite"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for node in iter_nodes(context.module):
            if not isinstance(node, SUPPORTED_COMPOUNDS) or context.is_protected(node):
                continue
            if self._is_elif(context, node):
                # LibCST represents ``elif`` as an If stored in another If's
                # orelse. Rendering that child alone spells it as ``if`` and
                # changes the surrounding branch chain.
                continue
            body = node.body
            if isinstance(body, cst.IndentedBlock):
                action = self._inline(context, node, body)
                if action is not None:
                    yield action
            elif isinstance(body, cst.SimpleStatementSuite):
                action = self._expand(context, node, body)
                if action is not None:
                    yield action

    def _inline(
        self,
        context: AnalysisContext,
        node: cst.CSTNode,
        block: cst.IndentedBlock,
    ) -> Action | None:
        if len(block.body) != 1 or block.footer:
            return None
        if self._has_prefix_trivia(node):
            return None
        line = block.body[0]
        if not isinstance(line, cst.SimpleStatementLine) or not canonical_line(line):
            return None
        if not self._single_unterminated_small_statement(line.body):
            return None
        node_pos = context.positions[node]
        line_pos = context.positions[line]
        outer_indent = context.index.leading_indent(node_pos.start.line)
        child_indent = context.index.leading_indent(line_pos.start.line)
        if child_indent != outer_indent + "    ":
            return None
        original = context.text(node)
        if (
            "#" in original
            or ";" in original
            or "\r" in original
            or self._has_explicit_semicolon(node)
        ):
            return None
        suite = cst.SimpleStatementSuite(
            body=tuple(line.body),
            leading_whitespace=cst.SimpleWhitespace(""),
            trailing_whitespace=line.trailing_whitespace,
        )
        changed = node.with_changes(body=suite)
        replacement = _outer_indented(
            context.module.code_for_node(changed), outer_indent
        )
        if replacement.endswith("\n"):
            replacement = replacement[:-1]
        start, end = context.span(node)
        return replacement_action(
            rule="inline_simple_suite",
            inverse_rule="expand_simple_suite",
            site=f"suite@{node_pos.start.line}",
            start=start,
            end=end,
            expected=original,
            replacement=replacement,
            parameters=(("indent", "4spaces"),),
        )

    def _expand(
        self,
        context: AnalysisContext,
        node: cst.CSTNode,
        suite: cst.SimpleStatementSuite,
    ) -> Action | None:
        if not canonical_suite(suite):
            return None
        if not self._single_unterminated_small_statement(suite.body):
            return None
        if self._has_prefix_trivia(node):
            return None
        original = context.text(node)
        suite_text = context.module.code_for_node(suite)
        if (
            "#" in original
            or ";" in original
            or self._has_explicit_semicolon(node)
            or not suite_text.endswith("\n")
            or "\n" in suite_text[:-1]
        ):
            return None
        line = cst.SimpleStatementLine(
            body=tuple(suite.body),
            leading_lines=(),
            trailing_whitespace=suite.trailing_whitespace,
        )
        block = cst.IndentedBlock(
            body=(line,),
            header=cst.Newline(),
            indent="    ",
            footer=(),
        )
        changed = node.with_changes(body=block)
        node_pos = context.positions[node]
        outer_indent = context.index.leading_indent(node_pos.start.line)
        replacement = _outer_indented(
            context.module.code_for_node(changed), outer_indent
        )
        if replacement.endswith("\n"):
            replacement = replacement[:-1]
        start, end = context.span(node)
        return replacement_action(
            rule="expand_simple_suite",
            inverse_rule="inline_simple_suite",
            site=f"suite@{node_pos.start.line}",
            start=start,
            end=end,
            expected=original,
            replacement=replacement,
            parameters=(("indent", "4spaces"),),
        )

    @staticmethod
    def _has_prefix_trivia(node: cst.CSTNode) -> bool:
        return bool(
            getattr(node, "leading_lines", ())
            or getattr(node, "decorators", ())
            or getattr(node, "lines_after_decorators", ())
        )

    @staticmethod
    def _has_explicit_semicolon(node: cst.CSTNode) -> bool:
        return any(
            isinstance(
                getattr(candidate, "semicolon", cst.MaybeSentinel.DEFAULT),
                cst.Semicolon,
            )
            for candidate in iter_nodes(node)
        )

    @staticmethod
    def _is_elif(context: AnalysisContext, node: cst.CSTNode) -> bool:
        parent = context.parents.get(node)
        return (
            isinstance(node, cst.If)
            and isinstance(parent, cst.If)
            and parent.orelse is node
        )

    @staticmethod
    def _single_unterminated_small_statement(
        body: tuple[cst.BaseSmallStatement, ...],
    ) -> bool:
        return (
            len(body) == 1
            and getattr(body[0], "semicolon", cst.MaybeSentinel.DEFAULT)
            is cst.MaybeSentinel.DEFAULT
        )


class IndentationUnitRule:
    name = "indentation_unit"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        multiline_string_lines = self._multiline_string_lines(context)
        continuation_lines = self._continuation_lines(context)
        for node in iter_nodes(context.module):
            if not isinstance(node, cst.IndentedBlock) or not node.body:
                continue
            if context.is_protected(node):
                continue
            owner = context.parents.get(node)
            if owner is None or owner not in context.positions:
                continue
            first_line = context.positions[node.body[0]].start.line
            last_line = context.positions[node.body[-1]].end.line
            if any(
                line in multiline_string_lines or line in continuation_lines
                for line in range(first_line, last_line + 1)
            ):
                continue
            owner_line = context.positions[owner].start.line
            parent_indent = context.index.leading_indent(owner_line)
            first_indent = context.index.leading_indent(first_line)
            if not first_indent.startswith(parent_indent):
                continue
            old_unit = first_indent[len(parent_indent) :]
            if old_unit == "    ":
                old_unit, new_unit = "    ", "\t"
                rule, inverse = "four_spaces_to_tab", "tab_to_four_spaces"
            elif old_unit == "\t":
                old_unit, new_unit = "\t", "    "
                rule, inverse = "tab_to_four_spaces", "four_spaces_to_tab"
            else:
                continue
            edits: list[TextEdit] = []
            valid = True
            for line_number in range(first_line, last_line + 1):
                line = context.index.line_text(line_number)
                stripped = line.strip()
                if not stripped:
                    if line.rstrip("\r\n"):
                        valid = False
                    continue
                if stripped.startswith("#"):
                    valid = False
                    break
                indent = context.index.leading_indent(line_number)
                prefix = parent_indent + old_unit
                if not indent.startswith(prefix):
                    valid = False
                    break
                start = context.index.line_start(line_number) + len(parent_indent)
                edits.append(TextEdit(start, start + len(old_unit), old_unit, new_unit))
            if valid and edits:
                yield Action(
                    rule=rule,
                    inverse_rule=inverse,
                    site=f"block@{first_line}",
                    edits=tuple(edits),
                )

    @staticmethod
    def _multiline_string_lines(context: AnalysisContext) -> set[int]:
        result: set[int] = set()
        for item in context.tokens:
            if item.type == token.STRING and item.start[0] != item.end[0]:
                result.update(range(item.start[0], item.end[0] + 1))
        return result

    @staticmethod
    def _continuation_lines(context: AnalysisContext) -> set[int]:
        result: set[int] = set()
        for item in context.tokens:
            if item.type == token.NL:
                result.add(item.start[0])
                result.add(item.start[0] + 1)
        return result
