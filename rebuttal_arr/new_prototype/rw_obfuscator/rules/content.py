"""Reversible pools for comments, standalone strings, and assigned strings."""

from __future__ import annotations

import ast
import token
from typing import Iterable

import libcst as cst
from libcst.metadata import CodePosition

from ..context import AnalysisContext, iter_nodes
from ..model import Action, RuntimeState, TextEdit


def _line_content(piece: str) -> tuple[str, str]:
    if piece.endswith("\r\n"):
        return piece[:-2], "\r\n"
    if piece.endswith(("\n", "\r")):
        return piece[:-1], piece[-1:]
    return piece, ""


class CommentLineRule:
    """Replace comments without moving their source locations.

    A single ``#`` token has a ten-state star pool.  Consecutive full-line
    comments and standalone string expressions are each one indivisible unit
    with a 101-state star pool (the original plus 100 aliases).
    """

    name = "comment_line"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        yield from self._token_comments(context, runtime)
        yield from self._standalone_strings(context, runtime)

    def _token_comments(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        comments = [item for item in context.tokens if item.type == token.COMMENT]
        full_line = {
            index
            for index, item in enumerate(comments)
            if context.index.line_text(item.start[0])[: item.start[1]].strip() == ""
        }
        blocks: list[list[int]] = []
        index = 0
        while index < len(comments):
            if index not in full_line:
                index += 1
                continue
            block = [index]
            while (
                block[-1] + 1 < len(comments)
                and block[-1] + 1 in full_line
                and comments[block[-1] + 1].start[0]
                == comments[block[-1]].start[0] + 1
            ):
                block.append(block[-1] + 1)
            if len(block) > 1:
                blocks.append(block)
            index = block[-1] + 1

        blocked = {item for block in blocks for item in block}
        for block in blocks:
            yield from self._comment_block(
                context,
                runtime,
                comments[block[0]].start[0],
                comments[block[-1]].start[0],
            )

        for index, item in enumerate(comments):
            if index in blocked:
                continue
            current = item.string
            runtime.ensure_comment_pool(current)
            start = context.index.offset(CodePosition(item.start[0], item.start[1]))
            end = context.index.offset(CodePosition(item.end[0], item.end[1]))
            for target in runtime.comment_targets(current):
                yield Action(
                    rule="replace_comment_line",
                    inverse_rule="replace_comment_line",
                    site=f"comment@{item.start[0]}:{item.start[1]}",
                    edits=(TextEdit(start, end, current, target),),
                    parameters=(("old", current), ("new", target)),
                )

    def _comment_block(
        self,
        context: AnalysisContext,
        runtime: RuntimeState,
        first_line: int,
        last_line: int,
    ) -> Iterable[Action]:
        start = context.index.line_start(first_line)
        end = context.index.line_end(last_line)
        current = context.source[start:end]
        if runtime.block_comment_pool(current) is None:
            layouts: list[tuple[str, str]] = []
            for line_number in range(first_line, last_line + 1):
                body, ending = _line_content(context.index.line_text(line_number))
                indent = body[: len(body) - len(body.lstrip(" \t"))]
                layouts.append((indent, ending))
            replacements = tuple(
                self._render_comment_block(layouts, runtime.new_comment_alias())
                for _ in range(100)
            )
            runtime.register_block_comment_pool(current, (current, *replacements))

        for target in runtime.block_comment_targets(current):
            yield Action(
                rule="replace_comment_block",
                inverse_rule="replace_comment_block",
                site=f"comment-block@{first_line}-{last_line}",
                edits=(TextEdit(start, end, current, target),),
                parameters=(("old", current), ("new", target)),
            )

    @staticmethod
    def _render_comment_block(
        layouts: list[tuple[str, str]], alias: str
    ) -> str:
        return "".join(
            indent + (f"# {alias}" if index == 0 else "#") + ending
            for index, (indent, ending) in enumerate(layouts)
        )

    def _standalone_strings(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for node in iter_nodes(context.module):
            if not isinstance(node, cst.Expr) or not isinstance(
                node.value, cst.SimpleString
            ):
                continue
            literal = node.value
            prefix = literal.prefix
            if "b" in prefix.lower() or "f" in prefix.lower():
                continue
            quote = literal.quote
            raw = literal.raw_value
            if ("\n" in raw or "\r" in raw) and quote not in {"'''", '\"\"\"'}:
                continue
            start, end = context.span(literal)
            current = context.source[start:end]
            if runtime.block_comment_pool(current) is None:
                endings = "".join(
                    ending
                    for piece in raw.splitlines(keepends=True)
                    for _, ending in (_line_content(piece),)
                )
                replacements = tuple(
                    prefix + quote + runtime.new_comment_alias() + endings + quote
                    for _ in range(100)
                )
                runtime.register_block_comment_pool(current, (current, *replacements))

            line = context.positions[literal].start.line
            for target in runtime.block_comment_targets(current):
                yield Action(
                    rule="replace_comment_block",
                    inverse_rule="replace_comment_block",
                    site=f"standalone-string@{line}",
                    edits=(TextEdit(start, end, current, target),),
                    parameters=(("old", current), ("new", target)),
                )


class AssignedStringRule:
    """Move direct assigned string literals through ten equivalent spellings.

    Generated states use only Unicode escapes and implicit literal
    concatenation.  They are folded by Python itself and introduce no runtime
    helper calls, imports, or variables.
    """

    name = "assigned_string"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for node in iter_nodes(context.module):
            if isinstance(node, cst.Assign):
                value = node.value
            elif isinstance(node, cst.AnnAssign) and node.value is not None:
                value = node.value
            else:
                continue
            current = context.text(value)
            pool = runtime.string_pool(current)
            if pool is None:
                if not isinstance(value, cst.SimpleString):
                    continue
                try:
                    evaluated = ast.literal_eval(value.value)
                except (SyntaxError, ValueError):
                    continue
                if not isinstance(evaluated, str):
                    continue
                pool = runtime.register_string_pool(
                    current,
                    self._pool_members(
                        current,
                        evaluated,
                        runtime.config.replacement_pool_size,
                    ),
                )
            start, end = context.span(value)
            for target in pool:
                if target == current:
                    continue
                yield Action(
                    rule="replace_assigned_string",
                    inverse_rule="replace_assigned_string",
                    site=(
                        f"assigned-string@{context.positions[value].start.line}:"
                        f"{context.positions[value].start.column}"
                    ),
                    edits=(TextEdit(start, end, current, target),),
                    parameters=(("old", current), ("new", target)),
                )

    @staticmethod
    def _pool_members(root: str, value: str, size: int) -> tuple[str, ...]:
        if size != 10:
            raise ValueError("assigned-string rule currently requires pool size 10")
        encoded = "'" + "".join(f"\\U{ord(char):08x}" for char in value) + "'"
        variants = (
            f"u'' {encoded}",
            f"{encoded} u''",
            f'\"\" {encoded}',
            f'{encoded} \"\"',
            f'u\"\" {encoded} u\'\'',
            f"u'' {encoded} u\"\"",
            f'\"\" u\'\' {encoded}',
            f'{encoded} u\'\' \"\"',
            f'u\"\" u\'\' {encoded} u\"\"',
        )
        return (root, *variants)


__all__ = ["AssignedStringRule", "CommentLineRule"]
