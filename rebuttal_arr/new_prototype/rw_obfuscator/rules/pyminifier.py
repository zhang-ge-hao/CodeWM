"""Conservative reversible counterparts of Python-Minifier AST transforms."""

from __future__ import annotations

import re
from typing import Iterable

import libcst as cst

from ..context import AnalysisContext, canonical_line, iter_nodes
from ..model import Action, RuntimeState, TextEdit
from .base import replacement_action


class PlainImportBoundaryRule:
    name = "plain_import_boundary"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        yield from self._merges(context)
        yield from self._splits(context)

    def _merges(self, context: AnalysisContext) -> Iterable[Action]:
        for _, body in context.suite_bodies():
            for left, right in zip(body, body[1:]):
                if not self._single_import(left) or not self._single_import(right):
                    continue
                assert isinstance(left, cst.SimpleStatementLine)
                assert isinstance(right, cst.SimpleStatementLine)
                if (
                    not canonical_line(left)
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
                left_import = left.body[0]
                right_import = right.body[0]
                assert isinstance(left_import, cst.Import)
                assert isinstance(right_import, cst.Import)
                if not self._canonical_single_import(context, left_import):
                    continue
                if not self._canonical_single_import(context, right_import):
                    continue
                merged = cst.Import(
                    names=self._canonical_aliases(
                        tuple(left_import.names) + tuple(right_import.names)
                    )
                )
                line = cst.SimpleStatementLine(
                    body=(merged,),
                    leading_lines=(),
                    trailing_whitespace=right.trailing_whitespace,
                )
                replacement = context.module.code_for_node(line)
                start = context.span(left)[0]
                end = context.index.line_end(right_pos.end.line)
                expected = context.source[start:end]
                yield replacement_action(
                    rule="merge_adjacent_plain_imports",
                    inverse_rule="split_plain_import",
                    site=f"imports@{left_pos.start.line}-{right_pos.start.line}",
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
            if len(node.body) != 1 or not isinstance(node.body[0], cst.Import):
                continue
            if not canonical_line(node) or context.is_protected(node):
                continue
            import_node = node.body[0]
            # Merge is deliberately binary (two one-name lines), so accepting
            # larger imports here would create split actions with no immediate
            # inverse in the target state.
            if len(import_node.names) != 2:
                continue
            source = context.module.code_for_node(import_node)
            if not re.fullmatch(
                r"import [A-Za-z_]\w*(?: as [A-Za-z_]\w*)?"
                r"(?:,[A-Za-z_]\w*(?: as [A-Za-z_]\w*)?)+",
                source,
            ):
                continue
            line = context.positions[node].start.line
            indent = context.index.leading_indent(line)
            if import_node.semicolon is not cst.MaybeSentinel.DEFAULT:
                continue
            for cut in (1,):
                first = cst.SimpleStatementLine(
                    body=(
                        cst.Import(
                            names=self._canonical_aliases(
                                tuple(import_node.names[:cut])
                            )
                        ),
                    ),
                    trailing_whitespace=cst.TrailingWhitespace(
                        whitespace=cst.SimpleWhitespace(""),
                        comment=None,
                        newline=cst.Newline(),
                    ),
                )
                second = cst.SimpleStatementLine(
                    body=(
                        cst.Import(
                            names=self._canonical_aliases(
                                tuple(import_node.names[cut:])
                            )
                        ),
                    ),
                    trailing_whitespace=node.trailing_whitespace,
                )
                replacement = (
                    context.module.code_for_node(first)
                    + indent
                    + context.module.code_for_node(second)
                )
                start = context.span(node)[0]
                end = context.index.line_end(context.positions[node].end.line)
                yield replacement_action(
                    rule="split_plain_import",
                    inverse_rule="merge_adjacent_plain_imports",
                    site=f"import@{line}:cut{cut}",
                    start=start,
                    end=end,
                    expected=context.source[start:end],
                    replacement=replacement,
                    parameters=(("indent", indent), ("cut", str(cut))),
                )

    @staticmethod
    def _single_import(node: cst.BaseStatement) -> bool:
        return (
            isinstance(node, cst.SimpleStatementLine)
            and len(node.body) == 1
            and isinstance(node.body[0], cst.Import)
            and len(node.body[0].names) == 1
        )

    @staticmethod
    def _canonical_aliases(
        names: tuple[cst.ImportAlias, ...]
    ) -> tuple[cst.ImportAlias, ...]:
        result: list[cst.ImportAlias] = []
        for index, alias in enumerate(names):
            comma: cst.Comma | cst.MaybeSentinel
            if index + 1 == len(names):
                comma = cst.MaybeSentinel.DEFAULT
            else:
                comma = cst.Comma(
                    whitespace_before=cst.SimpleWhitespace(""),
                    whitespace_after=cst.SimpleWhitespace(""),
                )
            result.append(alias.with_changes(comma=comma))
        return tuple(result)

    @staticmethod
    def _canonical_single_import(
        context: AnalysisContext, node: cst.Import
    ) -> bool:
        if node.semicolon is not cst.MaybeSentinel.DEFAULT:
            return False
        return bool(
            re.fullmatch(
                r"import [A-Za-z_]\w*(?: as [A-Za-z_]\w*)?",
                context.module.code_for_node(node),
            )
        )


class RedundantPassRule:
    name = "redundant_pass"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for _, body in context.suite_bodies():
            yield from self._deletions(context, body)
            yield from self._insertions(context, body)

    def _deletions(
        self, context: AnalysisContext, body: tuple[cst.BaseStatement, ...]
    ) -> Iterable[Action]:
        if len(body) < 2:
            return
        for index, statement in enumerate(body):
            if index == 0 or not self._pass_line(statement):
                continue
            assert isinstance(statement, cst.SimpleStatementLine)
            if not canonical_line(statement) or context.is_protected(statement):
                continue
            position = context.positions[statement]
            indent = context.index.leading_indent(position.start.line)
            expected = indent + "pass\n"
            start = context.index.line_start(position.start.line)
            end = context.index.line_end(position.end.line)
            if context.source[start:end] != expected:
                continue
            yield replacement_action(
                rule="delete_redundant_pass",
                inverse_rule="insert_redundant_pass",
                site=f"pass@{position.start.line}",
                start=start,
                end=end,
                expected=expected,
                replacement="",
                parameters=(("indent", indent),),
            )

    def _insertions(
        self, context: AnalysisContext, body: tuple[cst.BaseStatement, ...]
    ) -> Iterable[Action]:
        if not body:
            return
        future_lines = {
            index
            for index, statement in enumerate(body)
            if self._future_import(statement)
        }
        for index, statement in enumerate(body):
            if context.is_protected(statement):
                continue
            if any(future > index for future in future_lines):
                continue
            position = context.positions[statement]
            line_text = context.index.line_text(position.end.line)
            if not line_text.endswith("\n"):
                continue
            indent = context.index.leading_indent(position.start.line)
            insertion = indent + "pass\n"
            offset = context.index.line_end(position.end.line)
            yield replacement_action(
                rule="insert_redundant_pass",
                inverse_rule="delete_redundant_pass",
                site=f"gap-after@{position.end.line}",
                start=offset,
                end=offset,
                expected="",
                replacement=insertion,
                parameters=(("indent", indent),),
            )

    @staticmethod
    def _pass_line(node: cst.BaseStatement) -> bool:
        return (
            isinstance(node, cst.SimpleStatementLine)
            and len(node.body) == 1
            and isinstance(node.body[0], cst.Pass)
        )

    @staticmethod
    def _future_import(node: cst.BaseStatement) -> bool:
        return (
            isinstance(node, cst.SimpleStatementLine)
            and len(node.body) == 1
            and isinstance(node.body[0], cst.ImportFrom)
            and isinstance(node.body[0].module, cst.Name)
            and node.body[0].module.value == "__future__"
        )


class SolePassZeroRule:
    name = "sole_pass_zero"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for _, body in context.suite_bodies():
            if len(body) != 1 or not isinstance(body[0], cst.SimpleStatementLine):
                continue
            line = body[0]
            if not canonical_line(line) or len(line.body) != 1 or context.is_protected(line):
                continue
            small = line.body[0]
            start, end = context.span(small)
            if isinstance(small, cst.Pass) and context.source[start:end] == "pass":
                yield replacement_action(
                    rule="sole_pass_to_zero",
                    inverse_rule="sole_zero_to_pass",
                    site=f"suite@{context.positions[line].start.line}",
                    start=start,
                    end=end,
                    expected="pass",
                    replacement="0",
                )
            elif (
                isinstance(small, cst.Expr)
                and isinstance(small.value, cst.Integer)
                and context.module.code_for_node(small.value) == "0"
            ):
                yield replacement_action(
                    rule="sole_zero_to_pass",
                    inverse_rule="sole_pass_to_zero",
                    site=f"suite@{context.positions[line].start.line}",
                    start=start,
                    end=end,
                    expected="0",
                    replacement="pass",
                )


class ReturnNoneRule:
    name = "return_none"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for node in iter_nodes(context.module):
            if not isinstance(node, cst.SimpleStatementLine):
                continue
            if len(node.body) != 1 or not isinstance(node.body[0], cst.Return):
                continue
            if not canonical_line(node) or context.is_protected(node):
                continue
            statement = node.body[0]
            start, end = context.span(statement)
            text = context.source[start:end]
            if (
                isinstance(statement.value, cst.Name)
                and statement.value.value == "None"
                and text == "return None"
            ):
                yield replacement_action(
                    rule="return_none_to_bare",
                    inverse_rule="bare_return_to_none",
                    site=f"return@{context.positions[node].start.line}",
                    start=start,
                    end=end,
                    expected="return None",
                    replacement="return",
                )
            elif statement.value is None and text == "return":
                yield replacement_action(
                    rule="bare_return_to_none",
                    inverse_rule="return_none_to_bare",
                    site=f"return@{context.positions[node].start.line}",
                    start=start,
                    end=end,
                    expected="return",
                    replacement="return None",
                )


class FinalBareReturnRule:
    name = "final_bare_return"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for node in iter_nodes(context.module):
            if not isinstance(node, cst.FunctionDef):
                continue
            if not isinstance(node.body, cst.IndentedBlock) or node.body.footer:
                continue
            body = tuple(node.body.body)
            if not body or context.has_reflection(node):
                continue
            if len(body) >= 2 and self._bare_return_line(body[-1]):
                line = body[-1]
                assert isinstance(line, cst.SimpleStatementLine)
                if canonical_line(line) and not context.is_protected(line):
                    position = context.positions[line]
                    indent = context.index.leading_indent(position.start.line)
                    start = context.index.line_start(position.start.line)
                    end = context.index.line_end(position.end.line)
                    expected = indent + "return\n"
                    if context.source[start:end] == expected:
                        yield replacement_action(
                            rule="delete_final_bare_return",
                            inverse_rule="append_final_bare_return",
                            site=f"function:{node.name.value}",
                            start=start,
                            end=end,
                            expected=expected,
                            replacement="",
                            parameters=(("indent", indent),),
                        )
            if not self._bare_return_line(body[-1]):
                last = body[-1]
                position = context.positions[last]
                if context.index.line_text(position.end.line).endswith("\n"):
                    indent = context.index.leading_indent(context.positions[body[0]].start.line)
                    offset = context.index.line_end(position.end.line)
                    yield replacement_action(
                        rule="append_final_bare_return",
                        inverse_rule="delete_final_bare_return",
                        site=f"function:{node.name.value}",
                        start=offset,
                        end=offset,
                        expected="",
                        replacement=indent + "return\n",
                        parameters=(("indent", indent),),
                    )

    @staticmethod
    def _bare_return_line(node: cst.BaseStatement) -> bool:
        return (
            isinstance(node, cst.SimpleStatementLine)
            and len(node.body) == 1
            and isinstance(node.body[0], cst.Return)
            and node.body[0].value is None
        )


class ObjectBaseRule:
    name = "object_base"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        assigned = context.assigned_names()
        if "object" in assigned or "*" in assigned:
            return
        for node in iter_nodes(context.module):
            if not isinstance(node, cst.ClassDef) or context.is_protected(node):
                continue
            if node.keywords:
                continue
            if len(node.bases) == 1:
                base = node.bases[0]
                if not isinstance(base.value, cst.Name) or base.value.value != "object":
                    continue
                if not isinstance(node.lpar, cst.LeftParen) or not isinstance(
                    node.rpar, cst.RightParen
                ):
                    continue
                start = context.span(node.lpar)[0]
                end = context.span(node.rpar)[1]
                if context.source[start:end] != "(object)":
                    continue
                yield replacement_action(
                    rule="delete_builtin_object_base",
                    inverse_rule="insert_builtin_object_base",
                    site=f"class:{node.name.value}",
                    start=start,
                    end=end,
                    expected="(object)",
                    replacement="",
                )
            elif not node.bases and not node.keywords:
                if node.lpar is not cst.MaybeSentinel.DEFAULT:
                    continue
                if node.type_parameters is not None:
                    continue
                name_end = context.span(node.name)[1]
                line_end = context.index.line_end(context.positions[node].start.line)
                colon_start = context.source.find(":", name_end, line_end)
                if colon_start < 0 or context.source[name_end:colon_start] != "":
                    continue
                yield replacement_action(
                    rule="insert_builtin_object_base",
                    inverse_rule="delete_builtin_object_base",
                    site=f"class:{node.name.value}",
                    start=colon_start,
                    end=colon_start,
                    expected="",
                    replacement="(object)",
                )


BUILTIN_EXCEPTIONS = frozenset(
    {
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "BaseException",
        "BufferError",
        "EOFError",
        "Exception",
        "ImportError",
        "IndexError",
        "KeyError",
        "LookupError",
        "MemoryError",
        "NameError",
        "NotImplementedError",
        "OSError",
        "OverflowError",
        "RuntimeError",
        "StopIteration",
        "SyntaxError",
        "SystemError",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
    }
)


class BuiltinRaiseParenthesesRule:
    name = "builtin_raise_parentheses"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        shadowed = context.assigned_names()
        if "*" in shadowed:
            return
        for node in iter_nodes(context.module):
            if not isinstance(node, cst.Raise) or context.is_protected(node):
                continue
            expression = node.exc
            if isinstance(expression, cst.Call):
                if expression.args or not isinstance(expression.func, cst.Name):
                    continue
                name = expression.func.value
                if name not in BUILTIN_EXCEPTIONS or name in shadowed:
                    continue
                start, end = context.span(expression)
                expected = name + "()"
                if context.source[start:end] == expected:
                    yield replacement_action(
                        rule="delete_builtin_raise_parentheses",
                        inverse_rule="insert_builtin_raise_parentheses",
                        site=f"raise@{context.positions[node].start.line}",
                        start=start,
                        end=end,
                        expected=expected,
                        replacement=name,
                    )
            elif isinstance(expression, cst.Name):
                name = expression.value
                if name not in BUILTIN_EXCEPTIONS or name in shadowed:
                    continue
                start, end = context.span(expression)
                if context.source[start:end] == name:
                    yield replacement_action(
                        rule="insert_builtin_raise_parentheses",
                        inverse_rule="delete_builtin_raise_parentheses",
                        site=f"raise@{context.positions[node].start.line}",
                        start=start,
                        end=end,
                        expected=name,
                        replacement=name + "()",
                    )


SAFE_INTEGER_PARENTS = (cst.Assign, cst.AnnAssign, cst.Return, cst.Arg)


class IntegerTemplateRule:
    name = "integer_template"
    ADD_RE = re.compile(r"\(\((\d+)\+(\d+)\)-(\d+)\)")
    XOR_RE = re.compile(r"\(\((\d+)\^(\d+)\)\^(\d+)\)")

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for node in iter_nodes(context.module):
            if not isinstance(node, cst.BaseExpression) or context.is_protected(node):
                continue
            parent = context.parents.get(node)
            if not isinstance(parent, SAFE_INTEGER_PARENTS):
                continue
            if isinstance(node, cst.Integer):
                text = context.text(node)
                if not re.fullmatch(r"(?:0|[1-9]\d*)", text):
                    continue
                start, end = context.span(node)
                for key in runtime.config.integer_template_keys:
                    for template_name, replacement in (
                        ("add_sub", f"(({text}+{key})-{key})"),
                        ("xor", f"(({text}^{key})^{key})"),
                    ):
                        yield replacement_action(
                            rule=f"expand_integer_{template_name}",
                            inverse_rule=f"fold_integer_{template_name}",
                            site=f"integer@{context.positions[node].start.line}:{context.positions[node].start.column}",
                            start=start,
                            end=end,
                            expected=text,
                            replacement=replacement,
                            parameters=(("key", str(key)),),
                        )
            else:
                lpar = getattr(node, "lpar", ())
                rpar = getattr(node, "rpar", ())
                if len(lpar) != 1 or len(rpar) != 1:
                    continue
                start = context.span(lpar[0])[0]
                end = context.span(rpar[0])[1]
                text = context.source[start:end]
                for template_name, pattern in (
                    ("add_sub", self.ADD_RE),
                    ("xor", self.XOR_RE),
                ):
                    match = pattern.fullmatch(text)
                    if not match or match.group(2) != match.group(3):
                        continue
                    yield replacement_action(
                        rule=f"fold_integer_{template_name}",
                        inverse_rule=f"expand_integer_{template_name}",
                        site=f"integer-template@{context.positions[node].start.line}:{context.positions[node].start.column}",
                        start=start,
                        end=end,
                        expected=text,
                        replacement=match.group(1),
                        parameters=(("key", match.group(2)),),
                    )
