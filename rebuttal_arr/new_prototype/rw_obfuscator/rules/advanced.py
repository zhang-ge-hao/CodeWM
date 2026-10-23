"""Canonical opaque predicates and bounded dispatcher flattening."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Iterator

import libcst as cst

from ..context import AnalysisContext, canonical_line, iter_nodes
from ..model import Action, RuntimeState
from .base import replacement_action


FORBIDDEN_PAYLOADS = (
    cst.Return,
    cst.Raise,
    cst.Break,
    cst.Continue,
    cst.Yield,
    cst.Global,
    cst.Nonlocal,
)


def _canonical_small_text(
    context: AnalysisContext, statement: cst.BaseStatement
) -> str | None:
    if not isinstance(statement, cst.SimpleStatementLine):
        return None
    if len(statement.body) != 1 or not canonical_line(statement):
        return None
    if isinstance(statement.body[0], cst.Expr) and isinstance(
        statement.body[0].value, (cst.SimpleString, cst.ConcatenatedString)
    ):
        # Moving the first string expression would change __doc__.
        return None
    text = context.module.code_for_node(statement)
    if (
        not text.endswith("\n")
        or "\n" in text[:-1]
        or "#" in text
        or "\r" in text
    ):
        return None
    return text[:-1]


def _simple_payload(
    context: AnalysisContext, statement: cst.BaseStatement
) -> str | None:
    text = _canonical_small_text(context, statement)
    if text is None:
        return None
    assert isinstance(statement, cst.SimpleStatementLine)
    small = statement.body[0]
    if isinstance(small, FORBIDDEN_PAYLOADS) or (
        isinstance(small, cst.Expr) and isinstance(small.value, cst.Yield)
    ):
        return None
    return text


def _zero_line(node: cst.BaseStatement) -> bool:
    return (
        isinstance(node, cst.SimpleStatementLine)
        and len(node.body) == 1
        and isinstance(node.body[0], cst.Expr)
        and isinstance(node.body[0].value, cst.Integer)
        and node.body[0].value.value == "0"
        and canonical_line(node)
    )


class OpaquePredicateRule:
    name = "opaque_predicate"
    TEST_RE = re.compile(r"\((\d+)\*(\d+)\)%2==([01])")

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        yield from self._wrap_actions(context, runtime)
        yield from self._unwrap_actions(context, runtime)

    def _wrap_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for owner, body in context.suite_bodies():
            if not isinstance(owner, cst.IndentedBlock):
                continue
            for statement in body:
                payload = _simple_payload(context, statement)
                if payload is None or context.is_protected(statement):
                    continue
                function = context.enclosing_function(statement)
                if function is None or context.has_reflection(function):
                    continue
                position = context.positions[statement]
                indent = context.index.leading_indent(position.start.line)
                start = context.index.line_start(position.start.line)
                end = context.index.line_end(position.end.line)
                expected = context.source[start:end]
                if expected != indent + payload + "\n":
                    continue
                for value in runtime.config.opaque_constants:
                    true_wrapper = self._render(
                        indent=indent,
                        payload=payload,
                        value=value,
                        live_then=True,
                    )
                    false_wrapper = self._render(
                        indent=indent,
                        payload=payload,
                        value=value,
                        live_then=False,
                    )
                    yield replacement_action(
                        rule="insert_true_opaque_guard",
                        inverse_rule="remove_true_opaque_guard",
                        site=f"statement@{position.start.line}",
                        start=start,
                        end=end,
                        expected=expected,
                        replacement=true_wrapper,
                        parameters=(("constant", str(value)),),
                    )
                    yield replacement_action(
                        rule="insert_false_opaque_guard",
                        inverse_rule="remove_false_opaque_guard",
                        site=f"statement@{position.start.line}",
                        start=start,
                        end=end,
                        expected=expected,
                        replacement=false_wrapper,
                        parameters=(("constant", str(value)),),
                    )

    def _unwrap_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        allowed = set(runtime.config.opaque_constants)
        for node in iter_nodes(context.module):
            if not isinstance(node, cst.If) or not isinstance(node.orelse, cst.Else):
                continue
            if not isinstance(node.body, cst.IndentedBlock) or not isinstance(
                node.orelse.body, cst.IndentedBlock
            ):
                continue
            if len(node.body.body) != 1 or len(node.orelse.body.body) != 1:
                continue
            test = context.module.code_for_node(node.test)
            match = self.TEST_RE.fullmatch(test)
            if not match:
                continue
            value, successor, result = map(int, match.groups())
            if value not in allowed or successor != value + 1:
                continue
            then_statement = node.body.body[0]
            else_statement = node.orelse.body.body[0]
            if result == 0 and _zero_line(else_statement):
                payload_statement = then_statement
                rule = "remove_true_opaque_guard"
                inverse = "insert_true_opaque_guard"
            elif result == 1 and _zero_line(then_statement):
                payload_statement = else_statement
                rule = "remove_false_opaque_guard"
                inverse = "insert_false_opaque_guard"
            else:
                continue
            payload = _simple_payload(context, payload_statement)
            if payload is None:
                continue
            position = context.positions[node]
            indent = context.index.leading_indent(position.start.line)
            start = context.index.line_start(position.start.line)
            end = context.index.line_end(position.end.line)
            expected = context.source[start:end]
            canonical = self._render(
                indent=indent,
                payload=payload,
                value=value,
                live_then=result == 0,
            )
            if expected != canonical:
                continue
            yield replacement_action(
                rule=rule,
                inverse_rule=inverse,
                site=f"opaque@{position.start.line}",
                start=start,
                end=end,
                expected=expected,
                replacement=indent + payload + "\n",
                parameters=(("constant", str(value)),),
            )

    @staticmethod
    def _render(*, indent: str, payload: str, value: int, live_then: bool) -> str:
        truth = 0 if live_then else 1
        then_payload = payload if live_then else "0"
        else_payload = "0" if live_then else payload
        return (
            f"{indent}if ({value}*{value + 1})%2=={truth}:\n"
            f"{indent}    {then_payload}\n"
            f"{indent}else:\n"
            f"{indent}    {else_payload}\n"
        )


@dataclass(frozen=True)
class _ParsedArm:
    label: int
    payload: str | None
    next_label: int | None
    condition: str | None = None
    true_label: int | None = None
    false_label: int | None = None
    is_exit: bool = False


@dataclass(frozen=True)
class _ParsedDispatcher:
    pc: str
    start_label: int
    arms: tuple[_ParsedArm, ...]
    start: int
    end: int
    indent: str
    source: str


class DispatcherRule:
    name = "dispatcher"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        yield from self._flatten_sequences(context, runtime)
        yield from self._flatten_conditionals(context, runtime)
        yield from self._restore_actions(context, runtime)

    def _flatten_sequences(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for owner, body in context.suite_bodies():
            if not isinstance(owner, cst.IndentedBlock):
                continue
            for count, labels, order in runtime.config.sequential_dispatchers:
                for start_index in range(0, len(body) - count + 1):
                    region = body[start_index : start_index + count]
                    payloads = [_simple_payload(context, item) for item in region]
                    if any(payload is None for payload in payloads):
                        continue
                    if any(context.is_protected(item) for item in region):
                        continue
                    first = region[0]
                    function = context.enclosing_function(first)
                    if function is None or context.has_reflection(function):
                        continue
                    positions = [context.positions[item] for item in region]
                    if any(
                        right.start.line != left.start.line + 1
                        for left, right in zip(positions, positions[1:])
                    ):
                        continue
                    indent = context.index.leading_indent(positions[0].start.line)
                    if any(
                        context.index.leading_indent(position.start.line) != indent
                        for position in positions
                    ):
                        continue
                    start = context.index.line_start(positions[0].start.line)
                    end = context.index.line_end(positions[-1].end.line)
                    expected = context.source[start:end]
                    canonical_region = "".join(
                        indent + payload + "\n"
                        for payload in payloads
                        if payload is not None
                    )
                    if expected != canonical_region:
                        continue
                    for pc in runtime.helper_names:
                        if re.search(rf"\b{re.escape(pc)}\b", context.source):
                            continue
                        replacement = self._render_sequence(
                            indent,
                            pc,
                            tuple(payload for payload in payloads if payload is not None),
                            labels,
                            order,
                        )
                        yield replacement_action(
                            rule="flatten_straight_line",
                            inverse_rule="restore_straight_line",
                            site=f"region@{positions[0].start.line}:{count}",
                            start=start,
                            end=end,
                            expected=expected,
                            replacement=replacement,
                            parameters=(
                                ("pc", pc),
                                ("labels", ",".join(map(str, labels))),
                                ("order", ",".join(map(str, order))),
                            ),
                        )

    def _flatten_conditionals(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for node in iter_nodes(context.module):
            if not isinstance(node, cst.If) or not isinstance(node.orelse, cst.Else):
                continue
            if context.is_protected(node) or not isinstance(node.body, cst.IndentedBlock):
                continue
            if not isinstance(node.orelse.body, cst.IndentedBlock):
                continue
            if len(node.body.body) != 1 or len(node.orelse.body.body) != 1:
                continue
            true_payload = _simple_payload(context, node.body.body[0])
            false_payload = _simple_payload(context, node.orelse.body.body[0])
            if true_payload is None or false_payload is None:
                continue
            condition = context.module.code_for_node(node.test)
            if any(piece in condition for piece in ("\n", "#", ":=", "lambda", "yield")):
                continue
            function = context.enclosing_function(node)
            if function is None or context.has_reflection(function):
                continue
            position = context.positions[node]
            indent = context.index.leading_indent(position.start.line)
            start = context.index.line_start(position.start.line)
            end = context.index.line_end(position.end.line)
            expected = context.source[start:end]
            canonical = (
                f"{indent}if {condition}:\n"
                f"{indent}    {true_payload}\n"
                f"{indent}else:\n"
                f"{indent}    {false_payload}\n"
            )
            if expected != canonical:
                continue
            for labels, order in runtime.config.conditional_dispatchers:
                for pc in runtime.helper_names:
                    if re.search(rf"\b{re.escape(pc)}\b", context.source):
                        continue
                    replacement = self._render_conditional(
                        indent,
                        pc,
                        condition,
                        true_payload,
                        false_payload,
                        labels,
                        order,
                    )
                    yield replacement_action(
                        rule="flatten_simple_if",
                        inverse_rule="restore_simple_if",
                        site=f"if@{position.start.line}",
                        start=start,
                        end=end,
                        expected=expected,
                        replacement=replacement,
                        parameters=(
                            ("pc", pc),
                            ("labels", ",".join(map(str, labels))),
                            ("order", ",".join(map(str, order))),
                        ),
                    )

    def _restore_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        for parsed in self._parse_dispatchers(context, runtime):
            # A dispatcher helper may itself walk through its private rename
            # pool.  It must return to the exact helper name selected by the
            # flatten action before the dispatcher can be removed; otherwise
            # restore -> flatten would not have an exact one-step inverse.
            if parsed.pc not in runtime.helper_names:
                continue
            sequential = self._restore_sequence(parsed, runtime)
            if sequential is not None:
                replacement, labels, order, payloads = sequential
                if parsed.source != self._render_sequence(
                    parsed.indent,
                    parsed.pc,
                    payloads,
                    labels,
                    order,
                ):
                    continue
                yield replacement_action(
                    rule="restore_straight_line",
                    inverse_rule="flatten_straight_line",
                    site=f"dispatcher@{parsed.start}",
                    start=parsed.start,
                    end=parsed.end,
                    expected=parsed.source,
                    replacement=replacement,
                    parameters=(
                        ("pc", parsed.pc),
                        ("labels", ",".join(map(str, labels))),
                        ("order", ",".join(map(str, order))),
                    ),
                )
            conditional = self._restore_conditional(parsed, runtime)
            if conditional is not None:
                (
                    replacement,
                    labels,
                    order,
                    condition,
                    true_payload,
                    false_payload,
                ) = conditional
                if parsed.source != self._render_conditional(
                    parsed.indent,
                    parsed.pc,
                    condition,
                    true_payload,
                    false_payload,
                    labels,
                    order,
                ):
                    continue
                yield replacement_action(
                    rule="restore_simple_if",
                    inverse_rule="flatten_simple_if",
                    site=f"dispatcher@{parsed.start}",
                    start=parsed.start,
                    end=parsed.end,
                    expected=parsed.source,
                    replacement=replacement,
                    parameters=(
                        ("pc", parsed.pc),
                        ("labels", ",".join(map(str, labels))),
                        ("order", ",".join(map(str, order))),
                    ),
                )

    def _parse_dispatchers(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterator[_ParsedDispatcher]:
        allowed_pc = set(runtime.dispatcher_names())
        for _, body in context.suite_bodies():
            for assignment_line, while_node in zip(body, body[1:]):
                if not isinstance(assignment_line, cst.SimpleStatementLine):
                    continue
                if len(assignment_line.body) != 1 or not isinstance(
                    assignment_line.body[0], cst.Assign
                ):
                    continue
                assignment = assignment_line.body[0]
                if len(assignment.targets) != 1:
                    continue
                target = assignment.targets[0].target
                if not isinstance(target, cst.Name) or target.value not in allowed_pc:
                    continue
                if not isinstance(assignment.value, cst.Integer):
                    continue
                if not isinstance(while_node, cst.While):
                    continue
                if not isinstance(while_node.test, cst.Name) or while_node.test.value != "True":
                    continue
                if not isinstance(while_node.body, cst.IndentedBlock):
                    continue
                if len(while_node.body.body) != 1 or not isinstance(
                    while_node.body.body[0], cst.If
                ):
                    continue
                arms = self._parse_arms(context, while_node.body.body[0], target.value)
                if arms is None:
                    continue
                start_pos = context.positions[assignment_line]
                while_pos = context.positions[while_node]
                if while_pos.start.line != start_pos.start.line + 1:
                    continue
                start = context.index.line_start(start_pos.start.line)
                end = context.index.line_end(while_pos.end.line)
                yield _ParsedDispatcher(
                    pc=target.value,
                    start_label=int(assignment.value.value),
                    arms=arms,
                    start=start,
                    end=end,
                    indent=context.index.leading_indent(start_pos.start.line),
                    source=context.source[start:end],
                )

    def _parse_arms(
        self, context: AnalysisContext, root: cst.If, pc: str
    ) -> tuple[_ParsedArm, ...] | None:
        result: list[_ParsedArm] = []
        current: cst.If | None = root
        while current is not None:
            test = context.module.code_for_node(current.test)
            match = re.fullmatch(rf"{re.escape(pc)} == (\d+)", test)
            if not match or not isinstance(current.body, cst.IndentedBlock):
                return None
            statements = tuple(current.body.body)
            if len(statements) == 2:
                first = _canonical_small_text(context, statements[0])
                second = _canonical_small_text(context, statements[1])
                if first is None or second is None:
                    return None
                if first == f"del {pc}" and second == "break":
                    arm = _ParsedArm(
                        label=int(match.group(1)),
                        payload=None,
                        next_label=None,
                        is_exit=True,
                    )
                else:
                    transition = re.fullmatch(rf"{re.escape(pc)} = (\d+)", second)
                    if not transition:
                        return None
                    arm = _ParsedArm(
                        label=int(match.group(1)),
                        payload=first,
                        next_label=int(transition.group(1)),
                    )
            elif len(statements) == 1:
                entry = _canonical_small_text(context, statements[0])
                if entry is None:
                    return None
                conditional = re.fullmatch(
                    rf"{re.escape(pc)} = (\d+) if \((.*)\) else (\d+)", entry
                )
                if not conditional:
                    return None
                arm = _ParsedArm(
                    label=int(match.group(1)),
                    payload=None,
                    next_label=None,
                    condition=conditional.group(2),
                    true_label=int(conditional.group(1)),
                    false_label=int(conditional.group(3)),
                )
            else:
                return None
            result.append(arm)
            if current.orelse is None:
                current = None
            elif isinstance(current.orelse, cst.If):
                current = current.orelse
            else:
                return None
        labels = [arm.label for arm in result]
        if len(labels) != len(set(labels)):
            return None
        return tuple(result)

    def _restore_sequence(
        self, parsed: _ParsedDispatcher, runtime: RuntimeState
    ) -> tuple[
        str,
        tuple[int, ...],
        tuple[int, ...],
        tuple[str, ...],
    ] | None:
        by_label = {arm.label: arm for arm in parsed.arms}
        logical: list[_ParsedArm] = []
        label = parsed.start_label
        seen: set[int] = set()
        while label not in seen and label in by_label:
            seen.add(label)
            arm = by_label[label]
            logical.append(arm)
            if arm.is_exit:
                break
            if arm.condition is not None or arm.payload is None or arm.next_label is None:
                return None
            label = arm.next_label
        if not logical or not logical[-1].is_exit or len(seen) != len(parsed.arms):
            return None
        payloads = tuple(
            arm.payload for arm in logical[:-1] if arm.payload is not None
        )
        labels = tuple(arm.label for arm in logical)
        textual_labels = tuple(arm.label for arm in parsed.arms)
        order = tuple(labels.index(label) for label in textual_labels)
        spec = (len(payloads), labels, order)
        if spec not in runtime.config.sequential_dispatchers:
            return None
        replacement = "".join(
            parsed.indent + payload + "\n"
            for payload in payloads
            if payload is not None
        )
        return replacement, labels, order, payloads

    def _restore_conditional(
        self, parsed: _ParsedDispatcher, runtime: RuntimeState
    ) -> tuple[
        str,
        tuple[int, ...],
        tuple[int, ...],
        str,
        str,
        str,
    ] | None:
        by_label = {arm.label: arm for arm in parsed.arms}
        entry = by_label.get(parsed.start_label)
        if entry is None or entry.condition is None:
            return None
        true_arm = by_label.get(entry.true_label)
        false_arm = by_label.get(entry.false_label)
        if true_arm is None or false_arm is None:
            return None
        if (
            true_arm.payload is None
            or false_arm.payload is None
            or true_arm.next_label is None
            or true_arm.next_label != false_arm.next_label
        ):
            return None
        exit_arm = by_label.get(true_arm.next_label)
        if exit_arm is None or not exit_arm.is_exit or len(by_label) != 4:
            return None
        labels = (entry.label, true_arm.label, false_arm.label, exit_arm.label)
        textual_labels = tuple(arm.label for arm in parsed.arms)
        order = tuple(labels.index(label) for label in textual_labels)
        if (labels, order) not in runtime.config.conditional_dispatchers:
            return None
        replacement = (
            f"{parsed.indent}if {entry.condition}:\n"
            f"{parsed.indent}    {true_arm.payload}\n"
            f"{parsed.indent}else:\n"
            f"{parsed.indent}    {false_arm.payload}\n"
        )
        return (
            replacement,
            labels,
            order,
            entry.condition,
            true_arm.payload,
            false_arm.payload,
        )

    @staticmethod
    def _render_sequence(
        indent: str,
        pc: str,
        payloads: tuple[str, ...],
        labels: tuple[int, ...],
        order: tuple[int, ...],
    ) -> str:
        lines = [f"{indent}{pc} = {labels[0]}\n", f"{indent}while True:\n"]
        for text_index, logical_index in enumerate(order):
            keyword = "if" if text_index == 0 else "elif"
            lines.append(f"{indent}    {keyword} {pc} == {labels[logical_index]}:\n")
            if logical_index == len(payloads):
                lines.append(f"{indent}        del {pc}\n")
                lines.append(f"{indent}        break\n")
            else:
                lines.append(f"{indent}        {payloads[logical_index]}\n")
                lines.append(
                    f"{indent}        {pc} = {labels[logical_index + 1]}\n"
                )
        return "".join(lines)

    @staticmethod
    def _render_conditional(
        indent: str,
        pc: str,
        condition: str,
        true_payload: str,
        false_payload: str,
        labels: tuple[int, int, int, int],
        order: tuple[int, ...],
    ) -> str:
        entry, true_label, false_label, exit_label = labels
        lines = [f"{indent}{pc} = {entry}\n", f"{indent}while True:\n"]
        for text_index, logical_index in enumerate(order):
            keyword = "if" if text_index == 0 else "elif"
            lines.append(f"{indent}    {keyword} {pc} == {labels[logical_index]}:\n")
            if logical_index == 0:
                lines.append(
                    f"{indent}        {pc} = {true_label} if ({condition}) else {false_label}\n"
                )
            elif logical_index == 1:
                lines.append(f"{indent}        {true_payload}\n")
                lines.append(f"{indent}        {pc} = {exit_label}\n")
            elif logical_index == 2:
                lines.append(f"{indent}        {false_payload}\n")
                lines.append(f"{indent}        {pc} = {exit_label}\n")
            else:
                lines.append(f"{indent}        del {pc}\n")
                lines.append(f"{indent}        break\n")
        return "".join(lines)
