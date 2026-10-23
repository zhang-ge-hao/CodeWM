"""Binding-aware, naturally reversible local-variable renaming."""

from __future__ import annotations

from collections import defaultdict
import keyword
from typing import Iterable

import libcst as cst
from libcst.metadata.scope_provider import Assignment, FunctionScope

from ..context import AnalysisContext, iter_nodes
from ..model import Action, RuntimeState, TextEdit


FORBIDDEN_ANCESTORS = (
    cst.CompFor,
    cst.ExceptHandler,
    cst.Global,
    cst.Nonlocal,
)


class VariableRenameRule:
    """Rename one local variable definition and every resolved reference.

    The fixed run-level name pool contains both original and generated names.
    Immediately after ``old -> new``, ``old`` is free again, so ``new -> old``
    is enumerated by this same rule without retaining transition history.
    """

    name = "rename_variable"

    def enumerate_actions(
        self, context: AnalysisContext, runtime: RuntimeState
    ) -> Iterable[Action]:
        seen_scopes: set[int] = set()
        for scope in context.scopes.values():
            if not isinstance(scope, FunctionScope) or id(scope) in seen_scopes:
                continue
            seen_scopes.add(id(scope))
            yield from self._scope_actions(context, runtime, scope)

    def _scope_actions(
        self,
        context: AnalysisContext,
        runtime: RuntimeState,
        scope: FunctionScope,
    ) -> Iterable[Action]:
        grouped: dict[str, list[Assignment]] = defaultdict(list)
        for assignment in scope.assignments:
            grouped[assignment.name].append(assignment)

        all_scope_names = {assignment.name for assignment in scope.assignments}

        for old_name, assignments in sorted(grouped.items()):
            if not self._eligible_group(
                context, runtime, scope, old_name, assignments
            ):
                continue

            nodes: set[cst.Name] = set()
            for assignment in assignments:
                assert isinstance(assignment.node, cst.Name)
                nodes.add(assignment.node)
                for access in assignment.references:
                    assert isinstance(access.node, cst.Name)
                    nodes.add(access.node)

            ordered_nodes = sorted(
                nodes,
                key=lambda node: (
                    context.positions[node].start.line,
                    context.positions[node].start.column,
                ),
            )
            if not ordered_nodes:
                continue

            function = context.enclosing_function(ordered_nodes[0])
            if function is None or context.has_reflection(function):
                continue

            # A target that occurs anywhere in this function, including a
            # descendant scope, could capture or be captured by a nested free
            # reference.  Conservatively reserve every syntactic Name in the
            # whole function subtree.  After old->new, ``old`` is absent and
            # therefore remains available for the immediate inverse.
            reserved_names = all_scope_names | {
                node.value
                for node in iter_nodes(function)
                if isinstance(node, cst.Name)
            }

            for new_name in runtime.variable_targets(old_name):
                if not self._eligible_target(new_name, old_name, reserved_names):
                    continue
                edits: list[TextEdit] = []
                for node in ordered_nodes:
                    start, end = context.span(node)
                    edits.append(
                        TextEdit(
                            start=start,
                            end=end,
                            expected=old_name,
                            replacement=new_name,
                        )
                    )
                line = context.positions[ordered_nodes[0]].start.line
                yield Action(
                    rule=self.name,
                    inverse_rule=self.name,
                    site=f"{function.name.value}:{old_name}@{line}",
                    edits=tuple(edits),
                    parameters=(("old", old_name), ("new", new_name)),
                )

    def _eligible_group(
        self,
        context: AnalysisContext,
        runtime: RuntimeState,
        scope: FunctionScope,
        name: str,
        assignments: list[Assignment],
    ) -> bool:
        if (
            not name.isidentifier()
            or keyword.iskeyword(name)
            or (name.startswith("__") and name.endswith("__"))
        ):
            return False

        helper_binding = (
            name in runtime.helper_names
            or runtime.variable_pool_root(name) in runtime.helper_names
        )

        for assignment in assignments:
            if not isinstance(assignment.node, cst.Name):
                # Parameters, function/class definitions, and imports are not
                # renamed by the conservative prototype.
                return False
            if context.is_protected(assignment.node) and not helper_binding:
                return False
            if any(
                isinstance(parent, FORBIDDEN_ANCESTORS)
                for parent in context.ancestors(assignment.node)
            ):
                return False
            for access in assignment.references:
                if access.scope is not scope or not isinstance(access.node, cst.Name):
                    # A closure reference would require coordinated nested-scope
                    # analysis; leave it untouched in the first prototype.
                    return False
                if context.is_protected(access.node) and not helper_binding:
                    return False
        return True

    @staticmethod
    def _eligible_target(
        target: str, old_name: str, all_scope_names: set[str]
    ) -> bool:
        return (
            target != old_name
            and target.isidentifier()
            and not keyword.iskeyword(target)
            and target not in all_scope_names
        )
