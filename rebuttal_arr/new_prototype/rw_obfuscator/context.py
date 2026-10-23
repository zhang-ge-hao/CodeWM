"""Parse-once analysis used by all action enumerators."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import io
import re
import tokenize
from typing import Iterable, Iterator

import libcst as cst
from libcst.metadata import (
    MetadataWrapper,
    ParentNodeProvider,
    PositionProvider,
    ScopeProvider,
)
from libcst.metadata.position_provider import CodePosition, CodeRange


REFLECTION_CALL_RE = re.compile(
    r"\b(?:eval|exec|locals|globals|vars|dir|getattr|setattr|delattr)\s*\("
)
FRAME_REFLECTION_RE = re.compile(
    r"\b(?:inspect|currentframe|_getframe|f_locals|co_varnames|__code__)\b"
)


class SourceIndex:
    """Translate LibCST line/column positions into source string offsets."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.lines = source.splitlines(keepends=True)
        if not self.lines:
            self.lines.append("")
        elif source.endswith(("\n", "\r")):
            # LibCST may report EOF at the empty line following a final newline.
            self.lines.append("")
        offsets: list[int] = []
        offset = 0
        for line in self.lines:
            offsets.append(offset)
            offset += len(line)
        self.offsets = tuple(offsets)

    def offset(self, position: CodePosition) -> int:
        line_index = position.line - 1
        if line_index < 0 or line_index >= len(self.offsets):
            raise IndexError(f"line outside source: {position.line}")
        return self.offsets[line_index] + position.column

    def span(self, code_range: CodeRange) -> tuple[int, int]:
        return self.offset(code_range.start), self.offset(code_range.end)

    def line_start(self, line: int) -> int:
        return self.offsets[line - 1]

    def line_end(self, line: int) -> int:
        start = self.line_start(line)
        return start + len(self.lines[line - 1])

    def line_text(self, line: int) -> str:
        return self.lines[line - 1]

    def leading_indent(self, line: int) -> str:
        text = self.line_text(line)
        return text[: len(text) - len(text.lstrip(" \t"))]


def iter_nodes(root: cst.CSTNode) -> Iterator[cst.CSTNode]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


@dataclass(frozen=True)
class AnalysisContext:
    source: str
    module: cst.Module
    wrapper: MetadataWrapper
    positions: dict[cst.CSTNode, CodeRange]
    parents: dict[cst.CSTNode, cst.CSTNode]
    scopes: dict[cst.CSTNode, object]
    python_ast: ast.Module
    tokens: tuple[tokenize.TokenInfo, ...]
    index: SourceIndex
    protected_names: frozenset[str]

    @classmethod
    def build(
        cls,
        source: str,
        *,
        protected_names: Iterable[str] = (),
    ) -> "AnalysisContext":
        parsed = cst.parse_module(source)
        wrapper = MetadataWrapper(parsed)
        module = wrapper.module
        positions = wrapper.resolve(PositionProvider)
        parents = wrapper.resolve(ParentNodeProvider)
        scopes = wrapper.resolve(ScopeProvider)
        python_ast = ast.parse(source)
        tokens = tuple(tokenize.generate_tokens(io.StringIO(source).readline))
        return cls(
            source=source,
            module=module,
            wrapper=wrapper,
            positions=positions,
            parents=parents,
            scopes=scopes,
            python_ast=python_ast,
            tokens=tokens,
            index=SourceIndex(source),
            protected_names=frozenset(protected_names),
        )

    @property
    def byte_length(self) -> int:
        return len(self.source.encode("utf-8"))

    def span(self, node: cst.CSTNode) -> tuple[int, int]:
        return self.index.span(self.positions[node])

    def text(self, node: cst.CSTNode) -> str:
        start, end = self.span(node)
        return self.source[start:end]

    def full_line_span(self, node: cst.CSTNode) -> tuple[int, int]:
        position = self.positions[node]
        return (
            self.index.line_start(position.start.line),
            self.index.line_end(position.end.line),
        )

    def ancestors(self, node: cst.CSTNode) -> Iterator[cst.CSTNode]:
        current = node
        while current in self.parents:
            current = self.parents[current]
            yield current

    def enclosing_function(self, node: cst.CSTNode) -> cst.FunctionDef | None:
        for parent in self.ancestors(node):
            if isinstance(parent, cst.FunctionDef):
                return parent
            if isinstance(parent, (cst.ClassDef, cst.Lambda)):
                return None
        return None

    def has_reflection(self, node: cst.CSTNode | None = None) -> bool:
        text = self.source if node is None else self.text(node)
        return bool(REFLECTION_CALL_RE.search(text) or FRAME_REFLECTION_RE.search(text))

    def is_protected(self, node: cst.CSTNode) -> bool:
        """Protect generated opaque/dispatcher scaffolding from other rules."""

        for candidate in (node, *self.ancestors(node)):
            if isinstance(candidate, cst.While):
                text = self.text(candidate)
                if any(
                    re.search(rf"\b{re.escape(name)}\b", text)
                    for name in self.protected_names
                ):
                    return True
            if isinstance(candidate, cst.If):
                test = self.module.code_for_node(candidate.test)
                if re.fullmatch(r"\(\d+\*\d+\)%2==[01]", test):
                    return True
            if (
                isinstance(candidate, cst.Name)
                and candidate.value in self.protected_names
            ):
                return True
        return False

    def suite_bodies(
        self,
    ) -> Iterator[tuple[cst.CSTNode, tuple[cst.BaseStatement, ...]]]:
        yield self.module, tuple(self.module.body)
        for node in iter_nodes(self.module):
            if isinstance(node, cst.IndentedBlock):
                yield node, tuple(node.body)

    def assigned_names(self) -> set[str]:
        result: set[str] = set()
        for node in ast.walk(self.python_ast):
            if isinstance(node, ast.Name) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                result.add(node.id)
            elif isinstance(node, (ast.arg, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                result.add(node.arg if isinstance(node, ast.arg) else node.name)
            elif isinstance(node, ast.alias):
                if node.name == "*":
                    result.add("*")
                else:
                    result.add(node.asname or node.name.split(".", 1)[0])
            elif isinstance(node, ast.ExceptHandler) and node.name is not None:
                result.add(node.name)
        return result


def canonical_line(line: cst.SimpleStatementLine) -> bool:
    return (
        not line.leading_lines
        and line.trailing_whitespace.comment is None
        and line.trailing_whitespace.whitespace.value == ""
    )


def canonical_suite(suite: cst.SimpleStatementSuite) -> bool:
    return (
        suite.leading_whitespace.value == ""
        and suite.trailing_whitespace.comment is None
        and suite.trailing_whitespace.whitespace.value == ""
    )
