"""Core immutable data structures for the random-walk obfuscator."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import string
from typing import Iterable, Mapping


@dataclass(frozen=True, order=True)
class TextEdit:
    """One exact source edit, using Python string (code-point) offsets."""

    start: int
    end: int
    expected: str
    replacement: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid edit range [{self.start}, {self.end})")

    @property
    def byte_delta(self) -> int:
        return len(self.replacement.encode("utf-8")) - len(
            self.expected.encode("utf-8")
        )


@dataclass(frozen=True)
class Action:
    """A concrete, uniformly sampled action at a particular source site."""

    rule: str
    site: str
    edits: tuple[TextEdit, ...] = ()
    inverse_rule: str | None = None
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        ordered = sorted(self.edits, key=lambda edit: (edit.start, edit.end))
        for left, right in zip(ordered, ordered[1:]):
            if left.end > right.start:
                raise ValueError(
                    f"overlapping edits in {self.rule}: {left!r} and {right!r}"
                )

    @property
    def is_identity(self) -> bool:
        return not self.edits

    @property
    def byte_delta(self) -> int:
        return sum(edit.byte_delta for edit in self.edits)

    @property
    def key(self) -> tuple[object, ...]:
        return (self.rule, self.site, self.parameters, self.edits)

    def apply(self, source: str) -> str:
        """Apply exact edits. This does not parse, reject, or retry."""

        result = source
        for edit in sorted(self.edits, key=lambda item: item.start, reverse=True):
            actual = result[edit.start : edit.end]
            if actual != edit.expected:
                raise ValueError(
                    f"stale {self.rule} action at {self.site}: "
                    f"expected {edit.expected!r}, found {actual!r}"
                )
            result = result[: edit.start] + edit.replacement + result[edit.end :]
        return result


IDENTITY = Action(rule="identity", site="program", inverse_rule="identity")


@dataclass(frozen=True)
class LengthBucket:
    lower: int
    upper: int

    @classmethod
    def for_source(cls, source: str, width: int = 2000) -> "LengthBucket":
        if width <= 0:
            raise ValueError("bucket width must be positive")
        size = len(source.encode("utf-8"))
        lower = (size // width) * width
        return cls(lower=lower, upper=lower + width)

    def contains(self, byte_length: int) -> bool:
        return self.lower <= byte_length < self.upper


DEFAULT_GENERATED_NAMES = tuple(
    [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    + [chr(code) for code in range(ord("a"), ord("z") + 1)]
    + [f"O{i}" for i in range(16)]
    + [f"lI{i}" for i in range(16)]
)


@dataclass(frozen=True)
class ObfuscatorConfig:
    bucket_width: int = 2000
    generated_names: tuple[str, ...] = DEFAULT_GENERATED_NAMES
    integer_template_keys: tuple[int, ...] = (1, 2, 3, 5, 7)
    opaque_constants: tuple[int, ...] = (17, 31, 47, 73)
    dispatcher_names: tuple[str, ...] = tuple(
        f"__rw_pc_{index}" for index in range(4)
    )
    replacement_pool_size: int = 10
    random_alias_length: int = 16
    helper_name_count: int = 20
    # (payload count, logical labels including exit, textual arm permutation)
    sequential_dispatchers: tuple[
        tuple[int, tuple[int, ...], tuple[int, ...]], ...
    ] = (
        (2, (11, 23, 37), (1, 2, 0)),
        (2, (13, 29, 43), (2, 0, 1)),
        (3, (5, 17, 31, 47), (2, 0, 3, 1)),
        (3, (7, 19, 37, 53), (3, 1, 0, 2)),
        (4, (3, 13, 27, 41, 59), (2, 4, 0, 3, 1)),
        (4, (9, 21, 35, 49, 63), (4, 1, 3, 0, 2)),
    )
    # (entry, true, false, exit, textual arm permutation)
    conditional_dispatchers: tuple[
        tuple[tuple[int, int, int, int], tuple[int, ...]], ...
    ] = (
        ((3, 17, 29, 43), (2, 0, 3, 1)),
        ((7, 19, 37, 53), (3, 1, 0, 2)),
    )
    validate_selected_output: bool = False
    enabled_rules: frozenset[str] | None = None


@dataclass
class RuntimeState:
    bucket: LengthBucket
    config: ObfuscatorConfig
    pool_rng: random.Random = field(repr=False)
    helper_names: tuple[str, ...]
    reserved_identifiers: set[str] = field(default_factory=set)
    variable_pools: dict[str, tuple[str, ...]] = field(default_factory=dict)
    variable_member_to_root: dict[str, str] = field(default_factory=dict)
    comment_pools: dict[str, tuple[str, ...]] = field(default_factory=dict)
    comment_member_to_root: dict[str, str] = field(default_factory=dict)
    block_comment_pools: dict[str, tuple[str, ...]] = field(default_factory=dict)
    block_comment_member_to_root: dict[str, str] = field(default_factory=dict)
    string_pools: dict[str, tuple[str, ...]] = field(default_factory=dict)
    string_member_to_root: dict[str, str] = field(default_factory=dict)
    used_comment_aliases: set[str] = field(default_factory=set)

    def _random_letters(self) -> str:
        alphabet = string.ascii_letters
        length = self.config.random_alias_length
        return "".join(self.pool_rng.choice(alphabet) for _ in range(length))

    def ensure_variable_pool(self, name: str) -> tuple[str, ...]:
        root = self.variable_member_to_root.get(name)
        if root is not None:
            return self.variable_pools[root]
        members = [name]
        unavailable = self.reserved_identifiers | set(self.variable_member_to_root)
        while len(members) < self.config.replacement_pool_size:
            candidate = self._random_letters()
            if candidate in unavailable or candidate in members:
                continue
            members.append(candidate)
        pool = tuple(members)
        self.variable_pools[name] = pool
        for member in pool:
            self.variable_member_to_root[member] = name
        self.reserved_identifiers.update(pool)
        return pool

    def variable_pool(self, name: str) -> tuple[str, ...] | None:
        root = self.variable_member_to_root.get(name)
        return None if root is None else self.variable_pools[root]

    def variable_pool_root(self, name: str) -> str | None:
        return self.variable_member_to_root.get(name)

    def variable_targets(self, name: str) -> tuple[str, ...]:
        root = self.variable_member_to_root.get(name)
        if root is None:
            pool = self.ensure_variable_pool(name)
            root = name
        else:
            pool = self.variable_pools[root]
        return pool[1:] if name == root else (root,)

    def new_comment_alias(self) -> str:
        while True:
            alias = self._random_letters()
            if alias in self.used_comment_aliases:
                continue
            self.used_comment_aliases.add(alias)
            return alias

    def ensure_comment_pool(self, value: str) -> tuple[str, ...]:
        root = self.comment_member_to_root.get(value)
        if root is not None:
            return self.comment_pools[root]
        members = [value]
        while len(members) < self.config.replacement_pool_size:
            candidate = f"# {self.new_comment_alias()}"
            if candidate in self.comment_member_to_root or candidate in members:
                continue
            members.append(candidate)
        pool = tuple(members)
        self.comment_pools[value] = pool
        for member in pool:
            self.comment_member_to_root[member] = value
        return pool

    def comment_pool(self, value: str) -> tuple[str, ...] | None:
        root = self.comment_member_to_root.get(value)
        return None if root is None else self.comment_pools[root]

    def comment_targets(self, value: str) -> tuple[str, ...]:
        root = self.comment_member_to_root.get(value)
        if root is None:
            pool = self.ensure_comment_pool(value)
            root = value
        else:
            pool = self.comment_pools[root]
        return pool[1:] if value == root else (root,)

    def register_block_comment_pool(
        self, root: str, members: tuple[str, ...]
    ) -> tuple[str, ...]:
        existing = self.block_comment_member_to_root.get(root)
        if existing is not None:
            return self.block_comment_pools[existing]
        if len(members) != 101:
            raise ValueError("multiline-comment pool must contain one original and 100 replacements")
        if members[0] != root or len(set(members)) != len(members):
            raise ValueError("invalid multiline-comment pool")
        for member in members:
            owner = self.block_comment_member_to_root.get(member)
            if owner is not None and owner != root:
                raise ValueError("multiline-comment pools must be disjoint")
        self.block_comment_pools[root] = members
        for member in members:
            self.block_comment_member_to_root[member] = root
        return members

    def block_comment_pool(self, value: str) -> tuple[str, ...] | None:
        root = self.block_comment_member_to_root.get(value)
        return None if root is None else self.block_comment_pools[root]

    def block_comment_targets(self, value: str) -> tuple[str, ...]:
        root = self.block_comment_member_to_root.get(value)
        if root is None:
            raise KeyError(f"unregistered multiline comment: {value!r}")
        pool = self.block_comment_pools[root]
        return pool[1:] if value == root else (root,)

    def register_string_pool(
        self, root: str, members: tuple[str, ...]
    ) -> tuple[str, ...]:
        existing = self.string_member_to_root.get(root)
        if existing is not None:
            return self.string_pools[existing]
        if len(members) != self.config.replacement_pool_size:
            raise ValueError("assigned-string pool has the wrong size")
        if len(set(members)) != len(members):
            raise ValueError("assigned-string pool members must be unique")
        for member in members:
            owner = self.string_member_to_root.get(member)
            if owner is not None and owner != root:
                raise ValueError("assigned-string pool members must be disjoint")
        self.string_pools[root] = members
        for member in members:
            self.string_member_to_root[member] = root
        return members

    def string_pool(self, value: str) -> tuple[str, ...] | None:
        root = self.string_member_to_root.get(value)
        return None if root is None else self.string_pools[root]

    def dispatcher_names(self) -> frozenset[str]:
        names = set(self.helper_names)
        for helper in self.helper_names:
            pool = self.variable_pool(helper)
            if pool is not None:
                names.update(pool)
        return frozenset(names)


@dataclass(frozen=True)
class StepRecord:
    step: int
    rule: str
    site: str
    action_count: int
    byte_length_before: int
    byte_length_after: int
    parameters: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "rule": self.rule,
            "site": self.site,
            "action_count": self.action_count,
            "byte_length_before": self.byte_length_before,
            "byte_length_after": self.byte_length_after,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class WalkResult:
    source: str
    records: tuple[StepRecord, ...] = ()

    @property
    def rule_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.rule] = counts.get(record.rule, 0) + 1
        return counts
