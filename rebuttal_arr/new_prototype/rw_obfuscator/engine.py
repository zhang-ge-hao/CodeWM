"""Strict uniform random-walk kernel."""

from __future__ import annotations

import ast
import random
from typing import Iterable, Sequence

import libcst as cst

from .context import AnalysisContext
from .model import (
    Action,
    IDENTITY,
    LengthBucket,
    ObfuscatorConfig,
    RuntimeState,
    StepRecord,
    WalkResult,
)
from .rules import default_rules


class RandomWalkObfuscator:
    """A finite reversible action walk over one initial program's bucket.

    Expensive inverse-enumerability and byte-round-trip checks deliberately do
    not run here. They are development-time properties tested in ``tests/``.
    """

    def __init__(
        self,
        initial_source: str,
        *,
        seed: int = 0,
        config: ObfuscatorConfig | None = None,
        rules: Sequence[object] | None = None,
    ) -> None:
        self.config = config or ObfuscatorConfig()
        initial_context = AnalysisContext.build(initial_source)
        initial_identifiers = {
            item.string
            for item in initial_context.tokens
            if item.string.isidentifier()
        }
        helper_rng = random.Random(f"rw-helper-names:{seed}")
        helper_names: list[str] = []
        while len(helper_names) < self.config.helper_name_count:
            suffix = "".join(
                helper_rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
                for _ in range(self.config.random_alias_length)
            )
            candidate = f"__rw_pc_{suffix}"
            if candidate in initial_identifiers or candidate in helper_names:
                continue
            helper_names.append(candidate)
        self.runtime = RuntimeState(
            bucket=LengthBucket.for_source(initial_source, self.config.bucket_width),
            config=self.config,
            pool_rng=random.Random(f"rw-replacement-pools:{seed}"),
            helper_names=tuple(helper_names),
            reserved_identifiers=set(initial_identifiers) | set(helper_names),
        )
        for name in sorted(initial_context.assigned_names()):
            if name.isidentifier():
                self.runtime.ensure_variable_pool(name)
        self.rules = tuple(default_rules() if rules is None else rules)
        self.rng = random.Random(seed)

    def enumerate_actions(self, source: str) -> tuple[Action, ...]:
        """Enumerate the flat concrete action set, including one identity."""

        context = AnalysisContext.build(
            source,
            protected_names=self.runtime.dispatcher_names(),
        )
        current_length = context.byte_length
        unique: dict[tuple[object, ...], Action] = {IDENTITY.key: IDENTITY}
        for rule in self.rules:
            for action in rule.enumerate_actions(context, self.runtime):
                if self.config.enabled_rules is not None and action.rule not in self.config.enabled_rules:
                    continue
                if action.is_identity:
                    continue
                if not self.runtime.bucket.contains(current_length + action.byte_delta):
                    continue
                unique.setdefault(action.key, action)
        # A stable ordering makes a seed reproducible without altering the flat
        # uniform distribution over concrete actions.
        return tuple(sorted(unique.values(), key=lambda action: repr(action.key)))

    def step(self, source: str, *, step_index: int = 0) -> tuple[str, StepRecord]:
        actions = self.enumerate_actions(source)
        action = self.sample_action(actions)
        result = action.apply(source)
        if self.config.validate_selected_output:
            # Debug assertion only. A failure aborts; it is never converted to a
            # self-loop or post-sampling rejection.
            ast.parse(result)
            cst.parse_module(result)
        record = StepRecord(
            step=step_index,
            rule=action.rule,
            site=action.site,
            action_count=len(actions),
            byte_length_before=len(source.encode("utf-8")),
            byte_length_after=len(result.encode("utf-8")),
            parameters=action.parameters,
        )
        return result, record

    def sample_action(self, actions: Sequence[Action]) -> Action:
        """Choose one concrete action uniformly from a flat non-empty list.

        Keeping this tiny operation explicit makes the transition kernel
        directly testable: the identity action has no special branch and is
        sampled in exactly the same way as every non-identity action.
        """

        if not actions:
            raise ValueError("cannot sample from an empty action set")
        return actions[self.rng.randrange(len(actions))]

    def walk(self, source: str, steps: int) -> WalkResult:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        current = source
        records: list[StepRecord] = []
        for index in range(steps):
            current, record = self.step(current, step_index=index)
            records.append(record)
        return WalkResult(source=current, records=tuple(records))

    def action_counts(self, source: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for action in self.enumerate_actions(source):
            counts[action.rule] = counts.get(action.rule, 0) + 1
        return counts

def obfuscate(
    source: str,
    *,
    steps: int,
    seed: int = 0,
    config: ObfuscatorConfig | None = None,
) -> WalkResult:
    engine = RandomWalkObfuscator(source, seed=seed, config=config)
    return engine.walk(source, steps)
