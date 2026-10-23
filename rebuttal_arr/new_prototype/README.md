# RW-Python-Minifier prototype

This directory contains a Python-only finite reversible random-walk
obfuscator. It is intentionally independent of the rest of the repository:
existing project code and experiment records are read-only, while every corpus
runner output and temporary execution directory is placed below `data/` here.

## Kernel

For a current source string `c`, the engine constructs one flat list:

```text
{one identity action} union {all currently applicable concrete actions}
```

It samples exactly one entry uniformly. There is no extra lazy coin, no
rule-level weighting, and no post-sampling rejection. Invalid/out-of-bucket
actions are excluded during current-state enumeration using structural
preconditions and exact byte deltas.

The initial UTF-8 source length selects a fixed half-open bucket
`[2000*x, 2000*(x+1))`. Per-variable and per-comment replacement pools are
generated deterministically from the run seed and then remain fixed for that
walk. Numeric spellings, constant templates, opaque constants, labels, and
permutations come from fixed finite catalogs.

Inverse-enumerability and byte-exact round trips are design/test properties,
not runtime kernel checks.

## Implemented rule families

- Binding-aware local-variable rename. Every original or newly discovered
  local name owns a disjoint ten-state star pool: the original spelling points
  to nine unique 16-letter mixed-case aliases, and every alias points only back
  to the original.
- A single physical `#` comment has the same directed ten-state shape.
  Generated comment bodies are direct 16-letter mixed-case aliases.
- Consecutive full-line `#` comments form one indivisible unit. Standalone
  string expressions, including function/class/module docstrings, are also
  replaced as whole literals. Each such unit has a 101-state star pool: 100
  aliases from the original and exactly one return action from an alias. The
  replacement preserves the unit's physical line count and starting position.
- Direct assigned string literals use ten value-equivalent source spellings.
  Generated spellings combine `\\UXXXXXXXX` escapes with implicit adjacent
  string-literal concatenation and introduce no runtime helper.
- Optional symbolic-token space insertion/deletion.
- Simple-statement newline/semicolon join and split.
- Canonical four-space block suite inline/expand.
- Whole-block four-space/tab indentation conversion.
- One layer of grouping parentheses in safe whole-value positions.
- Optional trailing commas in calls and container displays.
- Explicit numeric spelling pairs (`0.5/.5`, `1.0/1.`, exponent `+`, and
  canonical decimal/lowercase hexadecimal).
- Adjacent plain-import merge/split.
- Redundant `pass` insertion/deletion and sole `pass`/`0` replacement.
- `return None`/bare return and canonical final bare-return insertion/deletion.
- Sole builtin `object` base insertion/deletion.
- Empty-call parentheses for unshadowed builtin exceptions in `raise`.
- Finite integer add/sub and XOR identity templates.
- Canonical true-live and false-live opaque predicate wrappers. These use no
  temporary local variable.
- Bounded 2--4 statement dispatcher flattening with finite labels and arm
  permutations. Each concrete shape has twenty actions whose output differs
  only in a seed-derived, collision-free helper-variable name.
- Canonical one-statement-per-branch `if/else` diamond flattening.

The prototype deliberately excludes literal hoisting, comment insertion or
deletion, f-string normalization, nested/container string rewriting,
`from ... import ...` combining, annotations, positional-only argument
conversion, and arbitrary constant folding.

Generated opaque and dispatcher scaffolds use a recognisable canonical grammar
and are protected from ordinary layout/literal rules. A dispatcher helper may
walk through its own variable pool, but must return to the helper spelling used
by the flatten action before the dispatcher can be restored. Restore operations
recover opaque payloads directly or follow dispatcher state edges rather than
trusting textual arm order.

## Layout

```text
rw_obfuscator/
  context.py          LibCST, AST, scope, token, and source-position analysis
  model.py            concrete actions, text edits, buckets, and traces
  engine.py           strict uniform transition kernel
  rules/
    variable.py       variable rename
    content.py        comments, standalone strings, assigned strings
    lexical.py        spaces, parentheses, commas, numeric spellings
    structural.py     statements, suites, indentation
    pyminifier.py     reversible Python-Minifier-derived AST rules
    advanced.py       opaque predicates and dispatchers
  cli.py              single-file frontend
  corpus.py           HumanEval/MBPP differential subprocess runner
tests/
  test_model.py
  test_roundtrip.py
  test_semantics.py
data/                 all generated reports and subprocess working directories
```

## Single-file interface

From this directory:

```bash
python -m rw_obfuscator input.py --steps 50 --seed 0 --output data/output.py \
  --trace data/trace.json
```

Inspect the flat action counts at one state:

```bash
python -m rw_obfuscator input.py --list-actions
```

The public Python API is:

```python
from rw_obfuscator import RandomWalkObfuscator

engine = RandomWalkObfuscator(source, seed=0)
result = engine.walk(source, steps=50)
print(result.source)
print(result.rule_counts)
```

## Differential corpus runner

Saved model-generation records already contain `solution`, `test`,
`entry_point`, and `passed`. The runner nevertheless re-executes every baseline
in the current environment, and only obfuscates a baseline that actually
passes. Baseline and obfuscated code are executed with the same test in fresh
subprocesses with time and resource limits.

```bash
python -m rw_obfuscator.corpus \
  --require-stored-pass \
  --seeds 0,1,2 \
  --steps 1,5,10,25,50 \
  --max-cases 100 \
  --output corpus_results.jsonl
```

With no `--inputs`, the runner discovers Python `generate.jsonl` files below
the repository's read-only `data/result/`. `--include-reference` additionally
loads canonical HumanEval and MBPP records. The output argument is resolved
under this prototype's `data/` directory and cannot escape it.

The subprocess isolation is intended to contain ordinary benchmark failures,
timeouts, and crashes; it is not a security sandbox for hostile code.

## Tests

The test suite contains byte-exact two-way rule tests and multi-step semantic
checks:

```bash
pytest
```

For a pair `F/G`, round-trip tests check both that `G(F(c)) == c` and that the
inverse action is discoverable by enumerating actions from `F(c)`. These checks
are deliberately absent from production transition sampling.
