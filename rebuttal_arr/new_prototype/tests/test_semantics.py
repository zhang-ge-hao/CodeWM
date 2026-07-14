from __future__ import annotations

from rw_obfuscator import RandomWalkObfuscator


SOURCE = """\
def score(values):
    total = 0
    for value in values:
        if value >= 0:
            total += value
        else:
            total -= value
    return total
"""


def evaluate(source: str) -> tuple[int, int, int]:
    namespace: dict[str, object] = {}
    exec(source, namespace, namespace)
    function = namespace["score"]
    return function([]), function([1, 2, 3]), function([-1, 2, -3])


def test_multi_step_walk_preserves_example_behavior() -> None:
    expected = evaluate(SOURCE)
    for seed in range(5):
        engine = RandomWalkObfuscator(SOURCE, seed=seed)
        walked = engine.walk(SOURCE, 50)
        assert evaluate(walked.source) == expected
