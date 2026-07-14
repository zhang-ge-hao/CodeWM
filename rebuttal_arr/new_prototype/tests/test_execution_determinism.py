from __future__ import annotations

from rw_obfuscator.corpus import execute_case


SOURCE = '''\
def first_non_repeating_character(str1):
    duplicated_list = []
    unique_list = []
    for char in str1.lower():
        duplicated_list.append(char)
        unique_list = set(duplicated_list)
    for char_in_list in unique_list:
        if duplicated_list.count(char_in_list) == 1:
            return char_in_list
'''

TEST = '''\
assert first_non_repeating_character("abcabc") is None
assert first_non_repeating_character("abc") == "a"
assert first_non_repeating_character("ababc") == "c"
'''


def test_hash_dependent_execution_uses_requested_seed() -> None:
    results = [
        execute_case(SOURCE, TEST, timeout=2.0, memory_mb=256, hash_seed=10771)
        for _ in range(4)
    ]
    assert all(result.passed for result in results)
