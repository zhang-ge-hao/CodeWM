from __future__ import annotations

from rw_obfuscator.model import Action, IDENTITY, LengthBucket, TextEdit


def test_text_edit_and_identity() -> None:
    source = "x = 1\n"
    action = Action(
        rule="example",
        inverse_rule="example_inverse",
        site="x",
        edits=(TextEdit(0, 1, "x", "name"),),
    )
    assert action.apply(source) == "name = 1\n"
    assert action.byte_delta == 3
    assert IDENTITY.apply(source) == source


def test_length_bucket() -> None:
    bucket = LengthBucket.for_source("x" * 2100)
    assert (bucket.lower, bucket.upper) == (2000, 4000)
    assert bucket.contains(2000)
    assert not bucket.contains(4000)


def test_length_bucket_counts_utf8_bytes_not_code_points() -> None:
    source = "é" * 1000
    bucket = LengthBucket.for_source(source)
    assert len(source) == 1000
    assert len(source.encode("utf-8")) == 2000
    assert (bucket.lower, bucket.upper) == (2000, 4000)
