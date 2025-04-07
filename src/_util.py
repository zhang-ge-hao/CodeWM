import hashlib
import contextlib
import tempfile
import os
import json
from dataclasses import asdict


@contextlib.contextmanager
def change_dir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with change_dir(dirname):
            yield dirname


def dataclass_2_str(obj):
    return json.dumps(asdict(obj))