import hashlib
import contextlib
import tempfile
import os
import json
from dataclasses import asdict


def hash_str_to_int(string):
    hash_obj = hashlib.sha3_512(string.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    return hash_int & 0xFFFFFFFFFFFFFFFF


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