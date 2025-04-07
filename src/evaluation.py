import os, sys, io
import subprocess
import shutil
import uuid
import contextlib
import signal
import logging
sys.path.append("src")
from _dataclass import *
from _util import create_tempdir


def evaluate(task: GenTask|ObfTask, gen_task: GenTask):
    # WARNING
    # This program exists to execute untrusted model-generated code. Although
    # it is highly unlikely that model-generated code will do something overtly
    # malicious in response to this test suite, model-generated code may act
    # destructively due to a lack of model capability or alignment.
    # Users are strongly encouraged to sandbox this evaluation suite so that it
    # does not perform destructive actions on their host or network.
    # Once you have read this disclaimer and taken appropriate precautions,
    # uncomment the following line and proceed at your own risk:
    assert all(o is not None for o in [
        task.p4d, task.g4d, task.solution])
    with create_tempdir():
        if gen_task.language == "py":
            import_helper = [
                "import math", "import re", "import sys", "import copy",
                "import datetime", "import itertools", "import collections",
                "import heapq", "import functools", "import hashlib", 
                "import numpy", "import numpy as np", "import string",
                "from typing import *", "from collections import *",
            ]
            import_helper = "\n".join(import_helper)
            code_with_test = "\n\n".join([import_helper, task.solution, gen_task.test])
            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(10):
                        exec(code_with_test, exec_globals)
                    task.passed = True
            except TimeoutException as e:
                task.passed = False
            except AssertionError as e:
                task.passed = False
            except Exception as e:
                task.passed = False
        elif gen_task.language == "js":
            code_with_test = "\n\n".join([task.solution, gen_task.test])
            code_file_name = "test.js"
            with open(code_file_name, "w") as file:
                file.write(code_with_test)
            try:
                exec_result = None
                with time_limit(10):
                    exec_result = subprocess.run([f"node", code_file_name], 
                                                timeout=10, 
                                                capture_output=True)

                if exec_result.stderr.decode():
                    task.passed = False
                elif exec_result.stdout.decode():
                    task.passed = False
                else:
                    task.passed = True
            except TimeoutException:
                task.passed = False
            finally:
                if task.passed is None:
                    task.passed = False
        else:
            raise NotImplementedError()
        if not task.passed:
            logging.warning(f"Not Pass:\n{code_with_test}\n{'*' * 30}\n")


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False

class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class TimeoutException(Exception):
    pass