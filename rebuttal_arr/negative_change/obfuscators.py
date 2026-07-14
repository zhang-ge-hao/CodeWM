"""Small, non-executing wrappers around the two Python obfuscators.

The wrappers invoke the command-line tools from the existing ``watermarking``
Conda environment.  Input programs are only transformed; they are never
imported, evaluated, compiled, or otherwise executed.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


OBFUSCATOR_NAMES = ("pyminify", "pyminifier")
DEFAULT_ENV_BIN = Path.home() / "conda" / "envs" / "watermarking" / "bin"
PYMINIFIER_BANNER = "# Created by pyminifier"


@dataclass(frozen=True)
class ObfuscationResult:
    name: str
    ok: bool
    code: str | None
    error_code: str | None
    error_message: str | None
    stderr: str
    returncode: int | None
    command: tuple[str, ...]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["command"] = list(self.command)
        return value


def _environment_bin(env_bin: Path | str | None = None) -> Path:
    if env_bin is not None:
        return Path(env_bin).expanduser().resolve()
    override = os.environ.get("CODEWM_WATERMARKING_ENV_BIN")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_ENV_BIN.resolve()


def _executable(name: str, env_bin: Path | str | None = None) -> Path:
    return _environment_bin(env_bin) / name


def available_obfuscators(
    *, env_bin: Path | str | None = None
) -> dict[str, dict[str, Any]]:
    """Report executable availability without installing or modifying anything."""

    availability: dict[str, dict[str, Any]] = {}
    for name in OBFUSCATOR_NAMES:
        executable = _executable(name, env_bin)
        availability[name] = {
            "path": executable.as_posix(),
            "exists": executable.is_file(),
            "executable": executable.is_file() and os.access(executable, os.X_OK),
        }
    return availability


def obfuscator_versions(
    *,
    env_bin: Path | str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, str | None]:
    """Return tool versions, or ``None`` when a version cannot be queried."""

    versions: dict[str, str | None] = {}
    for name in OBFUSCATOR_NAMES:
        executable = _executable(name, env_bin)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            versions[name] = None
            continue
        try:
            result = subprocess.run(
                [executable.as_posix(), "--version"],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            versions[name] = None
            continue
        output = (result.stdout or result.stderr).strip()
        versions[name] = output if result.returncode == 0 and output else None
    return versions


def _result(
    *,
    name: str,
    start: float,
    command: Sequence[str] = (),
    ok: bool = False,
    code: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    stderr: str = "",
    returncode: int | None = None,
) -> ObfuscationResult:
    return ObfuscationResult(
        name=name,
        ok=ok,
        code=code,
        error_code=error_code,
        error_message=error_message,
        stderr=stderr,
        returncode=returncode,
        command=tuple(command),
        elapsed_seconds=max(0.0, time.monotonic() - start),
    )


def _remove_pyminifier_banner(code: str) -> str:
    # Match the original experiment's behaviour, including newline handling.
    return "\n".join(
        line for line in code.split("\n") if PYMINIFIER_BANNER not in line
    )


def obfuscate_python(
    code: str,
    name: str,
    *,
    timeout_seconds: float = 10.0,
    env_bin: Path | str | None = None,
) -> ObfuscationResult:
    """Transform Python source using one supported command-line obfuscator.

    Every failure is returned as a structured result.  No shell is involved and
    the input program is never executed.
    """

    start = time.monotonic()
    if name not in OBFUSCATOR_NAMES:
        return _result(
            name=name,
            start=start,
            error_code="unsupported_obfuscator",
            error_message=f"Unsupported Python obfuscator: {name!r}",
        )
    if not isinstance(code, str):
        return _result(
            name=name,
            start=start,
            error_code="invalid_input",
            error_message=f"Expected code to be str, got {type(code).__name__}",
        )
    if not isinstance(timeout_seconds, (int, float)) or isinstance(
        timeout_seconds, bool
    ) or timeout_seconds <= 0:
        return _result(
            name=name,
            start=start,
            error_code="invalid_timeout",
            error_message=f"timeout_seconds must be positive, got {timeout_seconds!r}",
        )

    executable = _executable(name, env_bin)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        return _result(
            name=name,
            start=start,
            command=(executable.as_posix(),),
            error_code="executable_missing",
            error_message=f"Executable is unavailable: {executable}",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="codewm-negative-obfuscate-") as temp:
            work_dir = Path(temp)
            source_path = work_dir / "solution.py"
            source_path.write_text(code, encoding="utf-8")
            if name == "pyminify":
                command = [
                    executable.as_posix(),
                    "--remove-literal-statements",
                    source_path.name,
                ]
            else:
                command = [executable.as_posix(), source_path.name]

            try:
                completed = subprocess.run(
                    command,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=float(timeout_seconds),
                    check=False,
                    shell=False,
                )
            except subprocess.TimeoutExpired as exc:
                stderr = exc.stderr
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                return _result(
                    name=name,
                    start=start,
                    command=command,
                    error_code="timeout",
                    error_message=(
                        f"{name} exceeded the {float(timeout_seconds):g}s timeout"
                    ),
                    stderr=stderr or "",
                )
            except OSError as exc:
                return _result(
                    name=name,
                    start=start,
                    command=command,
                    error_code="process_error",
                    error_message=str(exc),
                )

            if completed.returncode != 0:
                return _result(
                    name=name,
                    start=start,
                    command=command,
                    error_code="process_failed",
                    error_message=f"{name} exited with status {completed.returncode}",
                    stderr=completed.stderr,
                    returncode=completed.returncode,
                )

            transformed = completed.stdout
            if name == "pyminifier":
                transformed = _remove_pyminifier_banner(transformed)
            return _result(
                name=name,
                start=start,
                command=command,
                ok=True,
                code=transformed,
                stderr=completed.stderr,
                returncode=completed.returncode,
            )
    except OSError as exc:
        return _result(
            name=name,
            start=start,
            error_code="io_error",
            error_message=str(exc),
        )


def obfuscate_all(
    code: str,
    *,
    timeout_seconds: float = 10.0,
    env_bin: Path | str | None = None,
) -> Mapping[str, ObfuscationResult]:
    """Run both supported transforms independently and return results by name."""

    return {
        name: obfuscate_python(
            code,
            name,
            timeout_seconds=timeout_seconds,
            env_bin=env_bin,
        )
        for name in OBFUSCATOR_NAMES
    }


__all__ = [
    "DEFAULT_ENV_BIN",
    "OBFUSCATOR_NAMES",
    "ObfuscationResult",
    "available_obfuscators",
    "obfuscate_all",
    "obfuscate_python",
    "obfuscator_versions",
]
