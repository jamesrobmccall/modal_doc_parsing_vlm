import os
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODAL_TESTS") != "1",
    reason="Set RUN_MODAL_TESTS=1 to run live Modal integration tests.",
)


def modal_bin() -> str:
    local_modal = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "modal"
    return os.environ.get(
        "MODAL_BIN",
        str(local_modal if local_modal.exists() else Path.home() / ".local" / "bin" / "modal"),
    )


def run_modal_command(*args: str) -> subprocess.CompletedProcess[str]:
    env_name = os.environ.get("MODAL_ENVIRONMENT")
    cmd = [modal_bin()]
    if env_name and args and args[0] in {"run", "app", "container"}:
        cmd.extend([args[0], "-e", env_name, *args[1:]])
    else:
        cmd.extend(args)
    return subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        text=True,
        capture_output=True,
    )


def test_modal_smoke_test_cli():
    result = run_modal_command(
        "run",
        "app.py::smoke_test",
        "--runtime-profile-name",
        "dev",
    )
    assert result.returncode == 0, result.stderr


def test_modal_cleanup_cli():
    result = run_modal_command("run", "app.py::cleanup_now")
    assert result.returncode == 0, result.stderr


def test_modal_entity_extraction_smoke_cli():
    result = run_modal_command("run", "app.py::smoke_entity_extraction")
    assert result.returncode == 0, result.stderr
