from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SENSOR_AGGREGATE_CONTRACT = "Sensor strategy-agnostic: no aggregate import"
ACTUATOR_AGGREGATE_CONTRACT = "Actuator strategy-agnostic: no aggregate import"
SENSOR_ACTUATOR_CONTROLLER_CONTRACT = "Sensor + Actuator: no controller import"
SENSOR_MARKET_SELECTION_CONTRACT = "Sensor: no market_selection import"
ACTUATOR_MARKET_SELECTION_CONTRACT = "Actuator: no market_selection import"
MARKET_SELECTION_AGGREGATE_CONTRACT = "Market selection: no aggregate import"
KALSHI_STUB_BOUNDARY_CONTRACT = "Kalshi stub helper stays at runtime boundaries"
MCP_PROXY_CONTRACT = "MCP proxy only calls the typed HTTP API"

IMPORT_LINTER_CONFIG = """
[project]
name = "scratch-import-linter"
version = "0.0.1"
requires-python = ">=3.11"

[tool.importlinter]
root_packages = ["pms"]
include_external_packages = true

[[tool.importlinter.contracts]]
name = "Sensor strategy-agnostic: no aggregate import"
type = "forbidden"
source_modules = ["pms.sensor"]
forbidden_modules = ["pms.strategies.aggregate"]

[[tool.importlinter.contracts]]
name = "Actuator strategy-agnostic: no aggregate import"
type = "forbidden"
source_modules = ["pms.actuator"]
forbidden_modules = ["pms.strategies.aggregate"]

[[tool.importlinter.contracts]]
name = "Sensor + Actuator: no controller import"
type = "forbidden"
source_modules = ["pms.sensor", "pms.actuator"]
forbidden_modules = ["pms.controller"]

[[tool.importlinter.contracts]]
name = "Sensor: no market_selection import"
type = "forbidden"
source_modules = ["pms.sensor"]
forbidden_modules = ["pms.market_selection"]

[[tool.importlinter.contracts]]
name = "Actuator: no market_selection import"
type = "forbidden"
source_modules = ["pms.actuator"]
forbidden_modules = ["pms.market_selection"]

[[tool.importlinter.contracts]]
name = "Market selection: no aggregate import"
type = "forbidden"
source_modules = ["pms.market_selection"]
forbidden_modules = ["pms.strategies.aggregate"]

[[tool.importlinter.contracts]]
name = "Kalshi stub helper stays at runtime boundaries"
type = "protected"
protected_modules = ["pms.core.venue_support"]
allowed_importers = [
    "pms.sensor.adapters.historical",
    "pms.sensor.adapters.market_discovery",
    "pms.market_selection.selector",
    "pms.actuator.executor",
    "pms.actuator.adapters.paper",
    "pms.actuator.adapters.backtest",
    "pms.research.execution",
]
as_packages = false

[[tool.importlinter.contracts]]
name = "MCP proxy only calls the typed HTTP API"
type = "forbidden"
source_modules = ["pms.agent_mcp"]
forbidden_modules = [
    "pms.sensor",
    "pms.controller",
    "pms.actuator",
    "pms.actuator.adapters",
    "py_clob_client_v2",
]
""".strip()


def _lint_imports_command() -> list[str]:
    executable = REPO_ROOT / ".venv" / "bin" / "lint-imports"
    if executable.exists():
        return [str(executable)]

    executable = Path(sys.executable).resolve().with_name("lint-imports")
    if executable.exists():
        return [str(executable)]

    resolved = shutil.which("lint-imports")
    if resolved is None:
        pytest.fail("lint-imports executable is not available in the active environment")
    return [resolved]


@pytest.mark.parametrize(
    ("violating_module", "source_text", "expected_contract"),
    [
        (
            "pms/sensor/bad.py",
            "from pms.strategies.aggregate import Strategy\n",
            SENSOR_AGGREGATE_CONTRACT,
        ),
        (
            "pms/sensor/bad.py",
            "from pms.market_selection.selector import MarketSelector\n",
            SENSOR_MARKET_SELECTION_CONTRACT,
        ),
        (
            "pms/actuator/bad.py",
            "from pms.market_selection.selector import MarketSelector\n",
            ACTUATOR_MARKET_SELECTION_CONTRACT,
        ),
        (
            "pms/market_selection/bad.py",
            "from pms.strategies.aggregate import Strategy\n",
            MARKET_SELECTION_AGGREGATE_CONTRACT,
        ),
        (
            "pms/controller/bad.py",
            "from pms.core.venue_support import kalshi_stub_error\n",
            KALSHI_STUB_BOUNDARY_CONTRACT,
        ),
        (
            "pms/agent_mcp/bad.py",
            "import py_clob_client_v2\n",
            MCP_PROXY_CONTRACT,
        ),
        (
            "pms/agent_mcp/bad.py",
            "import pms.actuator.adapters\n",
            MCP_PROXY_CONTRACT,
        ),
    ],
)
def test_import_linter_rejects_forbidden_dependencies(
    tmp_path: Path,
    violating_module: str,
    source_text: str,
    expected_contract: str,
) -> None:
    (tmp_path / "pyproject.toml").write_text(f"{IMPORT_LINTER_CONFIG}\n")
    for package in [
        "pms",
        "pms/sensor",
        "pms/actuator",
        "pms/controller",
        "pms/strategies",
        "pms/market_selection",
        "pms/core",
        "pms/agent_mcp",
    ]:
        package_dir = tmp_path / package
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "__init__.py").write_text("")
    (tmp_path / "pms/strategies/aggregate.py").write_text("class Strategy: ...\n")
    (tmp_path / "pms/core/venue_support.py").write_text(
        "def kalshi_stub_error(context: str) -> Exception:\n"
        "    return NotImplementedError(context)\n"
    )
    (tmp_path / "pms/market_selection/selector.py").write_text(
        "class MarketSelector: ...\n"
    )
    (tmp_path / violating_module).write_text(source_text)

    result = subprocess.run(
        [
            *_lint_imports_command(),
            "--config",
            str(tmp_path / "pyproject.toml"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert expected_contract in result.stdout


def test_current_tree_lints_clean() -> None:
    result = subprocess.run(
        _lint_imports_command(),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    for contract in [
        SENSOR_AGGREGATE_CONTRACT,
        ACTUATOR_AGGREGATE_CONTRACT,
        SENSOR_ACTUATOR_CONTROLLER_CONTRACT,
        SENSOR_MARKET_SELECTION_CONTRACT,
        ACTUATOR_MARKET_SELECTION_CONTRACT,
        MARKET_SELECTION_AGGREGATE_CONTRACT,
        KALSHI_STUB_BOUNDARY_CONTRACT,
        MCP_PROXY_CONTRACT,
    ]:
        assert contract in result.stdout
