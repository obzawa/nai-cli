"""The usage guide and the smoke run keep up with the command line: every command and option is in both where due."""

import argparse
import ast
import re
from pathlib import Path

from nai_cli import parser as nai_parser

ROOT = Path(__file__).resolve().parents[1]
USAGE = ROOT / "docs" / "usage.md"
SMOKE = ROOT / ".github" / "scripts" / "smoke.py"


def _commands() -> dict[str, argparse.ArgumentParser]:
    parser = nai_parser.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return {"": parser, **sub.choices}


def test_every_command_is_in_the_usage_guide():
    text = USAGE.read_text(encoding="utf-8")
    missing = [name for name in _commands() if name and not re.search(rf"`nai {name}\b", text)]
    assert not missing, f"docs/usage.md never shows these commands: {missing}"


def test_every_option_is_in_the_usage_guide():
    # An option counts as written in any of its spellings: -r, --recursive or --no-recursive
    text = USAGE.read_text(encoding="utf-8")
    missing = {}
    for name, parser in _commands().items():
        for action in parser._actions:
            if isinstance(action, argparse._HelpAction) or not action.option_strings:
                continue
            spellings = [s for s in action.option_strings if s != "-h"]
            if not any(re.search(rf"(?<![\w-]){re.escape(s)}(?![\w-])", text) for s in spellings):
                missing.setdefault(spellings[-1], []).append(name or "nai")
    assert not missing, f"docs/usage.md never mentions these options: {missing}"


def test_every_command_runs_in_the_smoke_run():
    # The first word of each command line the smoke run gives nai
    tree = ast.parse(SMOKE.read_text(encoding="utf-8"))
    run = {
        node.elts[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.List) and node.elts and isinstance(node.elts[0], ast.Constant)
    }
    missing = [name for name in _commands() if name and name not in run]
    assert not missing, f".github/scripts/smoke.py never runs these commands: {missing}"
