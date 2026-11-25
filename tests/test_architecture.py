"""The structure docs/architecture.md sets: layers, a core without I/O, who writes to disk, who talks to the terminal."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "nai_cli"

# Lower layers come later; a module may import its own layer and those below it
LAYERS = [
    {"cli", "parser", "__main__"},
    {"commands"},
    {"charge", "batch", "grid", "import_image", "caches"},
    {
        "request",
        "prompt",
        "tidy",
        "models",
        "presets",
        "meta",
        "rerun",
        "vibe",
        "inpaint",
        "enhance",
        "cost",
        "naming",
        "tokenizer",
        "lint",
        "tags",
        "resample",
        "images",
        "share",
        "random_prompt",
        "sheet",
        "pixel_snap",
        "comfyui",
        "tagger",
        "sam",
    },
    {"local_models", "fonts", "api", "auth", "fs", "ledger", "config", "errors", "decode", "values", "__init__"},
]
CORE = LAYERS[3]
# The core reads the files it is given, but never talks to the network, loads models, reads the config or writes
_CORE_MAY_USE = {"errors", "values", "decode", "__init__"}
_CORE_FORBIDDEN = {
    *(f"nai_cli.{m}" for m in LAYERS[4] - _CORE_MAY_USE),
    "httpx",
    "onnxruntime",
    "http.client",
    "socket",
    "urllib.request",
    "os.environ",
    "os.getenv",
}
# Every write goes through fs, so no file is overwritten and a dry run writes nothing
FILE_OWNER = "fs.py"
_WRITE_METHODS = {"write_bytes", "write_text", "mkdir", "unlink", "rename", "touch", "rmdir", "symlink_to"}
_OS_WRITES = {"open", "replace", "rename", "remove", "unlink", "makedirs", "mkdir", "rmdir"}
_TERMINAL = {"stdin", "stdout", "exit"}


def _modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _name(path: Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def _layer(module: str) -> int:
    return next(i for i, layer in enumerate(LAYERS) if module in layer)


def _is_module(name: str) -> bool:
    return (SRC / f"{name}.py").exists() or (SRC / name).is_dir()


def _imported(tree: ast.AST) -> list[str]:
    """The nai_cli modules a file imports, by the name the layer table uses."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "nai_cli":
            parts = node.module.split(".")
            if len(parts) > 1:
                out.append(parts[1])
            else:
                # from nai_cli import __version__ takes a name from the package itself
                out += [alias.name if _is_module(alias.name) else "__init__" for alias in node.names]
        elif isinstance(node, ast.Import):
            out += [a.name.split(".")[1] for a in node.names if a.name.startswith("nai_cli.")]
    return out


def layer_violations(module: str, tree: ast.AST) -> list[str]:
    return [f"{module} imports {target}" for target in _imported(tree) if _layer(target) < _layer(module)]


def core_io(module: str, tree: ast.AST) -> list[str]:
    if module not in CORE:
        return []
    bindings = {}
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                bindings[alias.asname or root] = alias.name if alias.asname else root
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            prefix = ("nai_cli." if node.level == 1 else "") + (node.module or "")
            for alias in node.names:
                full = f"{prefix.rstrip('.')}.{alias.name}"
                bindings[alias.asname or alias.name] = full
                names.add(full)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = _qualified(node, bindings)
            if name:
                names.add(name)
    return [f"{module} uses {name}" for name in sorted(names) if _core_forbids(name)]


def _qualified(node: ast.expr, bindings: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute) and (owner := _qualified(node.value, bindings)):
        return f"{owner}.{node.attr}"
    return ""


def _core_forbids(name: str) -> bool:
    return name.startswith("client.") or any(name == base or name.startswith(base + ".") for base in _CORE_FORBIDDEN)


def _bytes_buffers(function: ast.AST) -> set[str]:
    return {
        target.id
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and _is_call(node.value, "io", "BytesIO")
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _is_call(node: ast.AST, owner: str, attr: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
    )


def _write_mode(call: ast.Call, position: int) -> bool:
    mode = call.args[position] if len(call.args) > position else None
    mode = next((k.value for k in call.keywords if k.arg == "mode"), mode)
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and bool(set(mode.value) & set("wax+"))


def disk_writes(tree: ast.AST) -> list[int]:
    """Lines that write to disk; saving an image into an in-memory buffer does not count."""
    lines = []
    scopes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)] or [tree]
    buffers = set().union(*(_bytes_buffers(s) for s in scopes))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open" and _write_mode(node, 1):
            lines.append(node.lineno)
        if not isinstance(func, ast.Attribute):
            continue
        owner = func.value.id if isinstance(func.value, ast.Name) else None
        if (
            func.attr in _WRITE_METHODS
            or (owner == "os" and func.attr in _OS_WRITES)
            or owner == "shutil"
            or (func.attr == "open" and owner != "os" and _write_mode(node, 0))
        ):
            lines.append(node.lineno)
        elif func.attr == "save":
            target = node.args[0] if node.args else None
            if not (isinstance(target, ast.Name) and target.id in buffers):
                lines.append(node.lineno)
    return lines


def terminal_uses(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "sys" and node.attr in _TERMINAL:
                lines.append(node.lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            lines.append(node.lineno)
    return lines


def config_and_client_sites(tree: ast.AST) -> list[int]:
    return [
        node.lineno for node in ast.walk(tree) if _is_call(node, "config", "load") or _is_call(node, "api", "Client")
    ]


def inline_imports(tree: ast.AST) -> list[int]:
    lines = []
    for scope in ast.walk(tree):
        if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            lines += [n.lineno for n in ast.walk(scope) if isinstance(n, ast.Import | ast.ImportFrom)]
    return lines


def helpers_above_callers(tree: ast.AST) -> list[int]:
    """Private functions defined above a public function that calls them."""
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    helpers = {f.name: f.lineno for f in functions if f.name.startswith("_")}
    return sorted(
        {
            helpers[call.func.id]
            for f in functions
            if not f.name.startswith("_")
            for call in ast.walk(f)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and 0 < helpers.get(call.func.id, 0) < f.lineno
        }
    )


def _parsed() -> list[tuple[Path, ast.AST]]:
    return [(path, ast.parse(path.read_text(encoding="utf-8"))) for path in _modules()]


def test_every_module_has_a_layer():
    unknown = {_name(path) for path in _modules()} - set().union(*LAYERS)
    assert not unknown, f"add these to the layer table here and in docs/architecture.md: {unknown}"


def test_imports_follow_the_layers():
    assert [v for path, tree in _parsed() for v in layer_violations(_name(path), tree)] == []


def test_the_core_does_no_io():
    assert [v for path, tree in _parsed() for v in core_io(_name(path), tree)] == []


def test_only_fs_writes_to_disk():
    found = {path.name: disk_writes(tree) for path, tree in _parsed() if path.name != FILE_OWNER}
    assert {name: lines for name, lines in found.items() if lines} == {}


def test_only_cli_touches_the_terminal():
    found = {path.name: terminal_uses(tree) for path, tree in _parsed() if path.name != "cli.py"}
    assert {name: lines for name, lines in found.items() if lines} == {}


def test_one_site_loads_the_config_and_creates_the_client():
    sites = {
        str(path.relative_to(SRC)): config_and_client_sites(tree)
        for path, tree in _parsed()
        if config_and_client_sites(tree)
    }
    assert list(sites) == [str(Path("commands") / "context.py")] and len(next(iter(sites.values()))) == 2


def test_commands_do_not_see_options_parsing():
    for path, tree in _parsed():
        if _name(path) != "commands":
            continue
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert "argparse" not in imported and "parser" not in _imported(tree), path.name


def test_no_imports_inside_functions():
    assert {path.name: inline_imports(tree) for path, tree in _parsed() if inline_imports(tree)} == {}


def test_public_functions_come_before_their_helpers():
    found = {path.name: helpers_above_callers(tree) for path, tree in _parsed()}
    assert {name: lines for name, lines in found.items() if lines} == {}


@pytest.mark.parametrize(
    ("check", "source"),
    [
        (disk_writes, "from pathlib import Path\nPath('x').write_text('a')"),
        (disk_writes, "open('x', 'wb')"),
        (disk_writes, "import os\nos.replace('a', 'b')"),
        (disk_writes, "def f(img, path):\n    img.save(path)"),
        (disk_writes, "def f(p):\n    p.open(mode='a')"),
        (disk_writes, "import shutil\nshutil.copy('a', 'b')"),
        (terminal_uses, "print('x')"),
        (terminal_uses, "import sys\nsys.stdout.write('x')"),
        (config_and_client_sites, "from nai_cli import config\nconfig.load(p)"),
        (config_and_client_sites, "from nai_cli import api\napi.Client('t')"),
        (inline_imports, "def f():\n    import json"),
        (helpers_above_callers, "def _g():\n    pass\ndef f():\n    _g()"),
    ],
)
def test_checks_catch_what_they_forbid(check, source):
    assert check(ast.parse(source))


@pytest.mark.parametrize(
    ("check", "source"),
    [
        (disk_writes, "import io\ndef f(img):\n    buf = io.BytesIO()\n    img.save(buf, format='PNG')"),
        (disk_writes, "def f(p):\n    p.open(encoding='utf-8')"),
        (disk_writes, "open('x')"),
        (terminal_uses, "import sys\nsys.stderr.write('x')"),
        (inline_imports, "import json\nif True:\n    import os"),
        (helpers_above_callers, "def f():\n    _g()\ndef _g():\n    pass\ndef _h():\n    _g()"),
    ],
)
def test_checks_allow_what_they_permit(check, source):
    assert not check(ast.parse(source))


def test_layer_check_catches_an_upward_import():
    assert layer_violations("prompt", ast.parse("from nai_cli.commands import generate"))
    assert layer_violations("api", ast.parse("from nai_cli import images"))
    assert not layer_violations("commands", ast.parse("from nai_cli import images, fs"))


def test_core_check_catches_io():
    assert core_io("lint", ast.parse("from nai_cli import api"))
    assert core_io("vibe", ast.parse("from nai_cli import config"))
    assert core_io("vibe", ast.parse("from nai_cli import fs\nfs.replace(p, 'x')"))
    assert core_io("vibe", ast.parse("from nai_cli.fs import replace"))
    assert core_io("tagger", ast.parse("from nai_cli import fs\nfs.is_file(p)"))
    assert core_io("sam", ast.parse("import onnxruntime"))
    assert not core_io("naming", ast.parse("from nai_cli import errors, values\nvalues.split_name(p)"))
    assert not core_io("caches", ast.parse("from nai_cli import api, config, fs\nfs.replace(p, 'x')"))


@pytest.mark.parametrize(
    "source",
    [
        "from nai_cli import fs as files\nfiles.replace(p, b'')",
        "import nai_cli.fs\nnai_cli.fs.make_folder(p)",
        "from .config import data_dir",
        "from os import environ as env\nenv.get('XDG_DATA_HOME')",
        "import httpx as http\nhttp.get(url)",
        "def query(client): return client.suggest_tags('x', 'm', None, None)",
    ],
)
def test_core_check_catches_indirect_io(source):
    assert core_io("tags", ast.parse(source))
