"""The command line: argv into settings, the command run, its result printed as JSON with an exit code."""

import getpass
import sys
import traceback
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from nai_cli import config, errors, parser, values
from nai_cli.commands import COMMANDS
from nai_cli.commands.context import Context, Result


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result, code = run(sys.argv[1:] if argv is None else list(argv))
    # Commands that stream lines, batch, grid and nai meta of several images, end with a summary line
    text, code = _dumps(result, code, None if result.get("type") == "summary" else 1)
    print(text)
    return code


def run(argv: list[str]) -> Result:
    """One command, as its result and exit code; nothing is printed except the lines a batch streams."""
    return _guarded(lambda: _run(argv, None))


def _run(argv: list[str], parent: Context | None) -> dict[str, Any]:
    opts = vars(parser.build_parser().parse_args(argv))
    config_path: Path | None = opts.pop("config")
    account: str | None = opts.pop("account")
    if parent is None:
        path = config_path or config.default_path()
        read_stdin = _stdin_reader()
        ctx = Context.create(
            path, read_stdin, _secret_reader(read_stdin), _emit, _run_job, _parse_job, tuple(argv), account
        )
    else:
        # A job keeps the batch's config, and so its limits; batch refuses --config in a job
        ctx = parent if account is None else parent.with_account(account)
    try:
        command = COMMANDS[opts.pop("cmd")]
        return command.run(command.resolve(opts, ctx), ctx)
    finally:
        ctx.close()


def _run_job(argv: list[str], ctx: Context) -> Result:
    try:
        # Only the job's line belongs on standard output
        with redirect_stdout(sys.stderr):
            return _guarded(lambda: _run(argv, ctx))
    except SystemExit:
        # --help and --version print their text and exit
        return {"ok": False, "code": errors.BAD_ARGS, "message": "--help and --version cannot run in a batch"}, 1


def _parse_job(argv: list[str]) -> dict[str, Any]:
    try:
        with redirect_stdout(sys.stderr):
            return vars(parser.build_parser().parse_args(argv))
    except SystemExit:
        raise errors.NaiError(errors.BAD_ARGS, "--help and --version cannot run in a grid") from None


def _guarded(command: Callable[[], dict[str, Any]]) -> Result:
    try:
        return command(), 0
    except errors.NaiError as e:
        # Interrupted or failed after paying: reported with the spend, otherwise as below
        if e.code == errors.INTERNAL and e.__cause__ is not None:
            traceback.print_exception(e.__cause__)
        return e.to_json(), 130 if e.code == errors.INTERRUPTED else 1
    except KeyboardInterrupt:
        return {"ok": False, "code": errors.INTERRUPTED, "message": "interrupted"}, 130
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "code": errors.INTERNAL, "message": f"{type(e).__name__}: {e}"}, 1


def _stdin_reader() -> Callable[[], str]:
    taken = False

    def read() -> str:
        nonlocal taken
        if taken:
            raise errors.NaiError(errors.BAD_ARGS, "standard input is already in use, so @- cannot read it")
        taken = True
        try:
            raw = sys.stdin.buffer.read() if hasattr(sys.stdin, "buffer") else sys.stdin.read().encode("utf-8")
            return raw.decode("utf-8-sig")
        except (OSError, UnicodeDecodeError) as e:
            raise errors.NaiError(errors.BAD_ARGS, f"cannot read standard input: {e}") from e

    return read


def _secret_reader(read_stdin: Callable[[], str]) -> Callable[[str], str]:
    def read(prompt: str) -> str:
        # At a terminal the secret is typed without echo, as gh and ssh take one; piped, it is standard input
        if sys.stdin is None or not sys.stdin.isatty():
            return read_stdin().rstrip("\r\n")
        try:
            # getpass writes its prompt to the terminal, never to standard output, which holds the JSON
            return getpass.getpass(prompt)
        except EOFError:
            return ""

    return read


def _emit(line: dict[str, Any]) -> None:
    text, code = _dumps(line, 0, None)
    print(text, flush=True)
    if code:
        # The summary after it must not report a line that was lost
        raise errors.NaiError(errors.INTERNAL, "a line of output is not valid JSON")


def _dumps(result: dict[str, Any], code: int, indent: int | None) -> tuple[str, int]:
    try:
        return values.to_json(result, indent), code
    except ValueError as e:
        result = {"ok": False, "code": errors.INTERNAL, "message": f"output is not valid JSON: {e}"}
        return values.to_json(result, indent), 1
