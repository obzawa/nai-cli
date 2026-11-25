"""What a command sees of the outside: config, account, client, clock, standard input and job runs."""

import time
from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any

from nai_cli import api, auth, charge, config, errors

Result = tuple[dict[str, Any], int]
# A balance as read, or the error of a read that failed
Balance = tuple[api.Account | None, errors.NaiError | None]
LINES_IN_JOB = "a job prints one result; give each image its own nai meta job"


@dataclass
class _Shared:
    """What every job of a run shares: the config file is read once."""

    config_path: Path
    interrupted: bool = False
    loaded: config.Config | None = None
    # Read once per account for the dry runs of a run, as none of them spends
    balances: dict[str, Balance] = field(default_factory=dict)

    def interrupt(self) -> None:
        self.interrupted = True


@dataclass
class Traffic:
    """Requests put on the wire by every client of a run, which a batch paces."""

    # When the last request was sent, by the monotonic clock
    last: float | None = None
    # Seconds the next request waits after the last one; spent once, so a job's own requests do not wait
    wait: float = 0.0

    def before_send(self) -> None:
        if self.last is not None and self.wait > 0:
            time.sleep(max(0.0, self.wait - (time.monotonic() - self.last)))
        self.wait = 0.0
        self.last = time.monotonic()


@dataclass
class Context:
    shared: _Shared
    # From the front end: stdin read once, lines printed as they come, jobs run and parsed in process
    read_stdin: Callable[[], str]
    # A password: prompted for without echo at a terminal, else read from standard input
    read_secret: Callable[[str], str]
    emit: Callable[[dict[str, Any]], None]
    run_job: Callable[[list[str], "Context"], Result]
    parse_job: Callable[[list[str]], dict[str, Any]]
    now: Callable[[], datetime] = datetime.now
    # As typed, for the ledger
    argv: tuple[str, ...] = ()
    # A job of a batch or grid dry run
    dry_run: bool = False
    # Run as a job of a batch or grid, which prints one line for it
    in_job: bool = False
    # --account; otherwise the one chosen with nai account use
    account: str | None = None
    traffic: Traffic = field(default_factory=Traffic)
    _client: api.Client | None = field(default=None, init=False)

    @classmethod
    def create(
        cls,
        config_path: Path,
        read_stdin: Callable[[], str],
        read_secret: Callable[[str], str],
        emit: Callable[[dict[str, Any]], None],
        run_job: Callable[[list[str], "Context"], Result],
        parse_job: Callable[[list[str]], dict[str, Any]],
        argv: tuple[str, ...],
        account: str | None = None,
    ) -> "Context":
        return cls(_Shared(config_path), read_stdin, read_secret, emit, run_job, parse_job, argv=argv, account=account)

    @property
    def config(self) -> config.Config:
        if self.shared.loaded is None:
            self.shared.loaded = config.load(self.shared.config_path)
        return self.shared.loaded

    def forget_config(self) -> None:
        """After the config file is written, so later jobs of a batch read it again."""
        self.shared.loaded = None

    def refusal(
        self, price: int, account: api.Account | None, yes: bool, cost_json: dict[str, Any]
    ) -> dict[str, Any] | None:
        """The refusal the real run would meet before sending."""
        refused = charge.refusal(price, account, self.config, yes, cost_json)
        if self.dry_run and refused:
            raise errors.NaiError(refused["code"], refused["message"], dry_run=True, cost=cost_json)
        return refused

    @cached_property
    def account_name(self) -> str:
        """--account, else the one chosen with nai account use."""
        return self.account or auth.current()

    def charged(self, warnings: list[str], **entry: Any) -> AbstractContextManager[charge.Charge]:
        """Paid requests under the lock, written to the ledger with the account and the command line."""
        client = self.client()
        entry = {**entry, "account": self.account_name, "argv": list(self.argv)}
        return charge.charged(client, warnings, entry, self.now, self.shared.interrupt)

    def client(self) -> api.Client:
        """Created on first use; a command without a token fails here, before anything is paid for."""
        if self._client is None:
            name = self._known_account()
            self._client = self._new_client(name, renew=not self.dry_run)
        return self._client

    def balance(self) -> api.Account:
        """The account's balance; a dry run reads it without renewing a login, as it writes nothing."""
        if not self.dry_run:
            return self.client().account()
        account, failed = self._balance(self._known_account())
        if failed is not None:
            raise failed
        # No token: the client fails as the real run would
        return account or self.client().account()

    def dry_account(self, warnings: list[str]) -> api.Account | None:
        """For a command that sends nothing paid: the balance, read without renewing a login, or None to price as
        Opus when there is no token or it cannot be read."""
        # An account the real run would not find fails it before anything is priced
        account, failed = self._balance(self._known_account())
        if failed is not None:
            warnings.append(f"the balance could not be read, so the price assumes Opus: {failed.message}")
        return account

    def _balance(self, name: str) -> Balance:
        if not auth.has_token(self.config, name):
            return None, None
        if not self.dry_run:
            return self._read_balance(name)
        if name not in self.shared.balances:
            self.shared.balances[name] = self._read_balance(name)
        return self.shared.balances[name]

    def _known_account(self) -> str:
        name = self.account_name
        if name not in auth.names(self.config):
            raise errors.NaiError(errors.BAD_ARGS, f"no account named {name}", choices=auth.names(self.config))
        return name

    def _new_client(self, name: str, renew: bool) -> api.Client:
        found = auth.credentials(self.config, name, self.now(), renew=renew)
        return api.Client(found.token, found.endpoint, before_send=self.traffic.before_send)

    def _read_balance(self, name: str) -> Balance:
        try:
            client = self._new_client(name, renew=False)
            try:
                return client.account(), None
            finally:
                with suppress(Exception):
                    client.close()
        except errors.NaiError as e:
            return None, e

    def with_account(self, account: str) -> "Context":
        """A job that names its own account, with a client of its own."""
        return replace(self, account=account)

    def job(self, argv: list[str], dry_run: bool) -> "Context":
        # A job's standard input and output lines belong to the batch
        return replace(
            self,
            read_stdin=self._refuse_stdin,
            read_secret=self._refuse_secret,
            emit=self._refuse_lines,
            argv=tuple(argv),
            dry_run=dry_run,
            in_job=True,
        )

    def close(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            # Closing runs after the result is known, which a failure here must not replace
            try:
                client.close()
            except KeyboardInterrupt:
                self.shared.interrupt()
            except Exception:
                pass

    @staticmethod
    def _refuse_lines(line: dict[str, Any]) -> None:
        raise errors.NaiError(errors.BAD_ARGS, LINES_IN_JOB)

    @staticmethod
    def _refuse_secret(prompt: str) -> str:
        raise errors.NaiError(errors.BAD_ARGS, "a job cannot ask for a password; sign in with nai account login first")

    @staticmethod
    def _refuse_stdin() -> str:
        raise errors.NaiError(errors.BAD_ARGS, "standard input is already in use, so @- cannot read it")
