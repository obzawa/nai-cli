"""account: the balance and this month's spend, the accounts there are, and email sign-in."""

import calendar
from dataclasses import dataclass
from datetime import date
from typing import Any

from nai_cli import auth, errors, ledger
from nai_cli.commands.context import Context


@dataclass(frozen=True)
class AccountSettings:
    action: str
    name: str | None
    email: str | None
    month: date
    # Asked by a dry run of several jobs: checked, but the store is left as it was
    dry_run: bool


def resolve(opts: dict[str, Any], ctx: Context) -> AccountSettings:
    action, name, email = opts["action"], opts.get("name"), opts.get("email")
    if action in ("use", "login", "logout") and not name:
        raise errors.NaiError(errors.BAD_ARGS, f"account {action} needs the account's name")
    if action in ("show", "list") and name:
        raise errors.NaiError(errors.BAD_ARGS, f"account {action} takes no name; use --account NAME")
    if (action == "login") != bool(email):
        raise errors.NaiError(errors.BAD_ARGS, "--email goes with account login, which needs it")
    if opts.get("month") and action != "show":
        raise errors.NaiError(errors.BAD_ARGS, "--month only applies to account show")
    if action == "login" and ctx.dry_run:
        raise errors.NaiError(errors.BAD_ARGS, "account login signs in for real, so a dry run cannot include it")
    return AccountSettings(action, name, email, _month(opts.get("month"), ctx), ctx.dry_run)


def _month(text: str | None, ctx: Context) -> date:
    if not text:
        return ctx.now().date().replace(day=1)
    try:
        return date.fromisoformat(f"{text}-01")
    except ValueError:
        raise errors.NaiError(errors.BAD_ARGS, f"--month takes YYYY-MM: {text!r}") from None


def run(s: AccountSettings, ctx: Context) -> dict[str, Any]:
    match s.action:
        case "list":
            return {"ok": True, "accounts": auth.listing(ctx.config)}
        case "use":
            auth.use(s.name or "", ctx.config, dry_run=s.dry_run)
            return _changed(s)
        case "login":
            # The password is never an option, so it stays out of the shell history and the config
            password = ctx.read_secret("Password: ")
            if not password:
                raise errors.NaiError(errors.BAD_ARGS, "type the password at the prompt or pipe it to standard input")
            return {"ok": True, **auth.login(s.name or "", s.email or "", password, ctx.config, ctx.now())}
        case "logout":
            if not auth.logout(s.name or "", dry_run=s.dry_run):
                raise errors.NaiError(errors.BAD_ARGS, f"{s.name} is not an email login")
            return _changed(s)
    account = ctx.balance()
    until = s.month.replace(day=calendar.monthrange(s.month.year, s.month.month)[1])
    try:
        month = ledger.summary(s.month, until, "command")
    except OSError as e:
        raise errors.NaiError(errors.LEDGER, f"cannot read the ledger: {e}") from e
    return {
        "ok": True,
        "account": ctx.account_name,
        **account.to_json(),
        # The ledger is shared by every account on this computer, so the month is not this account's alone
        "month": {"month": s.month.isoformat()[:7], "scope": "all_accounts", **month},
    }


def _changed(s: AccountSettings) -> dict[str, Any]:
    return {"ok": True, "account": s.name, "dry_run": True} if s.dry_run else {"ok": True, "account": s.name}
