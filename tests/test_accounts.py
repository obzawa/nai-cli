"""Accounts: config tokens, relays, email logins and switching between them."""

import io
import json
import threading
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from helpers import FakeClient, fake_clients, run, run_batch
from PIL import Image
from samples import nai_image, pixel_art

from nai_cli import api, auth, cli, config, errors, fs, ledger
from nai_cli.commands import output

NOW = datetime(2026, 11, 18, 12, tzinfo=UTC)


def _add_config(text):
    path = config.default_path()
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


def _stdin(monkeypatch, text):
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(text.encode()), encoding="utf-8"))


def test_access_key_follows_the_official_derivation():
    key = auth.access_key("Someone@Example.com", "correct horse")
    assert len(key) == 64 and set(key) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    # The email is lowercased before salting; the password is not
    assert key == auth.access_key("someone@example.com", "correct horse")
    assert key != auth.access_key("someone@example.com", "Correct horse")


def test_password_start_is_cut_in_utf16_units():
    assert auth._js_prefix("abcdefgh", 6) == "abcdef"
    # A character outside the BMP takes two units; a pair cut in half becomes U+FFFD, as TextEncoder writes it
    assert auth._js_prefix("abcde\U0001f600x", 6) == "abcde�"
    assert auth._js_prefix("abcd\U0001f600x", 6) == "abcd\U0001f600"


def test_login_exchanges_the_key_on_the_image_host():
    seen = []

    def handler(request):
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(201, json={"accessToken": "jwt-token"})

    assert api.login(api.OFFICIAL, "KEY", transport=httpx.MockTransport(handler)) == "jwt-token"
    assert seen == [("https://image.novelai.net/user/login", {"key": "KEY"})]
    refused = httpx.MockTransport(lambda r: httpx.Response(401, json={"message": "no"}))
    with pytest.raises(errors.NaiError) as e:
        api.login(api.OFFICIAL, "KEY", transport=refused)
    assert e.value.code == errors.AUTH


@pytest.mark.parametrize(
    ("typed", "image", "main", "images"),
    [
        ("relay.example.com/", "", "https://relay.example.com", "https://relay.example.com"),
        ("localhost:8000", "", "http://localhost:8000", "http://localhost:8000"),
        ("https://a.example/api/", "https://b.example", "https://a.example/api", "https://b.example"),
    ],
)
def test_relay_addresses_are_normalized(typed, image, main, images):
    endpoint = api.endpoint(typed, image)
    assert (endpoint.api_base, endpoint.image_base) == (main, images) and not endpoint.official
    assert endpoint.user("/user/subscription") == main + "/user/subscription"


@pytest.mark.parametrize("bad", ["ftp://x.example", "https://x.example/?a=1", "https://u:p@x.example", "https://"])
def test_bad_relay_addresses_refused(bad):
    with pytest.raises(ValueError):
        api.endpoint(bad)


def test_relay_without_a_subscription_endpoint_is_unmeasured():
    relay = api.endpoint("https://relay.example")
    for reply in (
        httpx.Response(404, text="nope"),
        httpx.Response(200, text="<html>", headers={"content-type": "text/html"}),
    ):
        c = api.Client("t", relay, transport=httpx.MockTransport(lambda r, reply=reply: reply))
        assert c.account().measured is False
    # The official service never falls back
    official = api.Client("t", transport=httpx.MockTransport(lambda r: httpx.Response(404, text="nope")))
    with pytest.raises(errors.NaiError):
        official.account()


def test_accounts_from_the_config_and_switching(capsys, with_token, monkeypatch):
    _add_config('[accounts.relay]\ntoken = "relay-token"\napi_url = "https://relay.example"\n')
    clients = fake_clients(monkeypatch)
    code, out = run(capsys, "account", "list")
    assert code == 0 and [a["account"] for a in out["accounts"]] == ["default", "relay"]
    assert out["accounts"][0]["current"] and "token" not in json.dumps(out["accounts"][1]).replace("token_set", "")
    run(capsys, "--account", "relay", "gen", "--prompt", "x")
    assert clients[-1].endpoint.api_base == "https://relay.example"
    code, out = run(capsys, "account", "use", "relay")
    assert code == 0 and auth.current() == "relay"
    run(capsys, "gen", "--prompt", "x")
    assert clients[-1].endpoint.api_base == "https://relay.example"
    code, out = run(capsys, "account", "use", "nobody")
    assert code == 1 and out["code"] == "bad_args" and "relay" in out["choices"]
    code, out = run(capsys, "--account", "nobody", "gen", "--prompt", "x")
    assert code == 1 and out["code"] == "bad_args"


def test_email_login_keeps_the_key_not_the_password(capsys, monkeypatch):
    calls = []

    def login(endpoint, key):
        calls.append(key)
        return f"token-{len(calls)}"

    monkeypatch.setattr(auth.api, "login", login)
    _stdin(monkeypatch, "hunter2 secret\n")
    code, out = run(capsys, "account", "login", "mine", "--email", "Me@Example.com")
    assert code == 0 and out["account"] == "mine" and out["email"] == "me@example.com"
    stored = auth.credentials_path().read_text(encoding="utf-8")
    assert "hunter2" not in stored and calls[0] in stored and "token-1" in stored
    code, out = run(capsys, "account", "login", "default", "--email", "a@b.c")
    assert code == 1 and out["code"] == "bad_args"
    code, out = run(capsys, "account", "logout", "mine")
    assert code == 0 and "mine" not in auth.credentials_path().read_text(encoding="utf-8")


def test_login_needs_a_password_on_standard_input(capsys, monkeypatch):
    _stdin(monkeypatch, "")
    code, out = run(capsys, "account", "login", "mine", "--email", "me@example.com")
    assert code == 1 and out["code"] == "bad_args"
    code, out = run(capsys, "account", "login", "mine")
    assert code == 1 and out["code"] == "bad_args"


class _Terminal(io.StringIO):
    def isatty(self):
        return True


def test_at_a_terminal_the_password_is_prompted_for_without_echo(capsys, monkeypatch, tmp_path):
    keys = []
    monkeypatch.setattr(auth.api, "login", lambda endpoint, key: keys.append(key) or "token")
    monkeypatch.setattr("sys.stdin", _Terminal("never read\n"))
    prompts = []

    def getpass(prompt):
        prompts.append(prompt)
        return "hunter2"

    monkeypatch.setattr(cli.getpass, "getpass", getpass)
    code, out = run(capsys, "account", "login", "mine", "--email", "me@example.com")
    assert code == 0 and prompts == ["Password: "], out
    # The same key as the password piped in gives
    _stdin(monkeypatch, "hunter2\n")
    code, out = run(capsys, "account", "login", "mine", "--email", "me@example.com")
    assert code == 0 and keys[0] == keys[1], out

    def closed(prompt):
        raise EOFError

    monkeypatch.setattr("sys.stdin", _Terminal())
    monkeypatch.setattr(cli.getpass, "getpass", closed)
    code, out = run(capsys, "account", "login", "mine", "--email", "me@example.com")
    assert (code, out["code"]) == (1, "bad_args") and "prompt" in out["message"]
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text('["account", "login", "mine", "--email", "me@example.com"]\n', encoding="utf-8")
    monkeypatch.setattr(cli.getpass, "getpass", getpass)
    code, lines = run_batch(capsys, str(jobs))
    assert lines[0]["code"] == "bad_args" and "password" in lines[0]["message"] and len(prompts) == 1


def _logged_in(monkeypatch, expires):
    store = {
        "current": "mine",
        "logins": {"mine": {"email": "m@e.c", "access_key": "KEY", "token": "old", "expires": expires}},
    }
    auth._save(store)
    renewed = []

    def login(endpoint, key):
        renewed.append(key)
        return "new"

    monkeypatch.setattr(auth.api, "login", login)
    return renewed


def test_login_token_renewed_before_it_runs_out(monkeypatch):
    cfg = config.load(config.default_path())
    renewed = _logged_in(monkeypatch, (NOW + timedelta(days=10)).isoformat())
    assert auth.credentials(cfg, None, NOW).token == "old" and renewed == []
    renewed = _logged_in(monkeypatch, (NOW + timedelta(hours=3)).isoformat())
    assert auth.credentials(cfg, None, NOW).token == "new" and renewed == ["KEY"]
    # Saved, so the next command does not sign in again
    assert auth.credentials(cfg, None, NOW).token == "new" and renewed == ["KEY"]


def test_login_that_cannot_be_renewed_is_an_auth_error(monkeypatch):
    cfg = config.load(config.default_path())
    _logged_in(monkeypatch, NOW.isoformat())

    def refused(endpoint, key):
        raise errors.NaiError(errors.AUTH, "refused")

    monkeypatch.setattr(auth.api, "login", refused)
    with pytest.raises(errors.NaiError) as e:
        auth.credentials(cfg, None, NOW)
    assert e.value.code == errors.AUTH and "nai account login" in e.value.message


def test_unmeasured_relay_spend_is_unknown(capsys, with_token, monkeypatch):
    _add_config('[accounts.relay]\ntoken = "t"\napi_url = "https://relay.example"\n')

    def unmeasured(self):
        self.calls.append("account")
        return api.UNMEASURED

    monkeypatch.setattr(FakeClient, "account", unmeasured)
    code, out = run(capsys, "--account", "relay", "gen", "--prompt", "x", "--size", "1216x1216")
    # The balance cannot be checked, so the request goes ahead and its spend stays unknown
    assert code == 0 and out["cost"]["spent"] is None and out["cost"]["balance_known"] is False
    # Priced without Opus, since the subscription cannot be read
    estimate = out["cost"]["estimate"]
    assert estimate > 0 and not out["cost"]["free_by_opus"]
    code, out = run(capsys, "stats", "--days", "1")
    assert out["requests_with_unknown_spend"] == 1


def test_batch_jobs_choose_their_account(capsys, with_token, tmp_path, monkeypatch):
    _add_config('[accounts.relay]\ntoken = "relay-token"\napi_url = "https://relay.example"\n')
    clients = fake_clients(monkeypatch)
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text('["--account", "relay", "gen", "--prompt", "a"]\n["gen", "--prompt", "b"]\n', encoding="utf-8")
    code, lines = run_batch(capsys, str(jobs))
    assert code == 0 and [line["exit"] for line in lines[:-1]] == [0, 0]
    # The second job goes back to the account the batch runs with
    assert [c.endpoint.official for c in clients if "generate" in c.calls] == [False, True]


def test_an_expiry_without_a_time_zone_is_renewed(monkeypatch):
    cfg = config.load(config.default_path())
    renewed = _logged_in(monkeypatch, "2030-01-01T00:00:00")
    assert auth.credentials(cfg, None, NOW).token == "new" and renewed == ["KEY"]


@pytest.mark.parametrize("text", ['{"logins": [1, 2]}', "[1]", '{"logins": {"a": 1}, "current": 3}'])
def test_a_malformed_login_store_is_refused_not_written_over(capsys, text):
    path = auth.credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    for argv in (["account", "list"], ["account", "use", "default"]):
        code, out = run(capsys, *argv)
        malformed = text != '{"logins": {"a": 1}, "current": 3}'
        assert (code, out.get("code")) == ((1, "config_missing") if malformed else (0, None)), (text, argv, out)
    if text.startswith('{"logins": [') or text == "[1]":
        assert path.read_text(encoding="utf-8") == text


def test_a_renewal_cut_off_by_the_network_keeps_a_token_still_valid(monkeypatch):
    cfg = config.load(config.default_path())

    def down(endpoint, key):
        raise errors.NaiError(errors.NETWORK, "connection reset")

    _logged_in(monkeypatch, (NOW + timedelta(hours=3)).isoformat())
    monkeypatch.setattr(auth.api, "login", down)
    assert auth.credentials(cfg, None, NOW).token == "old"
    _logged_in(monkeypatch, (NOW - timedelta(hours=1)).isoformat())
    monkeypatch.setattr(auth.api, "login", down)
    with pytest.raises(errors.NaiError) as e:
        auth.credentials(cfg, None, NOW)
    assert e.value.code == errors.NETWORK and "nai account login" not in e.value.message


@pytest.mark.parametrize(
    ("table", "said"),
    [
        ('[accounts.default]\ntoken = "t"\napi_url = "https://relay.example"\n', "novelai_token"),
    ],
)
def test_tables_named_as_what_is_built_in_are_refused(capsys, table, said):
    _add_config(table)
    code, out = run(capsys, "gen", "--prompt", "x", "--dry-run")
    assert (code, out["code"]) == (1, "config_missing") and said in out["message"], out


def test_config_and_models_show_every_section(capsys, tmp_path):
    _add_config(f'[local]\ntagger = "{(tmp_path / "wd.onnx").as_posix()}"\n[defaults]\neffort = "medium"\n')
    code, out = run(capsys, "config")
    assert code == 0 and out["local"]["tagger"] == str(tmp_path / "wd.onnx") and out["local"]["tagger_kind"] is None
    code, out = run(capsys, "models")
    v5 = next(m for m in out["models"] if m["id"] == "nai-diffusion-5-full")
    assert code == 0 and out["default"] == "nai-diffusion-5-full-medium" and v5["effort"] == "high"
    assert {"legacy_uc", "default_quality", "default_uc_preset"} <= set(v5)


def _expiring_login(monkeypatch, left):
    """An email login chosen with nai account use whose token runs out after left; counts the renewals."""
    renewed = _logged_in(monkeypatch, (datetime.now(UTC) + left).isoformat(timespec="seconds"))
    return renewed, auth.credentials_path().read_bytes()


def _dry_rows(tmp_path):
    img = tmp_path / "a.png"
    nai_image(img, w=512, h=512)
    art = tmp_path / "art.png"
    Image.fromarray(pixel_art(16, 12, 8)[1], "RGBA").save(art)
    region = "rect:0.1,0.1,0.5,0.5"
    return {
        "gen": ["gen", "--prompt", "x", "--dry-run"],
        "i2i": ["i2i", "--image", str(img), "--prompt", "x", "--dry-run"],
        "inpaint": ["inpaint", "--image", str(img), "--region", region, "--prompt", "x", "--dry-run"],
        "enhance": ["enhance", "--image", str(img), "--dry-run"],
        "variations": ["variations", "--image", str(img), "--dry-run"],
        "upscale": ["upscale", str(img), "--dry-run"],
        "tool": ["tool", "lineart", str(img), "--dry-run"],
        "pixel-snap": ["tool", "pixel-snap", str(art), "--dry-run"],
        "vibe": ["vibe", str(img), "--dry-run"],
        "cost": ["cost"],
    }


@pytest.mark.parametrize("left", [timedelta(hours=1), timedelta(hours=-1)], ids=["about to run out", "run out"])
def test_no_dry_run_renews_or_writes_a_login(capsys, tmp_path, monkeypatch, left):
    clients = fake_clients(monkeypatch)
    renewed, before = _expiring_login(monkeypatch, left)
    rows = _dry_rows(tmp_path)
    for name, argv in rows.items():
        code, out = run(capsys, *argv)
        assert code == 0, (name, out)
        if left < timedelta(0) and "cost" in out and name != "pixel-snap":
            # The token cannot be used, so the price assumes Opus and says why
            assert out["cost"].get("assumed_opus") and any("Opus" in w for w in out["warnings"]), (name, out)
    # Typed as --account, and from inside a batch, which reads the balance once
    run(capsys, "--account", "mine", "gen", "--prompt", "x", "--dry-run")
    jobs = tmp_path / "jobs.jsonl"
    argvs = [[a for a in argv if a != "--dry-run"] for argv in rows.values()]
    argvs += [["account", "show"], ["account", "use", "default"]]
    jobs.write_text("\n".join(map(json.dumps, argvs)), "utf-8")
    reads = sum(c.calls.count("account") for c in clients)
    code, lines = run_batch(capsys, str(jobs), "--dry-run")
    failed = [line["job"] for line in lines[:-1] if not line["ok"]]
    # Read once for every price and for nai account show
    assert sum(c.calls.count("account") for c in clients) - reads == (1 if left > timedelta(0) else 0)
    if left < timedelta(0):
        # A token run out is not renewed to read it, so nai account show cannot show the balance
        assert code == 1 and failed == [lines[len(argvs) - 2]["job"]], lines
        assert "run out" in lines[len(argvs) - 2]["message"]
    else:
        assert code == 0 and failed == []
    code = cli.main(["grid", "--dry-run", "--x", "steps=20,28", "gen", "--prompt", "x"])
    assert code == 0 and json.loads(capsys.readouterr().out.splitlines()[0])["dry_run"]
    assert renewed == [] and auth.credentials_path().read_bytes() == before
    assert not any(set(c.calls) - {"account"} for c in clients)


def test_a_real_run_renews_the_same_login(capsys, tmp_path, monkeypatch):
    # The control: the login the dry runs leave alone is one a real request renews
    fake_clients(monkeypatch)
    renewed, before = _expiring_login(monkeypatch, timedelta(hours=1))
    code, _ = run(capsys, "gen", "--prompt", "x")
    assert code == 0 and renewed == ["KEY"] and auth.credentials_path().read_bytes() != before


def test_edits_keep_what_this_version_cannot_read(capsys, monkeypatch):
    store = {
        "current": "mine",
        "logins": {
            "mine": {"email": "m@e.c", "access_key": "KEY", "token": "old", "expires": "2099-01-01T00:00:00+00:00"},
            "later": {"email": "l@e.c", "passkey": {"kind": "something newer"}},
        },
        "version": 2,
    }
    auth._save(store)
    cfg = config.load(config.default_path())
    for edit in (lambda: auth.use("default", cfg), lambda: auth.use("mine", cfg), lambda: auth.logout("mine")):
        edit()
        saved = json.loads(auth.credentials_path().read_text(encoding="utf-8"))
        assert saved["logins"]["later"] == store["logins"]["later"] and saved["version"] == 2
    assert "mine" not in saved["logins"]
    # One it cannot use is no login to sign out of
    assert auth.logout("later") is False and "later" in json.loads(auth.credentials_path().read_text("utf-8"))["logins"]


def test_a_renewal_keeps_what_another_process_wrote_meanwhile(monkeypatch):
    cfg = config.load(config.default_path())
    _logged_in(monkeypatch, (NOW + timedelta(hours=3)).isoformat())

    def login(endpoint, key):
        # Another nai signs in to a second account while this one waits for the server
        auth._save(
            {**json.loads(auth.credentials_path().read_text("utf-8")), "current": "other"}
            | {
                "logins": {
                    **json.loads(auth.credentials_path().read_text("utf-8"))["logins"],
                    "other": {
                        "email": "o@e.c",
                        "access_key": "OTHER",
                        "token": "t",
                        "expires": "2099-01-01T00:00:00+00:00",
                    },
                }
            }
        )
        return "new"

    monkeypatch.setattr(auth.api, "login", login)
    assert auth.credentials(cfg, "mine", NOW).token == "new"
    saved = json.loads(auth.credentials_path().read_text("utf-8"))
    assert saved["current"] == "other" and set(saved["logins"]) == {"mine", "other"}
    assert saved["logins"]["mine"]["token"] == "new"


def test_a_renewal_does_not_bring_back_a_login_signed_out_meanwhile(monkeypatch):
    cfg = config.load(config.default_path())
    _logged_in(monkeypatch, (NOW + timedelta(hours=3)).isoformat())

    def login(endpoint, key):
        auth.logout("mine")
        return "new"

    monkeypatch.setattr(auth.api, "login", login)
    assert auth.credentials(cfg, "mine", NOW).token == "new"
    assert json.loads(auth.credentials_path().read_text("utf-8"))["logins"] == {}


def test_an_edit_waits_for_another_holding_the_store(monkeypatch):
    cfg = config.load(config.default_path())
    _logged_in(monkeypatch, (NOW + timedelta(days=10)).isoformat())
    held = fs.acquire(auth.credentials_path().with_name("accounts.lock"), timeout=1, poll=0.01)
    timer = threading.Timer(0.3, fs.release, [held])
    timer.start()
    started = time.monotonic()
    auth.use("default", cfg)
    assert time.monotonic() - started >= 0.25 and auth.current() == "default"
    timer.join()


def test_an_edit_held_off_past_its_wait_is_busy(capsys, monkeypatch):
    _logged_in(monkeypatch, (NOW + timedelta(days=10)).isoformat())
    real = fs.acquire
    monkeypatch.setattr(auth.fs, "acquire", lambda path, timeout, poll: real(path, 0.2, poll))
    held = real(auth.credentials_path().with_name("accounts.lock"), timeout=1, poll=0.01)
    try:
        code, out = run(capsys, "account", "use", "default")
    finally:
        fs.release(held)
    assert (code, out["code"]) == (1, "busy") and auth.current() == "mine", out


def test_choosing_a_login_signed_out_meanwhile_is_refused(monkeypatch):
    cfg = config.load(config.default_path())
    _logged_in(monkeypatch, (NOW + timedelta(days=10)).isoformat())
    # Listed when checked, gone by the time the store is locked
    monkeypatch.setattr(auth, "names", lambda cfg: [auth.DEFAULT, "mine", "gone"])
    with pytest.raises(errors.NaiError) as e:
        auth.use("gone", cfg)
    assert e.value.code == errors.BAD_ARGS and auth.current() == "mine"


@pytest.mark.parametrize("hours", [-1, 1])
def test_batch_tags_never_renews_a_login(capsys, tmp_path, monkeypatch, hours):
    fake_clients(monkeypatch)
    renewed, before = _expiring_login(monkeypatch, timedelta(hours=hours))
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text('["tags", "cat"]\n', encoding="utf-8")
    code, lines = run_batch(capsys, str(jobs), "--dry-run")
    assert renewed == [] and auth.credentials_path().read_bytes() == before
    assert code == (1 if hours < 0 else 0)
    if hours < 0:
        assert lines[0]["code"] == errors.AUTH


@pytest.mark.parametrize("dry", [False, True])
def test_balance_names_the_account_it_was_read_from(capsys, with_token, monkeypatch, tmp_path, dry):
    _add_config('[accounts.other]\ntoken = "other-token"\n')
    read = FakeClient.account

    def switch(self):
        auth.use("other", config.load(config.default_path()))
        return read(self)

    monkeypatch.setattr(FakeClient, "account", switch)
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text('["account", "show"]\n["account", "show"]', encoding="utf-8")
    code, lines = run_batch(capsys, str(jobs), *(["--dry-run"] if dry else []))
    assert code == 0 and [line["account"] for line in lines[:-1]] == ["default", "other"]


def test_ledger_keeps_paying_account_across_a_switch(capsys, with_token, monkeypatch, tmp_path):
    _add_config('[accounts.other]\ntoken = "other-token"\n')
    ready = output.ready

    def switch(destination):
        auth.use("other", config.load(config.default_path()))
        ready(destination)

    monkeypatch.setattr(output, "ready", switch)
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text('["gen", "--prompt", "first"]\n["gen", "--prompt", "next"]', encoding="utf-8")
    code, _ = run_batch(capsys, str(jobs))
    assert code == 0 and [row["account"] for row in ledger.entries()] == ["default", "other"]
