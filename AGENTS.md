# AGENTS.md

nai-cli is a command-line client for NovelAI image generation whose requests must match the official web client. [docs/official-parity.md](docs/official-parity.md) records every rule and where nai-cli differs on purpose.

## Commands

```sh
pip install -e ".[dev]"
pytest
ruff check
ruff format
```

Run all of them before finishing a change. CI runs the same checks on Linux, Windows and macOS, each with Python 3.11 to 3.14. Each also runs `python .github/scripts/smoke.py tokenizers`, which calls the installed `nai` once per offline feature in a throwaway home with the network blocked; add a case there for a new command, which `tests/test_docs.py` checks.

## Rules

- [docs/architecture.md](docs/architecture.md) sets the layers, the owner of each rule and the code rules. Read it before changing structure; review checks changes against it.
- The official web client's code is the only reference for request behaviour. Images made with other clients are test data, not a source of rules. Update official-parity.md whenever a rule changes.
- A deliberate difference from the official client needs a row under "Intentional differences". If it changes the generated image, it also needs a switch that restores the official behaviour where possible.
- Pixel processing in `resample.py`, `inpaint.py` and the precise reference canvas in `images.py`, and token counting in `tokenizer.py` are identical to the official client. The digests in `test_resample.py` and `test_inpaint.py` and the token ids in `test_tokenizer.py` come from the official code; never regenerate them from nai-cli's own output.
- Every command prints exactly one JSON object, except `nai batch`, `nai grid` and `nai meta` of several images or a folder, which print one JSON line per job or image and a summary line with `"type": "summary"` last. Error codes are a public interface: do not rename or remove them, and document new ones in `errors.py` and docs/usage.md.
- Tests never contact NovelAI. Use `--dry-run`, or the `with_token` fixture and the fake client in `tests/helpers.py`. `tests/conftest.py` blocks tokenizer downloads, so counts in tests are estimates unless a test supplies a tokenizer file.
- `tests/private/` holds generation settings from real images. It is not tracked; never commit it or quote from it.
- Code, comments, messages and documentation are in English. README.zh.md mirrors README.md.
- Comments explain why, not what.
- When commands or flags change, update docs/usage.md and `skills/nai-cli` as well. `tests/test_docs.py` fails when usage.md leaves out a command or option, `tests/test_skill.py` when the skill mentions a command or flag that does not exist.
- Keep the version in `pyproject.toml` and `src/nai_cli/__init__.py` in step.

## Releases

Record user-visible changes under `## Unreleased` at the top of CHANGELOG.md, adding the heading if it is missing. A pull request that changes `src/` fails the Changelog check unless it changes CHANGELOG.md too or carries the label `skip changelog`, for changes users do not meet. To release, rename that heading to `## X.Y.Z - YYYY-MM-DD`, set the same version in both files above, commit, and push the tag `vX.Y.Z`. The release workflow runs the full CI, checks the tag against the version and the changelog, builds and tests the package, publishes it to PyPI and creates the GitHub release with that changelog section. PyPI never accepts the same version twice, so fixing a published release takes a new version. If a release run fails partway, re-run only its failed jobs; re-running the whole workflow uploads to PyPI again and fails.
