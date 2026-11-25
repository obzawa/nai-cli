# Architecture

nai-cli is one core and a thin front end. The core knows NovelAI: how requests are built, what images carry, what things cost. The front end knows the command line: options, standard input, JSON on standard output and exit codes. Commands sit between them and decide what happens in which order. A Python API or another front end later reuses the commands unchanged.

## Layers

Each layer imports only from the layers below it. `tests/test_architecture.py` enforces the order, the core's limits below and the rule that only `fs.py` writes files.

| Layer | Modules | Knows about |
|---|---|---|
| Front end | `cli.py`, `parser.py`, `__main__.py` | argv, standard input, JSON output, exit codes |
| Commands | `commands/*.py` | the steps of one command: read inputs, check prices and budget, call the core, save, report |
| Orchestration | `charge.py`, `batch.py`, `grid.py`, `import_image.py`, `caches.py` | paid-request lifecycle, resumable job runs and grids of them, the official import assignment chain, the tokenizer files and Vibe encodings kept between runs |
| Core | `request.py`, `prompt.py`, `tidy.py`, `models.py`, `presets.py`, `meta.py`, `rerun.py`, `vibe.py`, `inpaint.py`, `enhance.py`, `cost.py`, `naming.py`, `tokenizer.py`, `lint.py`, `tags.py`, `resample.py`, `images.py`, `share.py`, `random_prompt.py`, `sheet.py`, `pixel_snap.py`, `comfyui.py`, `tagger.py`, `sam.py` | NovelAI rules as plain functions over plain data |
| I/O | `api.py`, `auth.py`, `fs.py`, `ledger.py`, `config.py`, `local_models.py`, `fonts.py`, `decode.py`, `values.py`, `errors.py` | network, accounts and logins, disk, the spend ledger, the config file, local model files, installed fonts, reading image bytes, values as files and options give them, error codes |

The core never reads argv, prints, or decides where files go. It reads the files it is given and the data shipped inside the package, but never looks for files, talks to the network, loads models, reads the config, the data folder or the ledger, or writes; of the I/O layer it imports only `errors`, `values` and `decode`. Whatever it needs from outside, such as a tokenizer or a local model, is loaded above it and passed in. A function that needs a decision from the user takes it as an argument.

### Front end

`parser.py` declares every option and nothing else: name, type, default, help. The same declarations feed `--help`, docs/usage.md checks and the skill check, so an option exists in one place.

`cli.py` turns argv into options, options into settings, runs the command and prints its result. Nothing else.

### From argv to settings

Arguments go through two stages, and commands only ever see the second.

1. **Options**: what was typed, as parsed by `parser.py`. Unset options are `None`, so "not given" and "given as false or empty" stay different; every on/off option has a `--no-` form.
2. **Settings**: one immutable dataclass per command, resolved once. Resolution reads `@FILE` and `@-` for the options typed on the command line, merges the config file, imports from an image, picks the model and applies its limits. Precedence is fixed: command line, then imported image settings where the command imports them, then the config file, then built-in defaults.

Commands never read the config file, never look at options and never store state on them. Text that comes from an image or a job file is data; it never passes through `@` resolution.

### Context

A command gets its outside world from one context object: the config, loaded once, a client created when first needed, the clock, standard input, the argv for the ledger and whether this is a dry run. The front end builds it once per run. Each `batch` job gets a child context that shares the config, so the batch's limits hold, has its own client and refuses standard input; a job may name its own `--account`, and `--config` in a job is refused. The shared part also keeps the spacing of requests between jobs, how jobs are parsed, and the balance a dry run of several jobs reads once per account. `config init` clears the loaded config so later jobs read the new file. A dry run never renews a login: its balance reads use the stored token as it is.

### Commands

One file per command family under `commands/`: generation (`gen`, `enhance`, `variations`, `inpaint`), `tool`, `upscale`, `vibe`, `meta`, `grid`, `strip`, `redact`, `watermark` (sharing copies in `copies.py`), `interrogate`, `local`, `cost`, `config`, `batch`, `account` and `stats`.

Each command is a settings dataclass, a `resolve(options, ctx)` function from the parsed options, as a plain dict, to settings, and a `run(settings, ctx)` function that returns a result. `run` never prints and never calls `sys.exit`, so `batch` and future front ends call it in process. Flag parsing is tested through `resolve`, behaviour through `run` with a fake context.

`--dry-run` is a field of the settings, and for `batch` a field of the context, never an extra argv word. Paid commands prepare and price the request without sending it. If the balance cannot be read, the estimate assumes Opus; commands that return account data or look up tags still need a valid token. No output or login changes persist, apart from the tokenizer cache on first use. To find what the file system refuses, it makes the folders and the file it would write and removes again only the ones it made.

A dry run of several jobs, `batch --dry-run`, `grid --dry-run` and the check a grid makes before its first cell, runs each job as its own dry run. No job creates outputs for another, so it cannot use artifacts of earlier jobs in the dry run; predicting that would mean simulating the server, the ledger and the disk.

## One owner per rule

Every rule lives in one place. A second implementation of a rule drifts from the first unnoticed, so review checks this first.

| Rule | Owner |
|---|---|
| Request body fields, order and defaults | `request.py` |
| Prompt text: quality tags, UC presets, automatic text, literal ranges, weight syntax | `prompt.py` |
| Prompt tidying the user switches on: commas and spaces, Stable Diffusion weights, numeric closings | `tidy.py` |
| What each model supports | `models.py` |
| Prices, tiers and free allowances | `cost.py` |
| File name candidates from templates and folders | `naming.py` |
| Every write of a file: collision numbering, never overwriting, atomic replacement, file locks | `fs.py` |
| Text a request can carry | `values.py` |
| Settings stored in images and how they are read | `meta.py` |
| Arguments that rerun an image, and what they cannot bring back | `rerun.py` |
| Where the tokenizer files and Vibe encodings are kept, and the tokenizer download | `caches.py` |
| How imported settings land on a target model | `import_image.py` |
| Accounts, logins and token renewal, edits of the account store under its lock | `auth.py` |
| Balance before and after, the per-request limit, ledger entries | `charge.py`, `ledger.py` |
| Successful jobs and resuming | `batch.py` |
| Grid axes and their cells | `grid.py` |
| Waiting between batch jobs | `commands/batch.py` sets it, `commands/context.py` waits |
| Redaction and watermark drawing | `share.py` |
| Installed fonts offered when the built-in one lacks glyphs | `fonts.py` |
| Names and writing of sharing copies | `commands/copies.py` |
| Local ONNX models: loading, the SAM files and their hashes, the optional install | `local_models.py` |
| Error codes | `errors.py` |

Callers ask the owner. A pre-check that a file name is free calls the same function that will later name the file; it never rebuilds the name.

## Code rules

- **One concept per file, named after it.** Split by concept, never by size: a long file about one thing is fine; a second concept is the reason to split. No grab-bag `utils.py`.
- **Functions do one job.** About 40 lines needs a reason. Order inside a file: imports, public types, the public entry point, then its helpers in the order they are used.
- **No boolean switches between two behaviours.** If a flag changes what a function does rather than a detail of it, write two functions or pass an enum.
- **States are explicit.** Job states, import parts, charge outcomes and the like are enums or dataclasses, never loose strings or `None` meaning something.
- **Validate at trust boundaries only.** Command-line options, image metadata, server responses, job files and config files are checked where they enter. Code past that point trusts its inputs and carries no defensive branches for states that cannot happen.
- **Production code has no test hooks.** No branches that detect monkeypatching, no test-only modes or exports. Tests replace dependencies at a seam the code already has: the client, the clock, `fs`, `os`.
- **No compatibility shims before 1.0** unless a decision asks for one.
- **Never write the same lines twice.** Shared behaviour goes to the owner of the rule, not to a new helper next to one caller.
- **Top-level imports only.** No imports inside functions.
- **Comments say why**, in one line. A file starts with one line saying what it is for.

## Contract with callers

Scripts and agents depend on this, so it changes only with a decision recorded in the changelog.

- Standard output carries results only: one JSON object per command. Streaming commands (`batch`, `grid`, multi-file `meta`) print one JSON line per item and end with a summary line; every line has a `type` field (`job` or `summary` for `batch`) so a reader can dispatch without guessing from the keys. Logs and progress go to standard error.
- Field names and types in JSON results are part of the contract, including what an empty result looks like.
- Exit code 0 means success. Error codes in `errors.py` are public and are never renamed or removed.
- Files are created, never overwritten. A name that is taken gets the next free number, including one given with `-o`. A Vibe export probes its requested name before encoding; generated images fall back to a default name if the requested one cannot be written.
- A paid request is settled even when it fails or is interrupted: the result is saved if it arrived, the spend is read from the balance and written to the ledger.
- The same inputs and seed produce the same request as the official client. Deliberate differences are listed in [official-parity.md](official-parity.md#intentional-differences).

## Definition of done

A change is done when every item below holds. Review goes through them in order and stops at the first one that fails.

**Checks**

- `ruff check`, `ruff format --check` and `pytest` pass locally and in CI on Windows, macOS and Linux.
- No existing test was deleted or weakened to make the change pass. A test that changes because behaviour changed on purpose is named in the change description with the reason.

**Structure, enforced by `tests/test_architecture.py`**

- Imports follow the layer table: no module imports from a layer above it.
- The core imports nothing of the I/O layer but `errors`, `values` and `decode`.
- Only `fs.py` writes to disk: no `open` in a write mode, `Path.open`, `os.open`, `write_bytes`, `write_text`, `Image.save` to a path, `shutil` copies or moves, `os.replace` or `rename` anywhere else.
- Only `cli.py` touches `sys.stdin`, `sys.stdout`, `print` and `sys.exit`.
- Only the context reads the config file and creates the client: one call site each.
- No module under `commands/` imports `argparse` or `parser`.
- No imports inside functions.

**Structure, checked in review**

- Each rule touched by the change has one implementation, in the owner the table names. Review greps for the rule's other spellings (for file names: `exists()`, `-{n}`, `_{index}`) and finds none outside the owner.
- No state is stored on parsed options; commands receive settings.
- Functions longer than about 40 lines, other than option declarations in `parser.py`, are listed in the change description with the reason.
- No test hooks, compatibility aliases or duplicate constants in `src/`.

**Behaviour**

- Each fixed defect has a test that fails before the fix.
- New behaviour is tested at its seam and in the combinations listed under "Adding a feature".
- A dry run and the real run of the same command take the same path up to the first paid request.

**Documentation**

- usage.md, official-parity.md and `skills/nai-cli` describe the new behaviour; `tests/test_skill.py` passes.

## Adding a feature

1. Decide which layer it belongs to, and which existing owners it needs.
2. Put new rules in their owner. If no owner fits, the feature introduces a new concept and gets a new module.
3. Add the command under `commands/` and its options in `parser.py`.
4. Test each thing at its seam, once: a rule at its module, option handling through `resolve`, behaviour through `run` with a fake context, the JSON contract including the empty case. Then the combinations: inside batch, run a second time, and the next request after a failure.
5. Update [usage.md](usage.md), [official-parity.md](official-parity.md) if it touches request behaviour, and `skills/nai-cli`.
