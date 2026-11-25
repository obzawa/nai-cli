---
name: nai-cli
description: Generate and edit images with NovelAI through the nai command-line client. Covers text to image, image to image, inpainting (fixing hands or faces, changing a detail, extending the canvas), Enhance, Variations, upscaling, Director tools such as background removal, line art, colorize and emotion, Vibe encoding, reading the generation settings stored in an image, looking up tags, and checking Anlas costs and balance. Use when the user wants to create or modify images with NovelAI, rerun or tweak a NovelAI image, or asks what an image was generated with.
license: MIT
allowed-tools: Bash(nai:*)
compatibility: Requires the nai command (pipx install nai-cli), a NovelAI persistent API token and network access to image.novelai.net.
---

# nai-cli

`nai` sends NovelAI image requests identical to the official web client's. Every command prints one JSON object to standard output, except `nai batch`, `nai grid` and `nai meta` of several images or a folder, which print one JSON line each and a summary line last. Run `nai COMMAND --help` for options instead of guessing flags.

## Results

- Success has `ok: true`. Failures exit non-zero with `{ok: false, code, message, ...}`. Branch on `code`; messages are written for people and may change.
- Generation results list the saved files in `images`, plus `seed`, `size`, `cost`, `lint` and `warnings`. `prompt` and `negative` are the text actually sent, with the automatic quality tags and UC preset already added.
- Pass `warnings` and any `lint` issues of level `error` or `warning` on to the user.

## Before spending Anlas

1. Run a new prompt or command with `--dry-run` first. It sends nothing paid and costs nothing.
2. `cost.estimate` is the price in Anlas; 0 means free with Opus. Mention the price when it is not 0.
3. `need_confirm` means the estimate exceeds the user's per-request limit. Show the `cost` to the user and add `--yes` only after they agree.
4. Never put the API token on the command line or in output.

## Common tasks

```sh
nai gen --prompt @prompt.txt --neg @negative.txt --size portrait --dry-run
nai i2i --image in.png --prompt @prompt.txt --strength 0.5
nai inpaint --image in.png --region ellipse:0.62,0.40,0.80,0.58 --prompt @prompt.txt --preview
nai enhance --image in.png --dry-run
nai meta in.png
nai account
nai stats --by command
```

- Write prompts to a file and pass `@FILE`, or pipe one in with `@-`. This avoids shell quoting problems with quotes, `::` weights, line breaks and prompts that start with `-`. A prompt given on the command line that itself starts with `@` is written with `@@`; text in the file is sent as it is.
- To rerun or adjust an existing image, run `nai meta IMAGE` and pass its `gen_args` to `nai gen`, followed by the changes. Later single-value options override earlier ones; repeatable ones such as `--char` add another entry. `gen_args` includes the original `--seed`, so pass a new one for a different image. Leave its other switches, such as `--neg-verbatim` and `--no-format-prompt`, in place: they make the rerun match. `warnings` from `nai meta` lists what the image does not keep, such as precise references; tell the user the rerun will differ there.
- To start from an image, use `--from-image IMAGE`; it brings in the prompts, the characters and the Vibes the image keeps when the model made them. Add `--import PARTS`, such as `--import all` or `--import prompt,settings`, for more; the seed is left out unless `seed` is named. `--actual-prompts` takes what the image drew from `||a|b||` groups.
- `enhance` and `variations` start from the settings stored in the image; options given override them, empty ones included. Turn off a switch the image has on with its negated form, such as `--no-furry`. They usually cost Anlas, since enhance sends more than 1 MP and variations several images.
- `--account NAME` runs a command with another account of the user's; `nai account list` names them. Never ask for or handle the password of `nai account login`: the user types it.
- `-o NAME.png` or `-o FOLDER` saves the images where a caller expects them; a name that is taken gets `-1`, `-2` and so on, so read the saved names from the output.
- `nai lint` checks a prompt without pricing or sending it. `tokens` counts it with the official tokenizers against the model's `limit`; `exact: false` means the counts are estimates.
- `nai tags QUERY...` suggests tags for the start of a tag; use it to check that a tag exists and how it is spelled before writing it into a prompt. Lookups are free.
- `nai models` lists each model's limits, supported features and the values `--quality` and `--uc-preset` accept.

## More tasks

Read the reference when the task comes up:

- Repainting part of an image or extending the canvas: [references/inpaint.md](references/inpaint.md).
- Many images from one file, or a sheet comparing settings: [references/batch.md](references/batch.md).
- Vibes, which V5 does not take, and precise references: [references/vibes.md](references/vibes.md).

Smaller ones:

- Before the user shares an image, `nai strip IMAGE` saves a copy without its prompt and settings. `nai redact IMAGE --region rect:x1,y1,x2,y2` pixelates part of it and `nai watermark IMAGE --text TEXT` signs it; both drop the settings too. Copies never replace the source.
- `nai tool pixel-snap IMAGE` turns AI pixel art into clean pixels, one per cell, for free; add `--upscale` for a viewable size.
- A prompt written for ComfyUI with `COUPLE`, `AND ... MASK(...)` or `|` lines splits into characters with `--comfyui-characters`.
- With the local extra installed (`nai local` shows it), `--region object:x,y` masks the object under a point for `inpaint` and `redact`, and `nai interrogate IMAGE` reads an image's tags into a prompt.
- When the user wants a surprise, `nai gen --random-prompt` draws a prompt and characters as the official dice button does; the result lists them.
- `--furry` is the official furry mode: the prompt starts with `fur dataset, `.

## Errors

| `code` | What to do |
|---|---|
| `need_confirm` | Show `cost`; rerun with `--yes` only after the user agrees. |
| `ledger` | The spend ledger or the generation lock cannot be used. Tell the user the path in `message`; nothing was sent. |
| `batch_failed` | Read each job's own line for its `code`, and handle it as listed here. |
| `model_unsupported` | Tell the user which `feature` the model lacks and let them choose another model or drop the feature. Never switch models silently. |
| `bad_args` | Correct the arguments using `message`, `choices` or `usage`. |
| `config_missing` | Ask the user to set `NOVELAI_TOKEN`, or to run `nai config init` and fill in `novelai_token`. |
| `empty_mask` | The area selects nothing. Enlarge the region or check the mask. |
| `no_metadata` | The image has no NovelAI settings. Ask the user for a prompt. |
| `auth`, `insufficient_anlas` | Report to the user. Do not retry. |
| `save_failed` | The result could not be written. After a paid request it ran and was charged: tell the user `cost.spent` and the files in `saved`; do not run it again without asking, even when `cost.spent` is 0. Free commands such as `pixel-snap` and `local download` have no `cost`. Fix the output folder before running it again. |
| `rate_limited`, `busy` | Wait briefly and retry once, then report. |
| `server`, `network` | Without `maybe_processed`, wait briefly and retry once, then report. With `maybe_processed: true` the request may have run and been charged: run `nai account`, tell the user how `anlas_left` changed, and send it again only if they agree. |
| `api` | NovelAI refused the request or sent an unusable reply. Report `message`; do not retry it unchanged. With `maybe_processed: true`, check the balance as for `server`. |
| `interrupted` | The user stopped the command. Report `cost.spent` and the files in `saved` if present; do not retry unless asked. |
| `internal` | A bug in nai-cli. Report `message` to the user; do not retry. |

Whatever the code, a `cost.spent` above 0 means Anlas were already spent, for example on a Vibe encoded before the generation failed. Tell the user before retrying, and list any files in `saved`.
