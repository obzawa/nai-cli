# nai-cli

English | [中文](README.zh.md)

nai-cli is an unofficial command-line client for [NovelAI](https://novelai.net) image generation. It covers text to image, image to image, inpainting, Enhance, upscaling, the Director tools, Vibes and precise references on V5, V4.5 and V4, and reads the settings stored in images so they can be generated again.

It builds requests by the same rules as the official web client, so the same settings and seed produce the same image; the [parity notes](docs/official-parity.md#intentional-differences) list the exceptions. Every command prints JSON, one object or one line per job, which makes it easy to call from scripts and other programs.

Documentation: [usage guide](docs/usage.md), [official client parity](docs/official-parity.md), [compatibility](docs/usage.md#compatibility)

## Install

Python 3.11 or later is required:

```sh
pipx install nai-cli
```

This installs the `nai` command in an environment of its own. `uv tool install nai-cli` and `pip install nai-cli` work as well. Reading tags from images and masking objects run on your computer and need `pipx install "nai-cli[local]"`.

## Configure

Create a persistent API token in NovelAI under Account Settings (it starts with `pst-`), then create a config file:

```sh
nai config init
```

The command prints the path of the new file. Put the token in its `novelai_token` field, or set the `NOVELAI_TOKEN` environment variable instead. Once the token works, `nai account` shows your balance.

## Usage

```sh
nai gen --prompt "1girl, rooftop, night" --dry-run
nai gen --prompt @prompt.txt --size landscape
nai inpaint --image a.png --region ellipse:0.62,0.40,0.80,0.58 --prompt "1girl, rooftop, night, open hand" --preview
nai enhance --image a.png
nai meta a.png
nai batch jobs.jsonl --dry-run
```

`--dry-run` builds and prices the request without sending anything paid. The `prompt` and `negative` fields of the output hold the final text, including the quality tags and UC preset added automatically; without `--dry-run`, the saved files are listed in `images`. `@prompt.txt` reads the prompt from a file. `inpaint` repaints an area given in fractions of the image and composites the result back onto it, and `--preview` saves an image marking that area before anything is sent. `enhance` upscales and refines an image with the settings stored in it, and `meta` reads those settings so that `nai gen` can reproduce the image. `batch` runs a list of commands, one per line, and prints a JSON line for each, so other programs can queue work and resume it after a failure.

`nai --help` lists all commands, and `nai COMMAND --help` shows their options. The [usage guide](docs/usage.md) covers each command, the output format and the error codes.

## Agent skill

nai-cli includes a skill in the [Agent Skills](https://agentskills.io) format that teaches AI agents to use it: dry-running before spending Anlas, asking before exceeding the spending limit, recovering from errors and repainting part of an image. Install it with:

```sh
npx skills add obzawa/nai-cli
```

This installs it for the current project; add `-g` to install it for all projects. The skill itself is in [skills/nai-cli](skills/nai-cli).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setting up, the checks a change must pass and how to report bugs; coding agents follow [AGENTS.md](AGENTS.md).

## Acknowledgements

Thanks to [NovelAI](https://novelai.net) for the models and the service. Many of the defaults and conveniences follow [NAI Launcher](https://github.com/Aaalice233/Aaalice_NAI_Launcher), and image resizing is ported from [Pica](https://github.com/nodeca/pica).

## License

[MIT](LICENSE)

Third-party code and its licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
