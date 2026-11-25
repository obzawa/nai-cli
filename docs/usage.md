# Usage

This guide covers configuration, commands, output and the rules nai-cli applies to requests. Installation is described in the [README](../README.md); the rules checked against the official web client are listed in [Official client parity](official-parity.md).

- [Configuration](#configuration)
- [Commands](#commands)
- [Generation options](#generation-options)
- [Image to image and inpainting](#image-to-image-and-inpainting)
- [Enhance and Variations](#enhance-and-variations)
- [Automatic additions](#automatic-additions)
- [Prompt tidying](#prompt-tidying)
- [Checking prompts](#checking-prompts)
- [Tags and models](#tags-and-models)
- [Output](#output)
- [Grids](#grids)
- [Sharing copies](#sharing-copies)
- [Local models](#local-models)
- [Scripting](#scripting)
- [Pricing](#pricing)
- [Compatibility](#compatibility)

## Configuration

nai-cli needs a persistent API token, created in the NovelAI web client under Account Settings (it starts with `pst-`). Provide it in either of two ways:

- the `NOVELAI_TOKEN` environment variable, or
- the `novelai_token` field of the config file created by `nai config init`.

The environment variable takes precedence. The config file also sets the output directory, the spending limit per request and default generation parameters:

```toml
novelai_token = "pst-..."

[output]
dir = "D:/nai-out"
folder = "{date}"         # folders under dir, see File names; "" saves in dir
filename = ""             # see File names

[cost]
max_per_request = 100     # larger estimates return need_confirm

[batch]
interval = 0              # seconds between the requests of two jobs, or [min, max]; at most 86400

[defaults]                # all optional; command line > [defaults] > model default
model = "v5"
effort = "high"           # V5 Full at high or medium effort
size = "832x1216"
steps = 28
cfg = 4
sampler = "k_euler_ancestral"
cfg_rescale = 0
noise_schedule = "karras"
quality = "standard"
uc_preset = "heavy"
n = 1
official_nsfw = false     # see Automatic additions
comfyui_characters = false  # see ComfyUI prompts
v5_hidden_options = false # see Parameters
premultiplied_alpha = false
format_prompt = false     # see Prompt tidying
format_underscores = false
space_numeric_close = false
convert_sd_weights = false
prompt_prefix = ""        # fixed tags, see Prompt tidying
prompt_suffix = ""
negative_prefix = ""
negative_suffix = ""

[local]                   # see Local models
tagger = ""
tagger_labels = ""
tagger_kind = ""
sam_dir = ""

[accounts.relay]          # more accounts, see Accounts
token = "..."
api_url = "https://relay.example.com"

```

| | Windows | macOS and Linux |
|---|---|---|
| Config file | `%APPDATA%\nai-cli\config.toml` | `~/.config/nai-cli/config.toml` |
| Images (default), ledger, Vibe cache, tokenizer files | `%LOCALAPPDATA%\nai-cli\` | `~/.local/share/nai-cli/` |

**Accounts.** The token above is the account `default`, a name no `[accounts]` table can take. Tables under `[accounts.NAME]` add more: another `token` for the official service, or a relay that speaks the NovelAI API, with its own token, `api_url` and optionally `image_url` (`api_url` when left out). `nai account login NAME --email EMAIL` signs in with an email and password as the official client does, reading the password from standard input; only the derived access key and the 30-day token are kept, in `accounts.json` in the data directory, and the token is renewed from the key a day before it runs out; if the network or the server fails then, the token still valid is used and the next command tries again. `nai account use NAME` chooses the account for later commands, `--account NAME` for one command, `nai account list` lists them without their tokens and `nai account logout NAME` forgets a login. A relay that does not report a subscription is used with its balance unknown: estimates assume no Opus, the balance is not checked, and the spend is recorded as unknown.

On macOS and Linux, `XDG_CONFIG_HOME` and `XDG_DATA_HOME` are respected. `nai config` prints the current configuration, showing only whether a token is set and where it comes from. `--config PATH` selects another config file for one command.

**File names.** Images are saved as `HHMMSS_SEED.png` in a folder per day, with `_1`, `_2` and so on for further images of a request. `folder` under `[output]` chooses the folders under `dir` in the same syntax as file names, such as `"{year}/{month}"` or `"{model}/{date}"`: only `/` and `\` written in the rule make folders, folders that come out empty or as dots only are left out, and a rule that starts at a root or drive or uses `.` or `..` is refused, so the rule itself cannot name a place outside `dir`. Folders that already exist under `dir` are used as they are, so a folder that is a link saves where the link points. A folder given with `-o` takes no rule. `filename` under `[output]` names the images with the official client's template instead, such as `"{prompt} s-{seed}"` or `"{date} {modelname} {prompt|tags:3} {seed}"`. The tokens are the official client's `prompt`, `seed`, `date`, `time`, `model`, `modelname`, `resolution`, `width`, `height`, `index`, `id`, `type` and `characters`, and nai-cli's own `year`, `month`, `day` and the settings sent, `steps`, `cfg` and `sampler`, which the official client refuses; filters such as `|first:20`, `|tags:3`, `|words:5`, `|pad:4`, `|pre:-`, `|suf:-`, `|lower` and `|upper` follow a token, and `{characters}` also takes `|max:N`, `|at:N`, `|each:N`, `|join:TEXT` and `|full` before them. `{{` writes a literal `{`. As in the official client, `prompt` is the prompt as typed and cut to 80 characters unless a length filter is given, each image of a request gets its own seed, `type` is `image2image`, `inpaint`, `enhanced` or `variations` (empty for text to image), and upscales and Director tool results take the prompt and seed of their source image with the tool's name as `type`. As in the official client, those results are named by the template only when it has `{type}`, since every one would otherwise take its source's name; without it, and without a template, they are named `SOURCE_TOOL.png`, such as `pic_upscale.png`, `pic_lineart.png` or `pic_emotion_happy.png`. Characters the file system refuses become `_`, names stop at 180 characters, a name that is taken gets `-1`, `-2` and so on, and a name that comes out empty falls back to `PROMPT s-SEED`. A template with an unknown token or filter is reported as `config_missing` before anything is sent, and a name or folder that cannot be saved falls back to the default name `HHMMSS_SEED.png`, with no text from the prompt, the source or `-o`, directly under `dir` or the `-o` folder, with a warning.

## Commands

| Command | Description |
|---|---|
| `nai gen` | Text to image. |
| `nai i2i --image IMAGE` | Image to image with `--strength` and `--noise`. |
| `nai inpaint --image IMAGE` | Repaint the area given by `--mask`, `--region` or `--expand`, then composite the result back. |
| `nai enhance --image IMAGE` | Upscale and refine an image at low strength using its embedded settings. |
| `nai variations --image IMAGE` | Generate similar images using an image's embedded settings. |
| `nai upscale IMAGE` | Upscale an image of up to 3145728 pixels in area, such as 1536×2048. |
| `nai tool TOOL IMAGE` | Director tools: `bg-removal`, `declutter`, `declutter-keep-bubbles`, `lineart`, `sketch`, `colorize`, `emotion` and `pixel-snap`. Background removal saves three images, named masked, generated and blend. `pixel-snap` cleans up pixel art: it finds the grid of the scaled pixels, takes the median colour of each cell and reduces the palette, with `--colors auto|off|1-256`, `--avoid-over-refining` and `--upscale`. It runs on this computer, as it does in the browser, so it is free; a large image takes up to a minute. |
| `nai vibe SOURCE...` | Save Vibes for V4.5 or V4 as the official client exports them: one as a `.naiv4vibe` file, several as a `.naiv4vibebundle` named by the date. A source is an image to encode (2 Anlas, nothing when it was encoded with the same settings before) or a Vibe file or image carrying Vibes, each with optional `:STRENGTH:INFO`. `--no-image` keeps only the encodings, `--into PNG` writes the bundle into a copy of that image, and `-o` chooses the file; a name that is taken gets the next number. `--model` picks the model to encode for (default `v4.5`), and `--strength` and `--info` set what a source leaves unset (0.6, and 0.7 for V4.5 Full or 1 otherwise). Takes `--dry-run` and `--yes`. The output names the `file` written and says whether every encoding was `cached`. A dry run reports the first free name it sees but does not reserve it. |
| `nai meta IMAGE...` | Read the generation settings stored in an image: PNG text chunks, the EXIF of WebP images, or the alpha channel. Several images or a folder (its PNG and WebP files, and with `-r` those of the folders inside) print one JSON line per image with `"type": "image"` and `image`, and a summary line with `"type": "summary"` counting `images`, `without_settings`, `damaged` and `failed`; `--jsonl` prints lines for one image too, and `--raw` adds the stored settings as they are. Every image has `image_ok`: an image whose pixels do not all decode, such as one cut off while saving, still gives the settings it keeps, but is reported as `bad_args` with `"image_ok": false` and a line in `warnings`; one past Pillow's pixel limit is too large to check, so it passes with `"image_ok": null` and a warning. A matching local `ledger` entry is optional: if history cannot be read, the image still returns its settings with a warning. An image without settings is reported in its line as `no_metadata`; damaged images and images that cannot be read make the command fail. |
| `nai strip IMAGE...` | Save a copy of each image without its generation settings, EXIF, colour profile or alpha-channel data, for sharing, as `NAME_clean.png`. An animated image stays animated, each frame cleaned. See [Sharing copies](#sharing-copies). |
| `nai redact IMAGE... --region SHAPE` | Save a copy with regions pixelated, blurred or filled, as `NAME_redacted.png`. |
| `nai watermark IMAGE...` | Save a copy with visible text or a logo, as `NAME_watermarked.png`. |
| `nai batch FILE` | Run commands listed one per line, printing a JSON line for each and a summary last. See Scripting. |
| `nai grid --x PARAM=VALUES [--y ...] COMMAND ...` | Run `gen`, `i2i` or `inpaint` once per cell of settings varied along two axes, printing a JSON line per cell, and save a labelled sheet of the images. See [Grids](#grids). |
| `nai interrogate IMAGE` | Read the tags of an image with a tagger run on this computer, joined into a prompt. See [Local models](#local-models). |
| `nai local [status \| download sam]` | Show what the local models need, or download the SAM model that `object:` regions use; `--dry-run` shows what would be fetched. |
| `nai lint` | Check prompts for common mistakes. |
| `nai tags QUERY...` | Suggest tags for the start of a tag or a Japanese word, as the official prompt box does. |
| `nai models` | List the models and the options each supports. |
| `nai cost` | Estimate the Anlas cost of a request without sending it: `--model`, `--effort`, `--size`, `--steps`, `--n` and `--strength` as in `gen`, `--vibes N` and `--encode N` for Vibes and how many need encoding, `--refs N` for precise references, and `--inpaint` to price with the inpainting model. |
| `nai account [list\|use\|login\|logout]` | Show the account, its balance and free quota and a month's spending under `month` (`--month YYYY-MM`, this month by default), or manage accounts; see Accounts. The ledger is shared by every account on this computer, so `month` has `"scope": "all_accounts"`. |
| `nai stats` | Totals from the ledger by `--by day`, `command` or `model`, for the last `--days` (30) or from `--since` to `--until`. |
| `nai config [init]` | Show the configuration, or create a config file; `init --force` replaces an existing one with the template. |

Run `nai COMMAND --help` for the options of each command. Some examples:

```sh
nai gen --prompt "1girl, rooftop, night" --dry-run
nai gen --prompt @scene.txt --neg @negative.txt --size landscape
nai gen --prompt "2girls, park" --char "girl, blue hair, waving" --char "girl, red hair"
nai i2i --image a.png --prompt "1girl, rooftop, night, rain" --strength 0.5
nai inpaint --image a.png --region ellipse:0.64,0.35,0.96,0.58 --prompt "open hand" --preview
nai inpaint --image a.png --expand 0,0,256,0 --prompt "1girl, rooftop, night, city lights"
nai enhance --image a.png --level 3 --dry-run
nai tool emotion a.png --emotion happy --defry 2
nai tags "blue h" "looking at v"
nai gen --from-image a.png --import prompt,characters --size landscape
echo "1girl, rain" | nai gen --prompt @- -o rain.png
nai cost --model v4.5 --size 1024x1024 --refs 1
```

## Generation options

`gen`, `i2i`, `inpaint`, `enhance`, `variations` and `lint` share these options; `lint` has no run options.

**Prompt.** `--prompt` and `--neg` take the prompt and the negative prompt. `@FILE` reads any prompt option from a UTF-8 file and sends its text as it is. On the command line, `@@` sends a literal `@`: `@@alice` sends `@alice`. Other text starting with `@`, such as the tag `@_@`, is sent as typed. A value that looks like a path and names no file is refused: one starting with a drive letter, `./`, `../`, `~/` or a slash, or one with a slash or a file extension and no comma. `--neg-verbatim` sends the negative prompt without UC preset or nsfw. `--char` adds a character prompt and can be repeated; `--char-neg` and `--char-pos x,y` pair with the characters in order. With positions given, characters are placed by coordinates; otherwise they take free slots in the same order as the official client. As in the official client, a character with an empty prompt keeps its slot but is not sent, and `1girl` and `1boy` in character prompts are sent as `girl` and `boy`; `warnings` says so. Enhance and Variations send the image's character prompts as stored.

**Random prompts.** `--random-prompt` writes the prompt and characters the way the official client's dice button does, and replaces those of an image given with `--from-image`; leave out `--prompt` and `--char`. Each run draws anew, and the result lists what was drawn. Every random character gets the negative prompt `lowres, aliasing, ` and the centre 0,0 with positions off, as there. The official code never draws the word-list entries that need a flag, such as female-only clothes, legwear and shoes, because a pick adds back the flags it needed instead of those it declares; the default applies the declared flags and corrects five records. `--random-prompt=official` follows the official code exactly.

**ComfyUI prompts.** `--comfyui-characters`, or `comfyui_characters = true` in `[defaults]`, splits a multi-character prompt written for ComfyUI into the base prompt and as many characters as the model takes (6, or 32 on V5): `base COUPLE(0 0.5) girl COUPLE(0.5 1) boy`, `cat MASK(0 0.5, 0 1) AND dog MASK(0.5 1, 0 1)`, or a base prompt with one character per line starting with `|`, where a `|` on the first line leaves the base prompt empty. Characters are placed at the centre of their `COUPLE`, `MASK` or `AREA` region with positions on, and one without a region takes the next free place; with no regions at all, positions stay off. `FILL()`, `IMASK()` and the weights after `AND` parts are dropped. A bracket after `COUPLE` that gives no region, such as `(1girl:1.2)`, stays in the prompt, and keywords inside quotes are text; single quotes that span a comma or a line are apostrophes, as in `'90s style, girls' uniform`. A region that is not a number, such as `nan`, places nothing. A prompt without these is sent as typed. It is off by default because a single ` | ` also divides an official prompt, and with it on `--char`, `--char-neg` and `--char-pos` are refused when the prompt gives characters.

**Parameters.** `--model` is one of `v5` (default), `v5-curated`, `v5-medium`, `v4.5`, `v4.5-curated`, `v4` and `v4-curated`. `--effort high|medium` picks the effort of V5 Full, as `effort` in `[defaults]` does; `v5-medium` is V5 Full at medium effort. `--size` takes `WIDTHxHEIGHT` or a preset such as `portrait`, `landscape`, `square`, `portrait-large` or `square-small`. The others are `--seed`, `--steps`, `--cfg`, `--sampler`, `--noise-schedule`, `--cfg-rescale`, `--quality standard|light|none`, `--uc-preset heavy|light|human|furry|none`, `--n`, `--transparent`, `--premultiplied-alpha`, `--furry`, `--no-auto-text`, `--variety` and, for V4, `--legacy-uc`. Without `--noise-schedule`, DPM++ 2M uses `exponential` and the other samplers `karras`, as the official client picks for a sampler; V5 always uses `karras`, and `--variety` is for V4.5 and V4, as the official client hides both for V5. `v5_hidden_options = true` in `[defaults]` lets V5 Full and V5 Curated take a noise schedule and `--variety` anyway; V5 Medium stays as it is. The server takes both on V5 and they change the image as they do on V4.5. `--sampler` takes the official list for V4 and V5: `k_euler_ancestral`, `k_euler`, `k_dpmpp_2s_ancestral`, `k_dpmpp_2m`, `k_dpmpp_2m_sde` and `k_dpmpp_sde`, with or without `k_`. DDIM is not among them, as NovelAI fails it on these models; an old image's `ddim` or `ddim_v3` becomes Euler Ancestral when imported or in `gen_args`, as the official client sends it. Transparent images are sent with straight alpha, the official default; `--premultiplied-alpha`, or `premultiplied_alpha` in `[defaults]`, chooses the official client's other mode. Values outside the official ranges are refused as well: steps 1 to 50, CFG from 0 with no upper limit, CFG rescale 0 to 1, seeds 0 to 9999999999, and sizes must be positive multiples of 64.

Options not given fall back to `[defaults]` in the config file, then to the model's defaults. V5 defaults to CFG 4 and 28 steps, the settings most commonly used with it, and 28 steps is the most the Opus free tier covers. The official client uses CFG 7 and 23 steps; put those in `[defaults]` to match it. V4.5 defaults to CFG 5 and 23 steps, as in the official client. `quality` and `uc_preset` in `[defaults]` apply to every model: one without that tier or preset takes the closest it has, while the same value typed as an option is refused. The number of images per request is limited by size: 4 for common sizes and up to 8 for small ones.

**References.** Vibes are for V4.5 and V4, precise references for V5 and V4.5. `--vibe FILE[:STRENGTH[:INFO]]` accepts a `.naiv4vibe` file, a `.naiv4vibebundle` file, an image the official client exported Vibes into, or a plain image, which is encoded on first use and cached. If the cache cannot be written, the encoding is still used and `warnings` says so. A bundle adds every Vibe it holds, and STRENGTH and INFO then apply to all of them. `--vibes-from IMAGE` adds the Vibes kept in a generated image's settings, with their strengths and information extracted, as the official client restores them when importing the image; they are kept for the model that made the image. Vibe files are checked as the official client imports them: each Vibe's id must be the SHA-256 of its image, or of its encoding when it holds no image. A Vibe without its image cannot be encoded again, so its information extracted stays as stored. `--ref IMAGE[:KIND[:STRENGTH[:FIDELITY]]]` adds a precise reference, up to 16, where KIND is `both` (character and style, the default), `character` or `style`. V5 also takes outfits: `outfit` references the clothes alone, `both` keeps the outfit and `both-any-outfit` lets it change, `character` lets it change and `character+outfit` keeps it, as the official "Prefer same outfit" box does. The official names `character&style`, `character&style_anyoutfit`, `character_mainoutfit` and `costume` are accepted too. On V4.5 the outfit preferences are sent as `both` and `character` with a warning, and `outfit` is refused. V5 has no fidelity and sends every reference at fidelity 1; a reference with V5 Full Medium runs V5 Full, as in the official client, and `warnings` says so. `--ref-strength` and `--ref-fidelity` set the strength or fidelity of every reference, over their own. An empty field keeps its default, as in `--ref a.png:style::0.5` or `--vibe a.naiv4vibe::0.7`. INFO is 0.01 to 1; strengths and fidelity have no limit, as in the official client. With several Vibes, strengths summing above 1 are scaled down to a sum of 1, as by default in the official client; `--no-normalize-vibe-strength` sends them as given.

**Run.** `--dry-run` builds the request and prices it without sending anything paid, and includes the request body. With a token it reads the balance to price it, without renewing an email login. It makes the output folders and the file it would write to see that the file system takes them, then removes the ones it made, so it leaves nothing behind but the tokenizer files cached on first use; if the balance cannot be read, the price assumes Opus and says so in `warnings`. Options are spelled out in full: `--outp` is not taken for `--output`. `--yes` allows requests above `max_per_request`. `--show-body` includes the request body in the output, with image data elided.

## Image to image and inpainting

Without `--size`, image to image chooses the size the way the official client imports an image: the configured size if the aspect ratio matches, the image's own size if both sides are multiples of 64 within the pixel limit, and otherwise a size scaled from a 1216 long side or an 896 short side. An enhanced 1664×2432 image is therefore sent at 832×1216, which is free with Opus, and `warnings` says so. Pass `--size` to send it larger. `--strength` takes 0.01 to 0.99 and `--noise` 0 to 0.99.

Inpainting takes the area to repaint from any combination of:

- `--mask IMAGE`: white or opaque areas are repainted. A mask of a different size is stretched to the source.
- `--region SHAPE`: `rect:x1,y1,x2,y2`, `ellipse:x1,y1,x2,y2` (bounding box), `poly:x,y,x,y,x,y,...` or `object:x,y,...`, in fractions of the canvas. `object` covers the object under the points, found by the local SAM model (see [Local models](#local-models)). A leading `-` subtracts the shape. Regions can be repeated and apply in order.
- `--expand L,T,R,B`: adds transparent canvas on each side (or one number for all sides). Each side is rounded up to a multiple of 64, as the official resize dialog rounds it, and the new area is repainted.

`--grow PX` grows the whole area. `--inpaint-strength` (0.01 to 1, default 1) lets part of the source show through below 1. The canvas is built like the official editor's: images with a side above 2560 are scaled down, then each side is rounded up to a multiple of 64 and the image is stretched to fit (1467×2144 becomes 1472×2176).

**Focus.** nai-cli crops the area around the mask, plus `--focus-padding` pixels of context (96 by default, 32 to 192), scales crops under 0.8 MP up to about 1 MP, and composites the result back at full size. This happens automatically when the crop covers the whole mask, fits in 1 MP and is smaller than the canvas, so repainting a hand on a large image stays free and keeps its resolution. `--focus` and `--no-focus` force it either way; `--size` applies only without focus.

Without focus the whole canvas is sent, as in the official client. Canvases up to 3 MP are sent at their own size; larger ones are scaled down like an import, and the output has the reduced size. `warnings` explains when this happens.

After generation the result is feathered and composited onto the source exactly as the official client does it, and the generation settings are carried over, so `nai meta` can read the output. `--keep-raw` also saves the image as returned by the server. `--preview` sends nothing paid and saves a preview showing the repainted area as the server sees it (in red, in 8 px cells) and the focus crop (in green); `preview` names the file, `null` in a dry run, and no limit refuses it. Character positions are given relative to the whole canvas and mapped into the crop.

## Enhance and Variations

Both start from the settings stored in the image (model, prompts, characters, steps, CFG and so on); options given on the command line override them, including empty ones such as `--neg ""`. Switches the image turned on are turned off with their `--no-` form: `--no-furry`, `--no-variety`, `--no-transparent`, `--no-neg-verbatim` and `--no-auto-text`. `--char` replaces the image's characters along with their negative prompts and positions. If the image has no stored settings, give `--prompt`. A new seed is used unless `--seed` is given. The image's prompts are sent as stored, or tidied only by `--format-prompt` and `--space-numeric-close` when given; prompts given on the command line are tidied as in `gen`. The Vibes kept in the image's settings come along, as with `--vibes-from`. A stored size is aligned to the grid as the official client sends it; when that exceeds the pixel limit, the longer side is reduced until it fits (1467×2144 becomes 1472×2112), and `warnings` says so, as it does for precise references and Vibes the image does not keep. A stored size NovelAI still does not accept is replaced as for an imported image.

`nai enhance` maps `--level 1` to `5` to strength 0.2, 0.4, 0.5, 0.6 and 0.7, with 0.1 noise at level 5; the default is 3, and `--strength` and `--noise` override the level. `--scale` chooses among the available scales and defaults to the largest. For V5 images under 2.5 MP that is `max`, where the server upscales to the pixel limit (832×1216 becomes 1467×2144). 832×1216 offers only 1.5 and 1; other sizes offer scales whose result is a multiple of 64 within the limit. Enhance results are larger than 1 MP and cost Anlas.

`nai variations` uses strength 0.8 and noise 0.1 and generates 3 images, or fewer when the size sent allows fewer. `--n`, `--strength` and `--noise` change this.

## Automatic additions

nai-cli adds the same text as the official client. The `prompt` and `negative` fields of the output, including with `--dry-run`, show the text as sent.

- **Quality tags** go at the end of the first mix segment of the prompt, before the first `text:` if there is one. `--quality none` disables them.
- **UC presets** go at the start of the negative prompt. `--uc-preset none` disables them.
- **Furry mode** (`--furry`) puts `fur dataset, ` in front of the prompt, quality tags and Enhance hint included, unless it already starts with `fur dataset` or `background dataset`.
- **teXt: block** (V5): quoted text in the prompt is copied into a trailing `teXt:` block so the model renders it. `--no-auto-text` disables it.
- **Enhance** adds `-2::upscaled, blurry::` after the quality tags, except at `max` or when the prompt already contains `upscaled, blurry`.
- **Inpainting** follows the rules of the inpainting model. V5 Curated inpaints with the V4.5 Curated inpainting model, so quality tags, UC presets and prices change accordingly; `warnings` mentions it. Vibes and precise references still follow the model chosen, as in the official client: no V5 model takes precise references when inpainting, so `--ref` is refused there.

**nsfw.** By default nai-cli never adds `nsfw` to the negative prompt. When the prompt contains `nsfw` (also `{nsfw}` or `[nsfw]`), it removes `nsfw` and its trailing comma from the negative prompt, so the two never cancel each other out by accident. Removal would break tags such as `2::nsfw::`, and `warnings` lists them.

`official_nsfw` in the config, or `--official-nsfw`, restores the official behaviour: V5 Full, V5 Medium, V4.5 Full and V4 Full prepend `nsfw, ` to the negative prompt when a UC preset is active and the prompt does not contain `nsfw`. Curated models never add it, and nothing is removed.

## Prompt tidying

Optional switches, all off by default. The official client has none of them.

- `format_prompt` or `--format-prompt` converts full-width commas, collapses extra whitespace and drops empty tags. With `format_underscores`, underscores become spaces as well. Quoted text, `text:` segments, `<markers>` and kaomoji are left untouched, and line breaks are kept.
- `space_numeric_close` or `--space-numeric-close` inserts a space when a closing `::` follows a number. NovelAI reads a number right before any `::` as a new weight, so in `1.2::year 2025::` every following tag is weighted 2025. `1.2::year 2025 ::` behaves as intended. With the switch off, `nai lint` reports `weight_numeric_close`.

- `convert_sd_weights` or `--convert-sd-weights` turns SD weights into NovelAI's: `(long hair:1.2)` and `[tag:0.8]` become `1.2::long hair::` and `0.8::tag::`. Brackets without a number stay, since NovelAI reads them too, and `\(` and `\)` lose their backslash. It runs before the other two.
- `prompt_prefix`, `prompt_suffix`, `negative_prefix` and `negative_suffix` are fixed tags joined around every prompt typed for a run with `, `. Prompts taken from an image are left as stored, and `--no-fixed-tags` leaves them out for one run.

When tidying changes a prompt, `gen_args` from `nai meta` turns it off explicitly so that reruns stay exact; it always turns fixed tags off, since the stored prompts already hold them.

## Checking prompts

`nai lint` takes the generation options, checks them as `gen` does and reports `issues` without pricing or sending anything; generation commands include the same issues as `lint`. Each issue has a `level`, a `code`, a `message` and the `field` it concerns.

`tokens` counts the prompt and the negative prompt as sent, with the tokenizer each model uses: Qwen3.5 for V5 and T5 for V4.5 and V4. As in the official prompt boxes, every mix segment is counted separately, the longest option of each `||a|b||` group counts, and character prompts add to the prompt and their negative prompts to the negative prompt. Above the model's `limit` NovelAI cuts the rest off, and `token_limit` is reported. The tokenizer files come from novelai.net, as in the official client, and are kept in the data folder after the first count (3.7 MB for V5, 0.6 MB for V4.5). Without them `exact` is `false` and the counts are rough estimates.

`text_too_long` warns when the text after `text:`, in the prompt or a character prompt, is longer than the official documentation says the model renders reliably: 750 characters for V5 Full, 374 for V5 Curated and 118 for V4.5, counted from the first character after `text:` with spaces and newlines included. No limit is published for V5 Medium, so it is not checked.

## Tags and models

`nai tags` looks up each query the way the official prompt box does while you type, which helps to check that a tag exists and how it is spelled. Each entry in `results` holds the `query`, the `lang` it was looked up in and the suggested `tags`, each with a `count` of how common it is (at most 10000). Queries that are more than 30% kana, kanji or full-width characters are looked up in Japanese, and their tags also carry the Japanese name in `jp`; Chinese words are treated the same way and often find little. The tag list follows `--model`, or the configured model, and `--furry` switches to the furry list. Lookups are free but need a token, and several queries are sent half a second apart.

`nai models` reports the capabilities in effect with `v5_hidden_options`. It lists every model with its `id`, the `aliases` accepted by `--model`, its `tokenizer`, `token_limit`, `max_characters` and `text_limit`, the features it supports (`vibe`, `precise_reference` with `reference_inpainting`, `reference_fidelity` and `outfit_reference`, `transparent`, `furry`, `auto_text`, `variety`, `noise_schedule`, `cfg_rescale`, and `legacy_uc`), the `effort` of a V5 Full id, the values `--quality` and `--uc-preset` accept with the ones it starts from, its default steps and CFG, and the settings V5 Medium fixes. `default` is the model used when `--model` is omitted, at the effort of `[defaults]`.

## Output

Every command prints one JSON object to standard output; `nai batch`, `nai grid`, and `nai meta` of several images or a folder, print one per line and a summary last. A successful generation:

```json
{
 "ok": true,
 "model": "nai-diffusion-5-full",
 "action": "generate",
 "seed": 3829104422,
 "size": "832x1216",
 "prompt": "1girl, rooftop, night, very aesthetic, masterpiece, no text",
 "negative": "lowres, artistic error, ...",
 "characters": [],
 "cost": {"estimate": 0, "free_by_opus": true, "if_quota_exhausted": 30, "spent": 0, "anlas_left": 9970, "quota_percent": 83.0},
 "lint": [],
 "tokens": {"prompt": 14, "negative": 52, "limit": 1471, "exact": true},
 "warnings": [],
 "images": [".../2026-11-16/153012_3829104422.png"]
}
```

`spent` is `null` when the balance could not be read after the request; the images are saved either way and `warnings` says so. When a step fails or is interrupted after a paid request ran, such as saving the images or generating after a Vibe was encoded, the error still carries `cost` with the spend and `saved` with the files already written, and the ledger records the request with its `error` code. A request that NovelAI refused or never received costs nothing and is not recorded. Interrupting while the closing balance is read or the ledger written still reports the spend, with a warning if the ledger may lack the request. An interrupt while closing a finished job keeps its result; a batch stops before starting the next job. `retries` lists any requests that were sent again.

On failure the exit status is non-zero and the output is `{"ok": false, "code": "...", "message": "..."}` with fields related to the error:

| Code | Meaning |
|---|---|
| `bad_args` | Invalid arguments; often includes `choices` or `usage`. |
| `config_missing` | No token, or the config file cannot be parsed. |
| `need_confirm` | The estimate exceeds `max_per_request`; includes `cost`. Rerun with `--yes` to proceed. |
| `ledger` | The ledger cannot be read, or the generation lock cannot be used. Nothing is sent. |
| `model_unsupported` | The model lacks a feature, named in `feature`: `vibe`, `precise_reference`, `outfit_reference`, `characters`, `variety`, `transparent`, `furry`, `legacy_uc` or `inpainting`. |
| `empty_mask` | The mask selects nothing. NovelAI would still charge for the request. |
| `no_metadata` | The image has no NovelAI generation settings. |
| `auth` | The token is invalid. |
| `insufficient_anlas` | The balance is below the estimate, found before sending and reported with `cost`, or NovelAI refused the request for lack of Anlas. The refused request costs nothing, but `cost.spent` still counts a Vibe encoded before it. |
| `rate_limited` | Still rate limited after several retries. |
| `busy` | Another nai on this machine holds the generation queue, or is still changing the account store. |
| `server`, `network`, `api` | Server error, connection failure, or a refused or unusable reply. A paid request that may have run despite the error has `maybe_processed: true`. When only some images of a reply are damaged, the others are kept and listed in `saved`. |
| `save_failed` | The result could not be written. After a paid request, `saved` lists the files written before the failure, `cost` the spend, and `maybe_processed: true` says the request ran; free commands such as `pixel-snap` and `local download` have no `cost`. |
| `interrupted` | Interrupted by the user. After a paid request ran, includes `cost` and `saved`. |
| `batch_failed` | Some jobs of `nai batch` failed; each has its own line, and `batch` holds the totals. |
| `internal` | A bug in nai-cli; the traceback is printed to standard error. |

## Grids

`nai grid` runs one command per cell, the way the official client's grid generation lays out a comparison: the values of `--x` change across the columns and those of `--y` down the rows. Axes come first and the command after them, with its own options as usual.

```bash
nai grid --x steps=20-28:4 --y cfg=4,5,6 gen --prompt "1girl, garden"
nai grid --x sampler=k_euler,k_dpmpp_2m_sde --x cfg=4,6 gen --prompt @prompt.txt --seed 7
nai grid --y prompt=@prompts.txt --x seed=r3 --dry-run gen --size portrait
```

A parameter is one of `steps`, `cfg` (also `scale`), `cfg_rescale`, `seed`, `size`, `sampler`, `noise_schedule`, `model`, `prompt`, `neg` (also `uc`), `ref_strength` (also `pr_strength`) and `ref_fidelity` (also `pr_fidelity`), the last two for every precise reference of the command. Numbers take values separated by commas and ranges `START-END:STEP`, the step 1 by default; `seed` also takes `r` or `r5` for one or five random seeds. `size`, `sampler`, `noise_schedule` and `model` take names separated by commas; `prompt` and `neg` take one value per line, or `@FILE`. Several parameters on one axis change together, a shorter list keeping its last value. A grid has at most 40 cells.

Each cell is a job as in `nai batch`, with the cell's values added last, so they override the command's own. The lines, the summary, `--interval`, `--stop-on-error` and `--dry-run` work as there. Before the first cell is sent, every cell is tried as a dry run, and a cell that cannot run fails the whole grid with nothing sent. When neither the command nor an axis sets a seed, and the command does not request a seed from an imported image, one random seed is used for every cell, so only the axes change. The summary's `grid` gives the columns, rows and axis names, and the `sheet` saved as `NAME_grid.png` beside the first image: every image shrunk to fit 512 pixels, labelled along both axes, with grey for cells that failed. `--no-sheet` saves none.

## Sharing copies

`strip`, `redact` and `watermark` never change the source. Each copy is a PNG named after its source with `_clean`, `_redacted` or `_watermarked`, saved in the output folder by its folder rule like a generated image, which goes by the settings the copy keeps, so a copy without them is filed by date alone; the file name template does not apply. `-o FOLDER` saves each copy there under its source's name, and `-o NAME.png` names the copy of one image. A name that is taken, `-o` included, gets the next number. Every copy is made before any is written, so one bad image leaves none behind.

Copies drop the generation settings, EXIF, colour profile and the low alpha bit where stealth settings hide, and the low colour bits when they hold the `stealth_rgbinfo` or `stealth_rgbcomp` data other tools write. `redact` and `watermark` take `--keep-metadata` to carry the source's settings over instead: PNG text as it is, and settings a WebP keeps in EXIF or an image hides in its alpha channel written as PNG text, since the edit would lose them. Both edit still images only; an animated one is refused.

```bash
nai redact pic.png --region rect:0.6,0.1,0.95,0.4
nai redact pic.png --region ellipse:0.3,0.3,0.7,0.7 --effect blur
nai redact pic.png --region rect:0.4,0.4,0.6,0.6 --invert --effect solid --color "#202020"
```

`--region` takes the shapes of [inpainting](#image-to-image-and-inpainting), in fractions of the image; a leading `-` subtracts and several apply in order. `--effect` is `pixelate` (the default), `blur` or `solid`. `--amount` sets the block size, 0.025 of the short edge by default, or the blur radius, 0.018. `--opacity` below 1 lets the source show through, and `--invert` covers everything outside the regions.

```bash
nai watermark pic.png
nai watermark pic.png --text "@me\n2026" --position top-right --auto-tone
nai watermark pic.png --logo logo.png --text "" --logo-position bottom-right
```

By default the watermark is the text `NovelAI` at the bottom right, sized 0.1 of the short edge, 0.035 from the edges, white at 0.82 opacity with a thin dark outline and shadow; a logo at the bottom left, 0.16 of the short edge on its longest side, at 0.85 opacity. With `--logo` and no `--text` only the logo is drawn. Text too long for the image shrinks to fit; `\n` breaks lines, which are right-aligned. The built-in font has little beyond ASCII; give `--font FILE` for accents, Chinese, Japanese or Korean (on Linux, Noto Sans CJK is usually under `/usr/share/fonts`). Text the font has no glyphs for is refused, naming the characters, instead of drawn as boxes; accents written as separate marks are checked as the composed letter, and marks, joiners and emoji variation selectors are checked with the character they belong to. A Pillow built without FreeType has only a small bitmap font for Latin-1 text, such as ASCII and accented letters; it refuses `--font` and other text, and grid sheet labels show `?` for characters it cannot draw.

`--position` and `--logo-position` take `top-left`, `top`, `top-right`, `left`, `center`, `right`, `bottom-left`, `bottom` or `bottom-right`, and `--size`, `--margin`, `--opacity` with their `--logo-` forms adjust each part. `--auto-place` mirrors the marks to another corner when it is at least 15% calmer than where they were put. `--auto-tone` greys them to stand out from what lies under them: lighter on dark areas, darker on light ones.

## Local models

Two features run models on this computer instead of NovelAI's servers, so they cost nothing. They need an extra install, and nothing is downloaded unless you ask:

```
pip install "nai-cli[local]"
nai local                  # what is installed and what is missing
nai local download sam     # about 140 MB, checked against its published hash; run again to replace a damaged file
```

The extra install is onnxruntime, which has wheels for 64-bit Windows, macOS and Linux with glibc. On Alpine and other musl systems, 32-bit systems, or a Python version it has no wheel for, pip cannot install it; everything else in nai-cli works without it.

**Object regions.** `object:x,y` in `--region` of `nai inpaint` and `nai redact` covers the object under the point, found by EfficientViT-SAM L0. Several points (`object:0.3,0.4,0.35,0.6`) mark the same object; a leading `-` takes it out of the mask. The model goes to `models/sam` under the data directory, or to `sam_dir` under `[local]`; it is the L0 encoder and decoder of [EfficientViT-SAM](https://github.com/mit-han-lab/efficientvit) by MIT Han Lab, whose code is under Apache-2.0; the Hugging Face repository it is fetched from states no licence of its own for the weights, which are not part of nai-cli.

```
nai inpaint --image pic.png --region object:0.52,0.31 --prompt "red hat"
nai redact pic.png --region object:0.5,0.4 --effect blur
```

**Interrogate.** `nai interrogate IMAGE` reads the tags of an image with an ONNX tagger you provide: WD14 taggers, CL tagger and CL tagger v2, EVA02 or PixAI. Give the model with `--model` or `tagger` under `[local]`; its tag list is found beside it (`selected_tags.csv`, `tag_mapping.json`, `tags.json`, a `config.json` that holds `tags_split`, and similar) or given with `--labels`; a model whose number of outputs differs from the tag list is refused, and its kind is guessed from the names or given with `--kind wd14|cl|cl-v2|eva02|pixai`. The result has `prompt`, the tags joined in the order artist, character, copyright, general and meta, and `tags` with each score. General tags need a score of 0.35 (`--threshold`), character tags 0.35 (`--character-threshold`); copyright, artist and meta tags use the model's own thresholds where it has them. `--categories` picks from `general`, `character`, `copyright`, `artist` and `meta` (default: the first three). Artist tags come with an `artist:` prefix, and tags keep the tag list's underscores.

```
nai interrogate pic.png --model ~/models/wd-vit-tagger-v3/model.onnx --categories general,character
```

## Scripting

nai-cli is designed to be called from scripts and other programs:

- Branch on `ok` and `code`. Messages are meant for people and may change.
- Out-of-range values such as image count, steps or CFG are rejected with `bad_args` rather than clamped.
- Run with `--dry-run` first to see the final prompts and the price.
- Treat `need_confirm` as a question for the user, and add `--yes` only after they agree.
- nai-cli never resends a paid request that may have reached NovelAI. After an error with `maybe_processed`, check the balance with `nai account` before trying again, and ask the user if it has dropped.
- Pass long prompts with `--prompt @FILE` to avoid shell quoting problems, or pipe one in with `@-`. Only one option can read standard input.
- `-o` saves the images of generation, upscaling and Director tools under a name of your choice: a `.png` name, extended with `_1`, `_2` for further images and with the output's name for background removal, or a folder used without day folders. Nothing is overwritten: a name that is taken, `-o` included, gets `-1`, `-2` and so on, as every name does.
- `--from-image IMAGE` starts `gen`, `i2i`, `inpaint` or `lint` from an image's settings, as the official import dialog does. `--import` picks the parts from `prompt`, `uc`, `characters`, `settings`, `seed` and `vibes`, or `all`; the default is `prompt,uc,characters,vibes`: the dialog's choice for an image from disk, plus the Vibes the image keeps, which come along only when the model is the one that made them. The quality tags come with `settings` and the UC preset with `uc`. As in the official import, `settings` leaves `--legacy-uc` and the alpha mode as configured; `enhance` and `variations` keep the image's, and `gen_args` records both. Naming `settings` or `vibes` in `--import` switches to the image's model unless `--model` is given; on another model, what it cannot do is dropped and `warnings` says so. Options given override the imported ones, switches such as `--no-variety` turn off what the image had on, and characters given replace the image's. `--clean-brackets` drops `[ ] { }` from every imported prompt, characters and negative prompts included, and `--actual-prompts` brings in the options the image drew from `||a|b||` groups instead of the groups. Text from an image is used as it is: no `@FILE` reading and no fixed tags. Inpainting keeps the size of the image it repaints. `warnings` lists what the imported parts cannot bring back, as for `gen_args`.
- `enhance` and `variations` take the image's prompts unless given, and send the options the image drew from `||a|b||` groups again, also for groups in a prompt typed for them, as the official client does.
- `nai batch FILE` runs one command per line of a JSONL file, or of standard input with `-`. A line is an argument list such as `["gen", "--prompt", "cat"]` or an object `{"id": "cat-1", "args": [...]}`; blank lines and lines starting with `#` are skipped. Each job prints its result as soon as it ends, with `"type": "job"`, `job` (its id, or its line number), `line` and `exit`, and the last line, with `"type": "summary"`, sums them in `batch`: `jobs`, `succeeded`, `failed`, `skipped` and `spent`. Lines end at line feeds only, so an id holding U+2028 or the like stays one line. Every job keeps its own limits, so `--yes` belongs in the job. `--dry-run` runs each job as its own dry run: nothing paid is sent and no output or login changes are kept, so a job does not see what the jobs before it would leave, such as the images they would make. Output checks may create and remove temporary probes; the tokenizer cache may be filled on first use. Every job line is marked `"dry_run": true` and the summary totals the `estimate`; the balance is read once per account. A job its limits would refuse fails there as in the run, so `--stop-on-error` stops at it. A job's own `type`, `line` or `exit` never replaces the batch's. `--stop-on-error` stops at the first failure. `--interval S` or `--interval MIN-MAX` keeps that many seconds, or a random time between the two, up to 86400, from the last request sent to the first request of the next job; jobs that send nothing, such as `strip` or `meta`, do not wait, and a job's own requests are not spaced; `interval` under `[batch]` sets the default. A job may name its own `--account` before its command; `--config` is refused in a job, so the batch's limits hold. Save the output and pass it to `--skip-done` to run only what is left: jobs that succeeded are skipped, and every other job runs again, a failed one too, so check the `cost` of a job that failed with `maybe_processed` before resuming. The output of several runs can be appended to one file: a job counts as done if any line says it succeeded, and dry-run lines never count. Jobs cannot read standard input.
- `gen_args` from `nai meta` is an argument list that reproduces an image when appended to `nai gen`. Sizes that are not multiples of 64, such as those of upscaled Enhance results, are aligned as the official client sends them, and the Vibes kept in the image come back through `--vibes-from`. `warnings` lists what the rerun cannot bring back: precise references, Vibes the settings do not keep, a size reduced to fit the pixel limit, and out-of-range numeric settings left to the defaults. `gen_args` is `null` when the image's prompts hold a NUL or text that is not valid UTF-8, which no request can carry; importing such an image is `bad_args`, as is such text read with `@FILE` or `@-`.
- An account runs one generation at a time. Concurrent nai-cli processes on the same machine wait for each other instead of failing.

AI agents can learn these rules, along with error recovery and the inpainting workflow, from the [nai-cli skill](../skills/nai-cli/SKILL.md). The README explains how to install it.

## Pricing

- Every generation is priced beforehand with the official client's formulas. With Opus, one image per request of up to 1024×1024 and 28 steps is free; precise references still cost 5 Anlas per reference per output image. V5 draws on a free quota that can run out, after which the normal price is charged; `if_quota_exhausted` in the output shows that price.
- Director tools are free with Opus up to 1024×1024, except background removal. Upscaling costs 1 to 4 Anlas by the source's area, Opus included.
- Inpainting is priced by the size sent, which is usually within 1 MP when focusing. Enhance is priced by the upscaled size; `max` is priced by the size the server upscales to.
- Requests estimated above `max_per_request` (100 by default) return `need_confirm` and are not sent. The estimate is made with the balance and free quota read just before sending. A balance below the estimate returns `insufficient_anlas` before anything is sent, so no Vibe is encoded for a generation that cannot run.
- The actual spend is the difference in balance before and after the request. Once a request has run, it is written to a local ledger with its spend, the images asked for, the files saved and the account that paid, even when a later step fails; a request refused before anything was charged is not written, and one whose process was killed while it ran is missing. A line that cannot be read is skipped. `nai account` totals the ledger by month and `nai stats` by day, command or model; `requests_with_unknown_spend` counts requests whose closing balance could not be read.

## Compatibility

The public interface is the command line: commands and options, the config file's keys, the JSON fields and line types the commands print, the error codes and the exit status. While nai-cli is at 0.x, an incompatible change to it comes only in a new minor version and is listed in the [changelog](../CHANGELOG.md); error codes are never renamed or removed.

The Python modules are internal and change without notice, as do the files in the data folder: the tokenizer and Vibe caches and the ledger's line format. Programs call `nai` and read its output rather than import `nai_cli` or read those files.

