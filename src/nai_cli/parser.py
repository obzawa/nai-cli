"""Every command-line option: name, type and help. Unset options parse as None; resolution supplies defaults."""

import argparse
import math
import re
import sys
from collections.abc import Sequence
from typing import Any

from nai_cli import (
    __version__,
    enhance,
    errors,
    grid,
    import_image,
    inpaint,
    ledger,
    models,
    presets,
    random_prompt,
    share,
    tagger,
    values,
)

SWITCH = argparse.BooleanOptionalAction
MODEL_HELP = "v5, v5-curated, v5-medium, v4.5, v4.5-curated, v4, v4-curated (default: v5)"
EFFORT_HELP = "V5 Full at high or medium effort (default: the config's effort, else high)"


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # A shortened option would change meaning once a later option shares its start
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:  # type: ignore[override]
        raise errors.NaiError(errors.BAD_ARGS, message, usage=self.format_usage().strip())

    def parse_args(self, args: Sequence[str] | None = None, namespace: Any = None) -> Any:  # type: ignore[override]
        argv = list(args) if args is not None else sys.argv[1:]
        # A job file can hold a NUL, which no prompt, option or file name takes
        if any("\x00" in arg for arg in argv):
            self.error("an argument holds a NUL character")
        parsed, extras = super().parse_known_args(_attached(argv), namespace)
        if extras:
            # argparse reports them with the top-level usage, which does not say what the command takes
            self._command(parsed).error(f"unrecognized arguments: {' '.join(extras)}")
        return parsed

    def _command(self, parsed: Any) -> "_Parser":
        commands = next(a for a in self._actions if isinstance(a, argparse._SubParsersAction))
        return commands.choices.get(getattr(parsed, commands.dest, None), self)


# A region that subtracts starts with -, which argparse would take for an unknown option
_SUBTRACTED = re.compile(rf"-(?:{'|'.join(inpaint.SHAPES)}):")


def _attached(argv: list[str]) -> list[str]:
    """--region -rect:... as --region=-rect:..., so the documented spelling works."""
    out: list[str] = []
    for i, arg in enumerate(argv):
        if arg == "--":
            return out + argv[i:]
        if out and out[-1] == "--region" and _SUBTRACTED.match(arg):
            out[-1] = f"--region={arg}"
        else:
            out.append(arg)
    return out


def finite(text: str) -> float:
    """Decimal option values; nan and infinity are refused."""
    try:
        number = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError(f"not a finite number: {text!r}")
    return number


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="nai",
        description="Command-line client for NovelAI image generation. Every command prints one JSON object; "
        "batch, grid and meta of several images print one line each and a summary.",
    )
    p.add_argument(
        "--config",
        type=values.path,
        metavar="PATH",
        help="config file (default: nai-cli/config.toml in the user config directory)",
    )
    p.add_argument("--account", metavar="NAME", help="account to use for this command (default: nai account use)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND", parser_class=_Parser)

    g = sub.add_parser("gen", help="generate images from a prompt")
    _add_generation_args(g)
    _add_import_args(g)
    _add_run_args(g, show_body=True)

    i = sub.add_parser("i2i", help="generate from a source image")
    _add_generation_args(i)
    _add_image_strength_args(i, "0.7", "0")
    _add_import_args(i)
    _add_run_args(i, show_body=True)

    ip = sub.add_parser("inpaint", help="repaint part of an image and composite the result back")
    _add_generation_args(ip)
    _add_image_strength_args(ip, "0.7", "0")
    _add_inpaint_args(ip)
    _add_import_args(ip)
    _add_run_args(ip, show_body=True)

    en = sub.add_parser("enhance", help="upscale and refine an image with its embedded settings")
    _add_generation_args(en)
    _add_image_strength_args(en, "the level's", "the level's")
    en.add_argument(
        "--level",
        type=int,
        choices=range(1, len(enhance.LEVELS) + 1),
        help="strength 0.2, 0.4, 0.5, 0.6 or 0.7, with 0.1 noise at level 5 (default: 3); --strength and --noise override it",
    )
    en.add_argument("--scale", help="max, 2, 1.5 or 1 (default: the largest available)")
    _add_run_args(en, show_body=True)

    va = sub.add_parser("variations", help="generate similar images with an image's embedded settings")
    _add_generation_args(va)
    _add_image_strength_args(va, str(enhance.VARIATION_STRENGTH), str(enhance.VARIATION_NOISE))
    _add_run_args(va, show_body=True)

    u = sub.add_parser("upscale", help="upscale an image of up to 3145728 pixels in area")
    u.add_argument("image", type=values.path)
    _add_run_args(u)

    t = sub.add_parser("tool", help="run a Director tool")
    t.add_argument("tool", choices=[*presets.AUGMENT_TOOLS, presets.PIXEL_SNAP])
    t.add_argument("image", type=values.path)
    t.add_argument(
        "--prompt",
        metavar="TEXT",
        help="prompt for colorize or emotion; @FILE reads it from a file, @- from standard input",
    )
    t.add_argument("--emotion", choices=presets.EMOTIONS, metavar="NAME", help="target emotion (default: neutral)")
    t.add_argument(
        "--defry",
        type=int,
        choices=range(presets.DEFRY_MAX + 1),
        help="how far colorize and emotion may deviate (default: 0)",
    )
    t.add_argument(
        "--colors",
        metavar="auto|off|N",
        help="pixel-snap palette: auto, off, or 1 to 256 colours (default: auto)",
    )
    t.add_argument(
        "--avoid-over-refining",
        action=SWITCH,
        help="pixel-snap keeps the grid found instead of trying a finer one that fits better",
    )
    t.add_argument(
        "--upscale",
        action=SWITCH,
        help="pixel-snap scales the result back up to about the source's size (default: one pixel per cell)",
    )
    _add_run_args(t)

    v = sub.add_parser(
        "vibe", help="save Vibes as a .naiv4vibe or .naiv4vibebundle file, encoding images (V4.5, V4, 2 Anlas each)"
    )
    v.add_argument(
        "sources",
        nargs="+",
        metavar="SOURCE[:STRENGTH[:INFO]]",
        help="an image to encode, or a Vibe file or image carrying Vibes; several make a bundle",
    )
    v.add_argument("--model", help="a model with Vibes (default: v4.5)")
    v.add_argument(
        "--info",
        type=finite,
        help="information extracted where a source gives none, 0.01-1 (default: 0.7 for V4.5 Full, 1 otherwise)",
    )
    v.add_argument("--strength", type=finite, help="strength stored where a source gives none (default: 0.6)")
    v.add_argument(
        "--image",
        dest="keep_image",
        action=SWITCH,
        help="keep the source images beside the encodings (default: on); --no-image keeps only the encodings, "
        "as the official export can",
    )
    v.add_argument(
        "--into", type=values.path, metavar="PNG", help="write the Vibes as a bundle into a copy of this image"
    )
    v.add_argument("-o", "--output", metavar="PATH", help="the file to write; an existing file is never replaced")
    _add_run_args(v, output=False)

    st = sub.add_parser(
        "strip", help="save copies of images without generation settings, EXIF or stealth data, for sharing"
    )
    _add_copy_sources(st, "clean")

    rd = sub.add_parser("redact", help="save copies of images with regions pixelated, blurred or filled, for sharing")
    _add_copy_args(rd, "redacted")
    rd.add_argument(
        "--region",
        action="append",
        required=True,
        metavar="SHAPE",
        help="rect:x1,y1,x2,y2, ellipse:x1,y1,x2,y2, poly:x,y,x,y,x,y,... or object:x,y,... (the object under "
        "the points, with the local SAM model) in fractions of the image. A leading - subtracts; repeat to apply "
        "several in order",
    )
    rd.add_argument("--effect", choices=share.EFFECTS, help="how regions are covered (default: pixelate)")
    rd.add_argument(
        "--amount",
        type=finite,
        metavar="FRACTION",
        help=f"block or blur size as a fraction of the short edge (default: {share.PIXEL_SIZE} or {share.BLUR})",
    )
    rd.add_argument("--color", metavar="#RRGGBB", help="fill of the solid effect (default: #000000)")
    rd.add_argument("--opacity", type=finite, help="strength of the cover, above 0 to 1 (default: 1)")
    rd.add_argument("--invert", action=SWITCH, help="redact everything outside the regions")

    wm = sub.add_parser("watermark", help="save copies of images with visible text or a logo, for sharing")
    _add_copy_args(wm, "watermarked")
    wm.add_argument("--text", help="text to write; \\n breaks lines (default: NovelAI unless --logo is given)")
    wm.add_argument(
        "--font",
        type=values.path,
        metavar="FILE",
        help="a TrueType or OpenType font (default: the built-in one, else an installed one that draws the text)",
    )
    wm.add_argument("--color", metavar="#RRGGBB", help="text colour (default: #FFFFFF)")
    wm.add_argument("--logo", type=values.path, metavar="IMAGE", help="an image to place as a logo")
    for prefix, what, defaults in (
        ("", "text", share.Watermark(None).text_layer),
        ("logo-", "logo", share.Watermark(None).logo_layer),
    ):
        wm.add_argument(
            f"--{prefix}position", choices=share.ANCHORS, help=f"where the {what} goes (default: {defaults.anchor})"
        )
        wm.add_argument(
            f"--{prefix}size",
            type=finite,
            metavar="FRACTION",
            help=f"{what} size as a fraction of the short edge (default: {defaults.size})",
        )
        wm.add_argument(
            f"--{prefix}margin",
            type=finite,
            metavar="FRACTION",
            help=f"distance from the edges (default: {defaults.margin})",
        )
        wm.add_argument(f"--{prefix}opacity", type=finite, help=f"{what} opacity (default: {defaults.opacity})")
    wm.add_argument("--auto-place", action=SWITCH, help="move the marks to the calmest mirrored corner")
    wm.add_argument("--auto-tone", action=SWITCH, help="grey the marks to stand out from what lies under them")

    b = sub.add_parser("batch", help="run commands listed one per line in a JSONL file, printing a line for each")
    b.add_argument(
        "jobs",
        metavar="FILE",
        help='JSON lines such as ["gen", "--prompt", "cat"] or {"id": "a", "args": [...]}; - reads standard input',
    )
    b.add_argument(
        "--skip-done",
        metavar="RESULTS",
        help="a previous batch's output: the jobs that succeeded there are skipped",
    )
    b.add_argument("--stop-on-error", action=SWITCH, help="stop at the first job that fails")
    b.add_argument("--dry-run", action=SWITCH, help="run every job as its own dry run, sending nothing paid")
    b.add_argument(
        "--interval",
        metavar="S|MIN-MAX",
        help="seconds from the last request to the next job's first, or a random wait MIN-MAX (default: [batch] interval)",
    )

    gr = sub.add_parser(
        "grid",
        help="run gen, i2i or inpaint once per cell of settings varied along two axes, and save a sheet of the images",
    )
    for axis, where in (("--x", "columns"), ("--y", "rows")):
        gr.add_argument(
            axis,
            action="append",
            metavar="PARAM=VALUES",
            help=f"settings that change across the {where}; repeat to change several together. PARAM is one of "
            f"{', '.join(grid.PARAMS)}. Numbers take 20,24,28 or a range 20-28:4; seeds also r or r5 for random ones; "
            "prompt and neg take one per line, or @FILE",
        )
    gr.add_argument("--sheet", action=SWITCH, help="save NAME_grid.png of the images beside the first (default: on)")
    gr.add_argument("--stop-on-error", action=SWITCH, help="stop at the first cell that fails")
    gr.add_argument("--dry-run", action=SWITCH, help="price every cell as the run would go, sending nothing paid")
    gr.add_argument(
        "--interval",
        metavar="S|MIN-MAX",
        help="seconds from the last request to the next cell's first (default: [batch] interval)",
    )
    gr.add_argument("job", nargs=argparse.REMAINDER, metavar="COMMAND ...", help="the command and its options")

    m = sub.add_parser("meta", help="read generation settings from images")
    m.add_argument(
        "image", nargs="+", type=values.path, help="images or folders; several, or a folder, print one JSON line each"
    )
    m.add_argument("--raw", action=SWITCH, help="include the raw Comment")
    m.add_argument("--jsonl", action=SWITCH, help="one JSON line per image and a summary, even for one image")
    m.add_argument("-r", "--recursive", action=SWITCH, help="also read the folders inside folders given")

    lt = sub.add_parser("lint", help="check prompts for common mistakes")
    _add_generation_args(lt)
    _add_import_args(lt)

    tg = sub.add_parser("tags", help="suggest tags the way the official prompt box does")
    tg.add_argument("query", nargs="+", help="start of a tag, or a Japanese word; give several to look up each")
    tg.add_argument("--model", help=MODEL_HELP)
    tg.add_argument("--furry", action=SWITCH, help="look up the furry tag list")

    sub.add_parser("models", help="list the models and what each supports")

    c = sub.add_parser("cost", help="estimate the Anlas cost of a request")
    c.add_argument("--model", help=MODEL_HELP)
    c.add_argument("--effort", choices=models.EFFORTS, help=EFFORT_HELP)
    c.add_argument("--size", metavar="WxH")
    c.add_argument("--steps", type=int)
    c.add_argument("--n", type=int)
    c.add_argument("--strength", type=finite, help="img2img strength, or inpaint strength with --inpaint (default: 1)")
    c.add_argument("--vibes", type=int, help="number of Vibes (default: 0)")
    c.add_argument("--encode", type=int, help="how many of the Vibes need encoding (default: 0)")
    c.add_argument("--refs", type=int, help="number of precise references (default: 0)")
    c.add_argument("--inpaint", action=SWITCH, help="price with the inpainting model")

    a = sub.add_parser("account", help="show the balance, list or switch accounts, or sign in with an email")
    a.add_argument("action", choices=["show", "list", "use", "login", "logout"], nargs="?", default="show")
    a.add_argument("name", nargs="?", help="the account to use, sign in as or sign out")
    a.add_argument(
        "--email",
        help="with login: the account's email; the password is asked for at a terminal, else read from standard input",
    )
    a.add_argument("--month", metavar="YYYY-MM", help="with show: the month to total (default: this month)")

    st = sub.add_parser("stats", help="total the requests in the ledger by day, command or model")
    st.add_argument("--days", type=int, metavar="N", help="the last N days, today included (default: 30)")
    st.add_argument("--since", metavar="YYYY-MM-DD", help="first day, instead of --days")
    st.add_argument("--until", metavar="YYYY-MM-DD", help="last day (default: today)")
    st.add_argument("--by", choices=ledger.GROUPINGS, help="grouping (default: day)")

    ig = sub.add_parser("interrogate", help="read the tags of an image with a local ONNX tagger, joined into a prompt")
    ig.add_argument("image", type=values.path)
    ig.add_argument(
        "--model", type=values.path, metavar="ONNX", help="the tagger model (default: tagger under [local])"
    )
    ig.add_argument(
        "--labels", type=values.path, metavar="FILE", help="its tag list (default: the one beside the model)"
    )
    ig.add_argument(
        "--kind", choices=[k.value for k in tagger.Kind], help="how the model reads images (default: from the names)"
    )
    ig.add_argument(
        "--threshold", type=finite, help=f"lowest score of general tags (default: {tagger.DEFAULT_THRESHOLD})"
    )
    ig.add_argument(
        "--character-threshold",
        type=finite,
        help=f"lowest score of character tags, and of copyright and artist tags without their own "
        f"(default: {tagger.DEFAULT_THRESHOLD})",
    )
    ig.add_argument(
        "--categories",
        metavar="LIST",
        help=f"comma list of {', '.join(c.value for c in tagger.CHOOSABLE)} "
        f"(default: {','.join(c.value for c in tagger.DEFAULT_CATEGORIES)})",
    )

    lc = sub.add_parser("local", help="check the optional models run on this computer, or download the SAM model")
    lc.add_argument("local_command", choices=["status", "download"], nargs="?", default="status")
    lc.add_argument("model", nargs="?", choices=["sam"], help="with download: the model to fetch")
    lc.add_argument("--dry-run", action=SWITCH, help="with download: show what would be fetched")

    cf = sub.add_parser("config", help="show the config, or create it with init")
    cf.add_argument("action", choices=["show", "init"], nargs="?", default="show")
    cf.add_argument("--force", action=SWITCH, help="with init, replace an existing config file with the template")

    return p


def _add_copy_sources(p: argparse.ArgumentParser, label: str) -> None:
    p.add_argument(
        "images", nargs="+", type=values.path, metavar="IMAGE", help="images, or folders for the images in them"
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help=f"a .png name for one image, or a folder (default: NAME_{label}.png in the output folder)",
    )
    p.add_argument("-r", "--recursive", action=SWITCH, help="also copy the images in the folders inside folders given")
    p.add_argument(
        "--rename",
        action=SWITCH,
        help="name copies by date and number, such as 20261125-0001.png, not after their source",
    )


def _add_copy_args(p: argparse.ArgumentParser, label: str) -> None:
    _add_copy_sources(p, label)
    p.add_argument(
        "--keep-metadata",
        action=SWITCH,
        help="carry the source's generation settings over, from PNG text, EXIF or the alpha channel, as PNG text; "
        "off by default, as for strip",
    )


def _add_run_args(p: argparse.ArgumentParser, show_body: bool = False, output: bool = True) -> None:
    run = p.add_argument_group("run")
    if output:
        run.add_argument(
            "-o",
            "--output",
            metavar="PATH",
            help="a .png name for the images, numbered from the second on, or a folder to save them in "
            "without day folders",
        )
    run.add_argument("--dry-run", action=SWITCH, help="build the request and estimate the cost without sending it")
    run.add_argument("--yes", action=SWITCH, help="run even when the estimate exceeds max_per_request")
    if show_body:
        run.add_argument(
            "--show-body", action=SWITCH, help="include the request body in the output, without image data"
        )


def _add_generation_args(p: argparse.ArgumentParser) -> None:
    text = p.add_argument_group("prompt")
    text.add_argument(
        "--prompt", metavar="TEXT", help="image description; @FILE reads it from a file, @- from standard input"
    )
    text.add_argument(
        "--random-prompt",
        nargs="?",
        const=random_prompt.Recipe.CORRECTED.value,
        choices=[r.value for r in random_prompt.Recipe],
        help="write a random prompt and characters, as the official dice button does; official follows its code "
        "exactly, which never draws entries that need flags (default: corrected)",
    )
    text.add_argument(
        "--neg", metavar="TEXT", help="negative prompt; @FILE reads it from a file, @- from standard input"
    )
    text.add_argument(
        "--neg-verbatim",
        action=SWITCH,
        help="send the negative prompt as is, without UC preset or nsfw; for reproducing images from other clients",
    )
    text.add_argument(
        "--comfyui-characters",
        action=SWITCH,
        help="split a ComfyUI multi-character prompt (COUPLE, AND with MASK or AREA, a character per | line) into "
        "the base prompt and characters placed at their regions' centres (default: off)",
    )
    text.add_argument(
        "--official-nsfw",
        action=SWITCH,
        help="add nsfw to the negative prompt unless the prompt contains it, as the official client does. "
        "Off by default: nsfw is never added, and removed from the negative prompt when the prompt contains it",
    )
    text.add_argument(
        "--format-prompt",
        action=SWITCH,
        help="tidy prompts before sending: full-width commas, extra whitespace and empty tags. "
        "Quoted text, text: segments, <markers> and kaomoji are left untouched",
    )
    text.add_argument(
        "--space-numeric-close",
        action=SWITCH,
        help="insert a space when a closing :: follows a number, so the number is not read as a new weight",
    )
    text.add_argument(
        "--convert-sd-weights",
        action=SWITCH,
        help="convert SD weights such as (tag:1.2) to 1.2::tag:: before sending",
    )
    text.add_argument(
        "--fixed-tags",
        action=SWITCH,
        help="join the config's prompt_prefix, prompt_suffix, negative_prefix and negative_suffix around the "
        "prompts typed (default: on)",
    )
    text.add_argument("--char", action="append", metavar="TEXT", help="character prompt; repeat for each character")
    text.add_argument(
        "--char-neg", action="append", metavar="TEXT", help="character negative prompt, matched to --char in order"
    )
    text.add_argument(
        "--char-pos", action="append", metavar="X,Y", help="character centre within 0-1, matched to --char in order"
    )
    params = p.add_argument_group(
        "parameters", "Unset values come from [defaults] in the config file, then from the model."
    )
    params.add_argument("--model", help=MODEL_HELP)
    params.add_argument("--effort", choices=models.EFFORTS, help=EFFORT_HELP)
    params.add_argument(
        "--size", metavar="WxH", help="such as 832x1216, or a preset such as portrait, landscape or square"
    )
    params.add_argument("--seed", type=int, help="random when omitted")
    params.add_argument("--steps", type=int, help="sampling steps")
    params.add_argument("--cfg", type=finite, help="prompt guidance")
    params.add_argument("--sampler", help=f"one of {', '.join(models.SAMPLERS)}")
    params.add_argument("--noise-schedule", choices=models.NOISE_SCHEDULES)
    params.add_argument("--cfg-rescale", type=finite, help="prompt guidance rescale, 0-1")
    params.add_argument("--quality", help="quality tags: standard, light or none")
    params.add_argument("--uc-preset", help="heavy, light, human, furry or none")
    params.add_argument("--n", type=int, help="images per request (default: 1); the limit depends on the size")
    params.add_argument("--transparent", action=SWITCH, help="transparent background (V5)")
    params.add_argument(
        "--premultiplied-alpha",
        action=SWITCH,
        help="transparent images use premultiplied alpha instead of straight alpha (V5)",
    )
    params.add_argument("--furry", action=SWITCH, help="furry mode: start the prompt with fur dataset")
    params.add_argument(
        "--auto-text",
        action=SWITCH,
        help="copy quoted text into a teXt: block (V5, default: on); --no-auto-text turns it off",
    )
    params.add_argument("--variety", action=SWITCH, help="Variety+ (V4.5, V4)")
    params.add_argument("--legacy-uc", action=SWITCH, help="the legacy handling of the negative prompt (V4)")
    refs = p.add_argument_group("references")
    refs.add_argument(
        "--vibe",
        action="append",
        metavar="FILE[:STRENGTH[:INFO]]",
        help="a .naiv4vibe or .naiv4vibebundle file, or an image (V4.5, V4)",
    )
    refs.add_argument(
        "--vibes-from",
        type=values.path,
        metavar="IMAGE",
        help="use the Vibes kept in a generated image's settings, as the official import does (V4.5, V4)",
    )
    refs.add_argument(
        "--normalize-vibe-strength",
        dest="normalize_vibes",
        action=SWITCH,
        help="scale several Vibe strengths down to a sum of 1 (default: on); --no-normalize-vibe-strength sends "
        "them as given",
    )
    refs.add_argument(
        "--ref",
        action="append",
        metavar="IMAGE[:KIND[:STRENGTH[:FIDELITY]]]",
        help="precise reference; KIND is both (default), both-any-outfit, character, character+outfit, style or "
        "outfit (V5), or both, character or style (V4.5)",
    )
    refs.add_argument("--ref-strength", type=finite, help="strength of every precise reference, over their own")
    refs.add_argument("--ref-fidelity", type=finite, help="fidelity of every precise reference, over their own (V4.5)")


def _add_import_args(p: argparse.ArgumentParser) -> None:
    source = p.add_argument_group("import", "Settings from an image, as the official import dialog brings in.")
    source.add_argument(
        "--from-image",
        type=values.path,
        metavar="IMAGE",
        help="start from an image's settings; options given override them",
    )
    source.add_argument(
        "--import",
        dest="import_parts",
        metavar="PARTS",
        help=f"what --from-image brings in, from {', '.join(import_image.PARTS)} or all "
        f"(default: {','.join(import_image.DEFAULT_PARTS)})",
    )
    source.add_argument(
        "--clean-brackets",
        action=SWITCH,
        help="drop [ ] { } from the imported prompts, as the import dialog can",
    )
    source.add_argument(
        "--actual-prompts",
        action=SWITCH,
        help="import the prompts as the image drew them, each ||a|b|| group replaced by the option it took",
    )


def _add_image_strength_args(p: argparse.ArgumentParser, strength: str, noise: str) -> None:
    p.add_argument("--image", type=values.path, required=True, help="source image")
    p.add_argument("--strength", type=finite, help=f"how much the image may change, 0.01-0.99 (default: {strength})")
    p.add_argument("--noise", type=finite, help=f"noise added to the source image, 0-0.99 (default: {noise})")


def _add_inpaint_args(ip: argparse.ArgumentParser) -> None:
    area = ip.add_argument_group("area", "Combine any of --mask, --region and --expand.")
    area.add_argument(
        "--mask",
        type=values.path,
        metavar="IMAGE",
        help="white or opaque areas are repainted; stretched to the source size",
    )
    area.add_argument(
        "--region",
        action="append",
        metavar="SHAPE",
        help="rect:x1,y1,x2,y2, ellipse:x1,y1,x2,y2, poly:x,y,x,y,x,y,... or object:x,y,... (the object under "
        "the points, with the local SAM model) in fractions of the canvas. A leading - subtracts; repeat to apply "
        "several in order over --mask",
    )
    area.add_argument("--grow", type=int, metavar="PX", help="grow the mask by this many pixels (default: 0)")
    area.add_argument(
        "--expand",
        metavar="L,T,R,B",
        help="add canvas on each side in pixels, or one number for all sides; the new area is repainted",
    )
    ip.add_argument(
        "--focus",
        action=SWITCH,
        help="repaint a crop around the mask at about 1 MP. Automatic by default: on when the crop fits in 1 MP",
    )
    ip.add_argument(
        "--focus-padding",
        type=int,
        metavar="PX",
        help=f"context around the mask when focusing, 32-192 (default: {inpaint.DEFAULT_FOCUS_PADDING})",
    )
    ip.add_argument(
        "--inpaint-strength", type=finite, help="0.01-1; below 1, part of the source shows through (default: 1)"
    )
    ip.add_argument("--preview", action=SWITCH, help="save a preview of the repainted area instead of generating")
    ip.add_argument("--keep-raw", action=SWITCH, help="also save the image returned by the server before compositing")
