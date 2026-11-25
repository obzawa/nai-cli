# Changelog

## Unreleased

- Precise references on V5 Full and V5 Curated, prepared in the official size buckets and keeping transparency. A reference with V5 Full Medium runs V5 Full, as in the official client. V5 takes no precise reference when inpainting.
- Outfit references: `--ref` kinds `outfit`, `character+outfit` and `both-any-outfit` on V5.
- `--ref-strength` and `--ref-fidelity` set every precise reference at once, and `nai grid` varies them as `ref_strength` and `ref_fidelity` (also `pr_strength`, `pr_fidelity`).
- `nai models` reports `reference_inpainting`, `reference_fidelity` and `outfit_reference`.
- `nai interrogate` reads the tag categories of CL tagger v2; every tag was taken as `other`.
- PNG tEXt and zTXt text is read as Latin-1, as the PNG specification and NovelAI write it; a no-break space or accented letter came out as a replacement character.
- `strip`, `redact` and `watermark` copies also lose stealth settings hidden in the colours (`stealth_rgbinfo`, `stealth_rgbcomp`).
- `strip`, `redact` and `watermark` take folders, `-r` for the folders inside them, keeping each copy's place under an `-o` folder.
- `--rename` names sharing copies by date and number, and a copy named after the official download name, which holds the start of the prompt and the seed, warns.
- `watermark` draws Chinese, Japanese, Korean and other text the built-in font lacks in an installed font, without `--font`.
- `nai account login` asks for the password at a terminal without showing it; piped input works as before.

## 0.1.0 - 2026-11-20

First release.
