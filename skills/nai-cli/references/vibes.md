# Vibes and precise references

Vibes carry the look of an image into a generation; precise references carry a character, a style or, on V5, an outfit.

## Which models

`--vibe` and `--vibes-from` need a V4.5 or V4 model (`--model v4.5`); on V5 they fail with `model_unsupported`, so tell the user and let them pick the model. `--ref` works on V5 and V4.5, but on V5 not with `nai inpaint`.

## Precise references

- `--ref IMAGE[:KIND[:STRENGTH[:FIDELITY]]]`. KIND: `both` (default), `character`, `style`; on V5 also `outfit` (clothes only), `character+outfit` (keep the outfit) and `both-any-outfit` (let it change).
- V5 ignores FIDELITY. V5 Medium switches to V5 Full when a reference is given, which costs more; `warnings` says so.
- `--ref-strength N` sets every reference's strength; a grid varies it with `--x pr_strength=0.5,1,1.5`.

## Using Vibes

- `--vibe` takes `.naiv4vibe` and `.naiv4vibebundle` files as well as images. An image is encoded first, for 2 Anlas, unless it was encoded with the same settings before.
- `--vibes-from IMAGE` reuses the Vibes kept in a generated image for the model that made the image, without encoding them again. A different target model cannot re-encode them: the settings hold encodings, not the source images.
- Generation costs an extra 5 Anlas per reference per output image, even with Opus. Two output images with one precise reference add 10 Anlas to the base price.

## Saving Vibes

`nai vibe SOURCE...` saves Vibes as a `.naiv4vibe` file, or a `.naiv4vibebundle` for several, for 2 Anlas per image encoded, or for nothing when the images were encoded with the same settings before. Check it with `--dry-run` first, as with generation. Reuse the saved file with `--vibe` to avoid paying again.
