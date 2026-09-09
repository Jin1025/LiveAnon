# LiveAnon demo site

Static GitHub Pages site prepared for `https://jin1025.github.io/LiveVoice/`.

## Contents

- `docs/index.html` — project page
- `docs/styles.css` — page styling
- `docs/assets/audio/` — 4 LibriTTS examples × Original / SVA / LiveAnon
- `docs/assets/architecture-placeholder.svg` — replace with the final architecture image

## Add the main figure

1. Put the final figure at `docs/assets/architecture.png`.
2. In `docs/index.html`, replace:
   `assets/architecture-placeholder.svg`
   with:
   `assets/architecture.png`
3. Remove the small replacement note below the figure.

## Activate Paper / Code links

In `docs/index.html`, replace the placeholder `href="#"`, remove the `disabled` class,
and delete the `Coming soon` spans when URLs are ready.

## GitHub Pages

Commit the `docs/` folder to the `main` branch, then in GitHub:
Settings → Pages → Deploy from a branch → `main` / `/docs`.
