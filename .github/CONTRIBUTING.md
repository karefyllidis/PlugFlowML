# Contributing

## Git commits and attribution

Do not add Cursor (or any automated coding agent) as a co-author. Commit messages must not include trailers such as:

`Co-authored-by: Cursor <…>`

This repository ships a **commit-msg** hook under `.githooks/` that **removes** any `Co-authored-by:` line whose metadata refers to Cursor (e.g. `cursor.com`, `cursoragent`, or the word `cursor` in that trailer). Install it once from the repo root:

```bash
git config core.hooksPath .githooks
```

On Unix-like shells you may mark the hook executable:

```bash
chmod +x .githooks/commit-msg
```

Git for Windows runs hooks via its bundled `sh`; the hook should work after setting `core.hooksPath` only.

If you use another hook directory locally, copy or symlink `.githooks/commit-msg` into your hook path instead.

## Code and docs

Match existing style in the files you change; keep changes focused on the task. See `README.md` and `STRUCTURE.md` for layout and conventions.

## Git and large files

Do not commit generated training data, model weights, SLURM progress logs, or local secrets unless the project explicitly requires it. Those paths are listed in `.gitignore` and summarized under **Version control** in `README.md`. If you add a new generated artifact path in code (e.g. a second figure export root), update `.gitignore` and the README table in the same change so contributors stay consistent.
