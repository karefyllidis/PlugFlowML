# Git hooks

## commit-msg

Removes any `Co-authored-by:` line that refers to **Cursor** (e.g. `Co-authored-by: Cursor <cursoragent@cursor.com>`, or any trailer containing `cursor` in the attribution). Other co-authors are left unchanged.

This is intentionally narrow: it does not strip lines just because they contain the letters "AI".

**Enable once** from the repository root:

```bash
git config core.hooksPath .githooks
```

On Unix-like systems you may also run:

```bash
chmod +x .githooks/commit-msg
```

See `.github/CONTRIBUTING.md` for the project policy on attribution.
