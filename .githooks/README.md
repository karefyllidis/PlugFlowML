# Git hooks

- **commit-msg** – Removes any `Co-authored-by:` line that mentions Cursor/AI (e.g. `Co-authored-by: Cursor <cursoragent@cursor.com>`) so Cursor is never listed as co-author.

**You must enable the hook** or it will not run. From the repo root, run once:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/commit-msg
```

Then future commits will have those lines stripped automatically.
