# Git hooks

- **commit-msg** – Removes any `Co-authored-by: ... Cursor ...` / `... AI ...` line from commit messages so Cursor AI is never mentioned as co-author.

To use these hooks in this repo (run once):

```bash
git config core.hooksPath .githooks
chmod +x .githooks/commit-msg
```
