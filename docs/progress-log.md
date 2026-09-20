# Progress Log

One entry per closed issue, written right after that issue's PR is opened. Purpose: a running, low-overhead record of what was actually implemented and why — separate from the PR diffs themselves, readable without digging through git history, and useful when writing the Project 1/2 reports later.

Each entry is written by a lightweight model pass (not the main working session) to keep this cheap to maintain. Format per entry:

```
## #<issue number> — <issue title>
**Date:** YYYY-MM-DD · **PR:** <link> · **Branch:** <name>

**What happened:** 2-4 sentences, plain language, what was built/fixed and why.

**Code and packages touched:**
- `path/to/file.py` — one line on what it does
- New dependency: `package-name` — why it was needed (omit if none)

**Verification:** how it was checked (tests run, real-data check pending in Colab, etc.)

**Follow-ups / open questions:** anything left for a later issue (omit if none)
```

---
