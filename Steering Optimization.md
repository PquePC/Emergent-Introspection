# Steering Optimization — moved

This directory became its own repository on 2026-08-11:

**https://github.com/PquePC/steering-optimization**

All 48 commits that touched it moved with it (`git subtree split`), so its history is intact
there and remains readable here up to this commit. Nothing was lost.

## Why it moved

It had grown into a self-contained tool — the M2 pipeline, its specification, its contract and
its test suite — with a different audience from the research write-up. This repo keeps the
science: the question, the methodology, the ethics register and the literature.

## If you are looking for

| | |
|---|---|
| The pipeline, and how to run it | `steering-optimization`, start at `m2/QUICKSTART.md` |
| What each measure means | `steering-optimization`, `M2 — Specification.md` |
| The research question and why it matters | `README.md`, here |
| Ethics position and risk register | `docs/risks-and-ethics.md`, here |

The dual-use rules in `CLAUDE.md` apply to both repositories. The new one carries its own copy.

## On the pod

The clone path changed. There is no longer a nested project directory, and no space in the path:

```
/workspace/steering-optimization          # was /workspace/Emergent-Introspection/Steering Optimization
```

Run data still lives outside the repo at `/workspace/m2_runs` and is unaffected.
