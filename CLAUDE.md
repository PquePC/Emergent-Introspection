# CLAUDE.md — read before acting in this repository

## What this project is

Research on whether language models detect activation-level concept injection differently when the
injected concept is **harmful**. See `README.md` for the science and `docs/project-plan.md` for the
method.

The work involves two sensitive artifacts: a **refusal-ablated (uncensored) Gemma3-27B**, and
**harmful concept vectors** (`weapon`, `poison`, `assault`). Neither is published or shared. The
measured output is a detection *rate*. **Rates are safe to publish; artifacts are not.**

---

## 🛑 Hard rules — never do these, and never propose them

1. **Never upload ablated model weights** anywhere: Hugging Face (public *or* private), Drive,
   gists, pastebins. There is no upside — the ablation rebuilds from a published config in minutes.
   Regenerate, never archive.
2. **Never expose the model on a public port or proxy.** An unauthenticated Jupyter/Gradio/vLLM
   endpoint on a rented pod *is* a public uncensored model.
3. **Never commit vectors or activations** (`*.pt`, `*.safetensors`, `*.npy`, `vectors/`,
   `activations/`). Publish cosine values instead: scalars reconstruct nothing.
4. **Never commit raw generations** (`results/`, `outputs/`, sweep logs). They contain what an
   uncensored model said with `weapon` injected. Aggregate rates and curated examples only.
5. **Never put funding strategy, cost tables, mentor framing, or novelty positioning in public docs
   or commit messages.** Public docs carry science only; that material lives in `private/`
   (gitignored).

## ⚠️ Stop and ask for explicit approval

Ask in chat and wait for a typed yes. A plan is not approval; approval for one action is not
approval for the next.

- Any `git push`
- Any first commit touching `results/`, `vectors/`, or notebook outputs
- Any upload to an external host, of anything
- Opening any pod port, proxy, or public URL
- Sending more than concept words and short self-reports to a third-party judge API
- Publishing per-concept detail rather than per-arm aggregates (per-concept = a lookup table)

If an action would make this work **more useful to someone attacking a model than to someone
auditing one**, raise it rather than doing it.

## ✅ Before any push

```bash
git status --short
git check-ignore -v private/     # expect: .gitignore:2:private/
git diff --cached | grep -inE '\$[0-9]|budget|funding|grant|mentor|novelty|scoop'
```

The last line must return nothing.

---

## Full detail

`private/OPERATIONAL-SECURITY.md` (local only, gitignored) has the complete list, the pod hygiene
rules, the judge-payload scope-creep warning, and the disclosure procedure. **Read it before any
release-shaped action.** If it is missing from your checkout, that file is local to the author's
machine and the rules above still apply in full.

## Repo orientation

| Question | File |
|---|---|
| What is the project, why it matters | `README.md` |
| Method, decision gates, deliverables | `docs/project-plan.md` |
| Ethics position, risk register, fallback design | `docs/risks-and-ethics.md` |
| Literature, what is contested | `docs/background.md` |
| References | `BIBLIOGRAPHY.md` |

`private/` is gitignored and stays that way. Do not move anything out of it into the public tree
without asking.
