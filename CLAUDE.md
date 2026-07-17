# CLAUDE.md — read before acting in this repository

## What this project is

Research on whether language models detect activation-level concept injection differently when the
injected concept is **harmful**. See `README.md` for the science and `docs/project-plan.md` for the
method. The measured output is a detection **rate** — rates are safe to publish.

## Sensitive artifacts

The work produces two things that must never leave a machine you control:

1. A **refusal-ablated (uncensored) Gemma3-27B** — a model that will comply with harmful requests.
2. **Harmful concept vectors** (`weapon`, `poison`, `assault`) and the raw activations behind them.

Publishing the detection *rate* is the goal. Publishing either artifact would hand someone a working
uncensored model or reusable attack vectors. **Rates are safe; artifacts are not.**

---

## 🛑 Hard rules — never do these, and never propose them

1. **Never upload ablated model weights** anywhere: Hugging Face (public *or* private), Drive,
   gists, pastebins. The ablation rebuilds from a published config in minutes — regenerate, never
   archive.
2. **Never expose the model on a public port or proxy.** An unauthenticated Jupyter/Gradio/vLLM
   endpoint on a rented pod *is* a public uncensored model.
3. **Never commit vectors, activations, or raw generations** (`*.pt`, `*.safetensors`, `*.npy`,
   `vectors/`, `activations/`, `results/`, `outputs/`, sweep logs). Vectors are reusable attack
   artifacts; generations are what an uncensored model said with `weapon` injected. Publish cosine
   scalars and aggregate rates instead — they reconstruct nothing.

## ⚠️ Stop and ask for explicit approval before

- Any upload to an external host, of anything
- Opening any pod port, proxy, or public URL
- Sending more than concept words and short self-reports to a third-party judge API
- Publishing per-concept detail rather than per-arm aggregates (per-concept = a lookup table)
- Any push whose diff carries any of the above

If an action would make this work **more useful to someone attacking a model than to someone
auditing one**, raise it rather than doing it.

## Repo orientation

| Question | File |
|---|---|
| What is the project, why it matters | `README.md` |
| Method, decision gates, deliverables | `docs/project-plan.md` |
| Ethics position, risk register, fallback design | `docs/risks-and-ethics.md` |
| Literature, what is contested | `docs/background.md` |
| References | `BIBLIOGRAPHY.md` |
