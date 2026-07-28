# G3 — geometry premise check: running results

*Running log of the G3 geometry check across models. G3 tests the premise the whole hypothesis rests on:*
***are harmful concept vectors more refusal-aligned than benign ones at the injection layer (~60% depth)?***

Method and decision rule: [`execution-guide.md` Stage G3](execution-guide.md). Extraction follows Arditi et al.
(difference-of-means `d_refusal`) and Lindsey-style concept vectors, swept across all layers and two pooling
protocols, with a top-k refusal-subspace projection as the cone-robustness check.

> 🔒 Outputs here are **per-arm aggregate cosines only** — safe to keep ([CLAUDE.md](../CLAUDE.md)). No
> per-concept values (that would be a lookup table), no vectors, no activations.

## Summary

| Model | Family | Injection layer | Harmful > benign? | Stable across protocols? | Verdict |
|---|---|---|---|---|---|
| `gemma-2-2b-it` | Gemma-2 (**pilot**) | L15 / 26 (0.6) | yes — Δ **+0.293** (last) | **yes** | premise holds *(pilot)* |
| `gemma-3-4b-it` | Gemma-3 (target family) | — | *pending* | *pending* | *pending* |
| `gemma-3-27b-it` | Gemma-3 (**target**) | L37 / 62 (0.6) | *not run — funded stage* | — | *pending* |

> The pilot validates the code path end-to-end and shows the premise is **plausible**. It does **not** settle
> it: harm/refusal coupling is family-specific (HARC depth-decoupling), so the decisive read for the project
> is on the **Gemma-3** family, not Gemma-2.

---

## `gemma-2-2b-it` (pilot) — premise holds

Injection layer **L15 / 26** (fraction 0.6). Config: 120 harmful/harmless pairs for `d_refusal`, ~10
concepts per arm, 100 baseline words, top-5 refusal subspace.

`cos(v_concept, d_refusal)` at the injection layer, per arm:

| Arm | last-token | max-pool | subspace proj. (last-token) |
|---|---|---|---|
| **harmful** | **+0.310** | **+0.172** | **0.341** |
| harm-adjacent | +0.099 | +0.078 | 0.171 |
| valence | +0.052 | +0.054 | 0.120 |
| benign | +0.018 | +0.039 | 0.144 |

Δ(harmful − benign): **+0.293** (last-token), **+0.134** (max-pool). Ordering **stable across both protocols**.

**Reading:**
- Harmful is clearly the most refusal-aligned arm, by a wide margin.
- **It's harmfulness, not valence** — valence (+0.052) sits on top of benign (+0.018); negative emotion alone
  does not align with refusal.
- **It's harmfulness, not mere topic** — harm-adjacent (+0.099) is elevated but ~3× below harmful. Danger
  proximity contributes a little; genuine harmfulness contributes most. Arms 2 and 3 are doing their job.
- Coupling is strong *at* the injection depth and does not decouple toward orthogonality before L15 in this
  model — addresses the HARC concern for the pilot.

**Caveat:** Gemma-2, not the Gemma-3 target family. Pending Gemma-3-4B (family-correct) and, when funded,
Gemma-3-27B at the true operating point (L37).

---

## `gemma-3-4b-it` — pending

Run G3.1–G3.4 with `G3_MODEL = "google/gemma-3-4b-it"`. Record here: `cos`-by-arm at the injection layer
(both protocols), ordering stability, and the subspace projection. The read to watch: does harmful stay
clearly above benign at ~60% depth, with valence ≈ benign and harm-adjacent intermediate, as in the pilot?
