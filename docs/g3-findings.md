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
| `gemma-3-4b-it` | Gemma-3 (target family) | L20 / 34 (0.6) | yes — Δ **+0.428** (last) | yes (harmful vs benign) | holds; **valence confound** at last-token |
| `gemma-3-27b-it` | Gemma-3 (**target**) | L37 / 62 (0.6) | *not run — funded stage* | — | *pending* |
| `Qwen2.5-3B-Instruct` | Qwen (cross-family) | L21 / 36 (0.6) | yes — Δ **+0.467** (last) | yes | holds; harm-specific (valence clean) |

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

## `gemma-3-4b-it` (target family) — premise holds, with a valence caveat

Injection layer **L20 / 34** (fraction 0.6). Same config (120 pairs, ~10 concepts/arm, 100 baseline words,
top-5 subspace). Note: `gemma-3-4b-it` is **multimodal** — activations read from the text stack
(`config.text_config`).

`cos(v_concept, d_refusal)` at the injection layer, per arm:

| Arm | last-token | max-pool | subspace proj. (last-token) |
|---|---|---|---|
| **harmful** | **+0.607** | **+0.156** | 0.694 |
| valence | +0.477 | +0.065 | 0.565 |
| harm-adjacent | +0.268 | +0.060 | 0.659 |
| benign | +0.179 | −0.010 | 0.480 |

Δ(harmful − benign): **+0.428** (last-token), **+0.166** (max-pool). Harmful > benign **stable across both
protocols**.

**Reading — the core premise holds family-correct, but the confound structure is protocol-dependent:**
- ✅ **Harmful is the top arm at the injection layer under both protocols** — the family-correct confirmation
  the project needed.
- ⚠️ **Valence is a live confound at the last-token protocol.** cos(valence) = +0.477 sits much closer to
  harmful (+0.607) than to benign (+0.179) — negative-valence concepts are *also* strongly refusal-aligned in
  Gemma-3. This did **not** appear in the Gemma-2 pilot (valence ≈ benign there). Under max-pool, valence
  collapses back to ≈ benign (+0.065 vs −0.010) and harmful is cleanly on top, but at much smaller magnitudes.
- The last-token protocol is the one that **matches how concept vectors are actually built and injected**
  (harness construction), so the valence elevation is the operationally relevant case, not a curiosity.
- The top-k subspace projection is **weakly discriminating** here — all arms fall in 0.48–0.69, harmful only
  marginally above harm-adjacent (0.694 vs 0.659). At this depth the refusal subspace captures a large
  fraction of every arm.
- The last-token cos trace is **volatile across layers** (large negative excursions ~L10–12); max-pool is
  smooth. L20 sits past the volatile region.

**Implications for the main experiment:**
1. **Arm 2 (negative-valence) is load-bearing, now confirmed empirically for the family.** A harmful-arm
   detection drop cannot be attributed to *harmfulness* without it, because harm and valence are
   geometrically close at the injection protocol. The dissociation becomes a **behavioural** question
   (Stage 3 arm contrast) — exactly what the four-arm design exists for.
2. **Pre-register the extraction protocol** and report cos per protocol per arm — harm-vs-valence
   separability depends on it (the 73° sensitivity, realised).
3. **Confirm at Gemma-3-27B, L37** when funded — the 4B is family-correct but neither the target model nor
   the exact operating point.

**Verdict:** premise holds (harmful > benign, stable across protocols, family-correct). G3 does not block
Stage 1. Carry the valence caveat forward as a first-class interpretive constraint on Chart 1 / Chart 2.

---

## `Qwen2.5-3B-Instruct` (cross-family) — premise holds, harm-specific

Injection layer **L21 / 36** (fraction 0.6). Same config. Run to test whether the Gemma-3 valence
entanglement is family-general or Gemma-specific. (Qwen is ungated, flat-config, and supports the system
role — no notebook changes needed to run it.)

`cos(v_concept, d_refusal)` at the injection layer, per arm:

| Arm | last-token | max-pool | subspace proj. (last-token) |
|---|---|---|---|
| **harmful** | **+0.441** | **+0.054** | **0.496** |
| harm-adjacent | +0.196 | +0.039 | 0.260 |
| valence | +0.176 | +0.023 | 0.291 |
| benign | −0.026 | −0.021 | 0.270 |

Δ(harmful − benign): **+0.467** (last-token) — the largest of the three families — and +0.075 (max-pool).
Stable across protocols.

**Reading — the cleanest separation of the three:**
- Harmful is decisively the top arm (+0.441); valence (+0.176) and harm-adjacent (+0.196) are both modest and
  clustered well below it, benign slightly negative.
- **The Gemma-3 valence entanglement does not replicate.** Valence sits near harm-adjacent and far from
  harmful — the Gemma-2-pilot pattern, not the Gemma-3-4B one.
- The subspace projection cleanly discriminates: harmful **0.496** vs everything else ~0.26–0.29 (~1.7×),
  unlike Gemma-3-4B where all arms crowded 0.48–0.69.

---

## Cross-family synthesis (3 families)

| Model | Premise (harmful > benign) | Harm-specificity (valence separable) |
|---|---|---|
| Gemma-2-2B | ✅ holds | ✅ clean (valence ≈ benign) |
| **Gemma-3-4B** | ✅ holds | ⚠️ **valence entangled at last-token** |
| Qwen2.5-3B | ✅ holds (largest Δ) | ✅ clean (valence ≈ harm-adjacent, ≪ harmful) |

**Two conclusions:**
1. **The premise is robust and family-general** — harmful concept vectors are more refusal-aligned than benign
   ones at ~60% depth in every family tested. The core hypothesis has purchase.
2. **The valence entanglement is Gemma-3-specific, not fundamental.** It appears in Gemma-3-4B but in neither
   Gemma-2 nor Qwen. Since the **target is Gemma-3-27B**, this makes **arm 2 (valence) essential for the target
   experiment specifically** — but it is a feature of Gemma-3's refusal geometry, not a universal confound in
   the method. Whether it persists at 27B / L37 is the thing to check first when funded.
