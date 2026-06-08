# Deep Research Report: doc2lora / code2lora Architecture Review

**Process**: 108 agents, 6 search angles, 25 sources fetched, 118 claims extracted, 25 verified via 3-vote adversarial verification → 19 confirmed, 6 refuted.

## Headline finding

**Doc2lora's per-layer per-module Perceiver+EinMix head is significantly over-parameterized relative to what the literature shows is needed.** T2L itself doesn't generate per-(layer, module) weights with independent slots — it uses a **shared MLP backbone conditioned on learnable depth embedding E[l] and module-type embedding E[m]**, batched in one forward pass. And on T2L's own ablation, the L variant (full A,B output) vs M variant (shared head, 38% fewer params) differs by only **0.2 points** on 10-task zero-shot avg (67.7 vs 67.5). Zhyper (Oct 2025) pushes this further: fixed per-(layer, module) A and B, generate only a rank-r modulation vector z, achieves 26× parameter reduction with only 0.5% performance gap.

This reframes the GRU bottleneck problem entirely (see below).

---

## Findings per investigation area

### 1. Per-layer vs shared LoRA

- **The "shared vs per-layer" dichotomy in the prompt is wrong.** T2L's shared head is *already* per-layer because of E[l] conditioning — `phi_{m,l} = concat[f(z), E[m], E[l]]`. The right comparison is *shared backbone + depth/module embeddings* (T2L, Hyperdecoders, Zhyper, HyperFormer) vs *fully independent per-slot parameterization* (doc2lora). [arxiv.org/abs/2506.06105, arxiv.org/pdf/2203.08304]
- **T2L L vs M = 0.2 pts** (67.7 vs 67.5 on 10-task avg) despite L having 55M params and M having 34M. T2L vs Hyperdecoders (per-instance conditioning) = 67.7 vs 67.3 — essentially tied. Direct evidence that heavy per-slot parameterization buys very little. [T2L paper Table 2]
- **The "different layers need different ranks" argument (NormAL LoRA, AdaLoRA) is real** — deeper layers prefer higher rank, FFN modules prefer higher rank than attention K/O. But this is encoder/NLU evidence (DeBERTaV3-base/MNLI), and importantly **a shared head conditioned on E[l] CAN produce per-layer-varying outputs** — the adversarial verification refuted the strawman that shared heads can't represent layer variation. [aclanthology.org/2025.findings-emnlp.1074]
- **How to test for redundancy** (the prompt's question): log per-(layer, module) LoRA Frobenius norms and pairwise cosine similarity across training. If outputs collapse to near-identical across layers, the per-layer capacity is unused. **The repo already has `ctx-probe` for this** (commit 5ffc1fb). Run it.

### 2. The GRU bottleneck

- **The proposed 2048 → 327,680 single-linear expansion is architecturally pathological** compared to standard practice. Three independent precedents (T2L, Hyperdecoders, Zhyper) all use the same pattern: feed the conditioning vector + learnable (layer, module) embeddings into a shared MLP called once per slot, batched. None predict all slots from one linear layer. [arxiv.org/abs/2506.06105, arxiv.org/pdf/2203.08304, arxiv.org/abs/2510.19733]
- **Direct fix**: use the GRU's hidden state as the **task embedding `z`** fed to T2L's existing shared head, instead of bypassing the head. The doc2lora repo already contains T2L's reference impl at `src/ctx_to_lora/modeling/text_to_lora_impl.py` (lines 587-595 layer_depth_encoder, 793 cat_emb concat) — you can wire the GRU's hidden state into the place where the SentenceTransformer encoder feeds in. This preserves the GRU's sequential aggregation contribution and eliminates the 160:1 expansion entirely.
- **Deeper MLP doesn't fix the problem** — the issue isn't depth, it's the "one input → all slots" topology. A 2-layer projection still has to fan out the same 327k values.
- "GRU → shared per-type (A,B) matching code2lora" is the *more conservative* fallback, but it loses the layer-conditioning T2L has shown is useful and is a step backward from doc2lora.

### 3. Gemma 4 Profile C: local vs global attention

- **Confirmed**: Gemma 3n/4 E2B uses 4 local sliding-window layers per 1 global full-attention layer, sliding_window=512 by default. Layer types are computed as `'full_attention' if (i+1) % 5 == 0 else 'sliding_attention'` — global at indices 4, 9, 14, **19, 24, 29, 34**. [HF transformers gemma3n docs; configuration_gemma3n.py; E4B config.json]
- **Profile C (15-18, 20-23, 25-28, 30-33) is therefore LOCAL-ONLY** — it hits exactly the 4-of-5 sliding-window layers and skips every global-attention layer. This resolves the open question flagged in `code2lora/GEMMA4.md` line 20.
- **Architectural implication**: for a hypernet conditioning on potentially long context, you're excluding precisely the layers that actually integrate across the full sequence. This may be the wrong choice for context-conditioning specifically.
- **Critical**: `layer_types` is configurable per-checkpoint via the config. **Read it from the actual loaded config, don't hardcode the pattern.**

### 4. Gemma 4 integration gotchas

Research could not resolve these from literature; they remain GPU-verification items. What we did surface:
- Numerous reports of Gemma 3/4 + PEFT/bf16 issues in HF tracker (peft#3129, transformers#36814).
- One practitioner blog ("3 bugs in 30 minutes") and the principled-intelligence text-only repack exist, but neither addresses the specific "extract bare text model from VLM, project last_hidden_state → logits, stash lm_head" pattern Plan A proposes.
- **No published source validates the bound-method `.generate` delegation pattern** for `Gemma4ForConditionalGeneration` → bare text model. This needs a GPU smoke test before committing.

### 5. Training stability

- **The `lora_alpha` question**: the rsLoRA blog (Kalajdzievski, HF) argues `alpha/sqrt(r)` scaling beats `alpha/r` at higher ranks, but at r=8 the difference is small. Setting `lora_alpha=8` (scaling=1.0) is a reasonable starting choice, but **not a principled answer** for Gemma 4 specifically — PEFT literature has no Gemma-targeted alpha recommendation we could find.
- **bf16 NaN traps**: there are multiple HF discuss threads on bf16 LoRA NaN issues (e.g. Llama-3.1 + DoRA), but no Gemma-4-specific guidance beyond the SDPA-vs-flash-attn fix you already applied. The fp32 upcast in `LoRA.forward` is the standard mitigation; whether it's *sufficient* for Gemma 4 requires running the actual config and watching for NaNs in the Perceiver attention scores.
- **Refuted**: "hypernetworks have known vanishing/exploding gradient issues addressable via spectral norm regularization" was killed 0-3 — not enough evidence in the cited source.

### 6. Alternative approaches you haven't considered

- **Zhyper (arXiv:2510.19733, Oct 2025)** — strongest "newer than T2L" candidate. Fixed per-(layer, module) A and B matrices, generate only `z_{l,t} ∈ R^r` via 3-layer MLP conditioned on `concat[c, e_t, e_l]`. ΔW = A·diag(z)·B. **26× parameter reduction** (4.2M vs 55M at rank 8) with 0.5% performance gap on average tasks. **Caveat: HumanEval drops 39.6 vs T2L 42.3** — slightly worse on code, directly relevant to code2lora's Qwen2.5-Coder use case.
- **HoRA (Cross-Head LoRA, arXiv:2510.04295)** — shared hypernetwork generates joint low-rank matrices for all attention heads *within* a layer (not across). Tangential to your design but reinforces the broader "structured sharing > fully-independent slots" pattern.
- **Multi-profile heads / weight-sharing between profiles**: literature didn't surface specific precedent for "separate head per dimension profile" within a single LLM. This is novel territory — empirical comparison against single-profile-C is the only way forward.
- **Cross-attention between GRU hidden state and learned layer queries**: this is just the Perceiver pattern with a different conditioning input. T2L's "use the input as task embedding fed to shared head with layer/module embeddings" is simpler and proven.

---

## Risks / issues that surfaced

1. **Code2LoRA's "one (A,B) per module type, broadcast across all layers" design is novel** — T2L always conditions on E[l]. We don't have empirical evidence this works on Qwen2.5-Coder vs a depth-conditioned shared head. The Code2LoRA paper is under review and unavailable. **Plan A inherits this assumption when porting to Gemma 4.**
2. **The original prompt's Hyperdecoders comparison numbers are wrong** (71.6 vs 73.6 in prompt; correct: 67.7 vs 67.3 in T2L Table 2). Worth correcting in your internal docs.
3. **T2L only targets q_proj and v_proj at rank 8** in its original config. Plan A's choice of q/o/gate/up/down (5 modules) is a 2.5× expansion in target modules vs the published T2L config. May be fine; not validated by any prior work.
4. **Profile C exclusion of global-attention layers** is a real architectural concern for context-conditioning — orthogonal to the Profile C "16 layers" decision in Plan A.

---

## Concrete recommendations

### For `GEMMA4.md` (Plan A — code2lora → Gemma 4 port)

**HIGH confidence:**
- **Read `layer_types` from the loaded config at runtime**, don't hardcode Profile C as "indices that look local." Confirm Profile C is local-only on actual checkpoint, log layer types per spec group.
- **Document explicitly that Profile C excludes all global-attention layers** (19, 24, 29, 34) and that this is a known design tradeoff to revisit.

**MEDIUM confidence:**
- **Add a Profile-B variant as a comparison run** before committing to Profile C only. The cost is one extra spec group; the upside is knowing whether you're leaving global-attention conditioning on the table.
- **Reduce module set to q_proj + v_proj initially** to match T2L's tested config. Expand to o/gate/up/down only after baseline works.

**LOW confidence:**
- The `lm_head` stashing and `.generate` delegation patterns are unvalidated by literature — keep them but verify on GPU early.

### For `GRU_INTEGRATION.md` (Plan B — GRU sequential variant in doc2lora)

**HIGH confidence:**
- **Don't bypass the Perceiver with a single 2048 → 327,680 linear.** Instead, feed the GRU's hidden state into the HyperLoRA head as the task embedding `z`, then let the existing shared MLP + depth/module embeddings produce per-slot outputs (T2L pattern). The existing `text_to_lora_impl.py` in your repo already implements this — wire it up.
- **Alternative if you want to keep "GRU sequential aggregation" as the main innovation**: replace the Perceiver+EinMix head wholesale with T2L's shared head pattern, and feed it the GRU hidden state. This is a smaller, cleaner architecture than the current proposal.

**MEDIUM confidence:**
- **Probe per-(layer, module) LoRA Frobenius norms before committing to the GRU integration.** Use the existing `ctx-probe` (commit 5ffc1fb). If norms are nearly identical across layers, the existing per-layer head is parameterizing redundancy — strong signal that the simpler shared head will work fine.
- **Consider Zhyper's ΔW = A·diag(z)·B factorization** as a third option. 26× param reduction is real; only the code-benchmark caveat (39.6 vs 42.3 HumanEval on Mistral-7B) gives pause for code2lora.

---

## Prioritized empirical verification list (GPU)

1. **[Highest priority]** Run `ctx-probe` on current doc2lora training run and dump per-(layer, module) LoRA Frobenius norms + pairwise diffs. If layer-to-layer variation is small, the case for replacing the Perceiver+EinMix head with T2L's shared head becomes overwhelming.
2. **[High]** On Gemma 4 Profile C training, log Perceiver attention scores in bf16 vs the SDPA fix and confirm no NaN regression with current config. Verify `LoRA.forward` fp32 upcast is hit in the actual code path.
3. **[High]** GPU smoke test the `model_utils.py` pattern for Gemma 4: extract `full_model.model.language_model`, bind `.generate`, verify `lm_head` projection at call site. This is unvalidated by literature.
4. **[Medium]** Run Profile-B comparison (includes global-attention layers) against Profile-C on the same data. Even one short run tells you whether global-attention conditioning matters for context-conditioning.
5. **[Medium]** If/when you implement the simpler T2L-shared-head GRU variant, A/B it against the current Perceiver-based doc2lora on closed_qa to confirm no regression.
6. **[Lower]** Sweep `lora_alpha` ∈ {4, 8, 16} on Gemma 4 — no published recommendation for this model specifically.

---

## Key sources

- T2L paper (Charakorn et al., ICML 2025): arxiv.org/abs/2506.06105 + ICML slides
- Zhyper (Abdalla et al., Oct 2025): arxiv.org/abs/2510.19733
- Hyperdecoders (Ivison & Peters, EMNLP 2022): arxiv.org/pdf/2203.08304
- HoRA (Oct 2025): arxiv.org/abs/2510.04295
- NormAL LoRA (EMNLP 2025 Findings): aclanthology.org/2025.findings-emnlp.1074
- Gemma 3n docs: huggingface.co/docs/transformers/en/model_doc/gemma3n
- E4B config.json (concrete layer_types): huggingface.co/lmstudio-community/gemma-3n-E4B-it-MLX-bf16

Full output (with refuted claims, source quality flags, and stats) is at `doc/RESEARCH_FULL_OUTPUT.md`.
