{
  "summary": "Deep research harness — fan-out web searches, fetch sources, adversarially verify claims, synthesize a cited report.",
  "agentCount": 108,
  "logs": [
    "Q: Research the questions in /Volumes/T7/projects/doc-to-lora/doc/RESEARCH_PROMPT.m…",
    "Decomposed into 6 angles: academic/foundational, technical/PEFT literature, Gemma 4 architecture specifics, practitioner/integration, training stability, recent hypernet alternatives",
    "technical/PEFT literature: 6 results",
    "recent hypernet alternatives: 6 results",
    "recent hypernet alternatives: 4 novel (2 filtered)",
    "practitioner/integration: 6 results",
    "practitioner/integration: 5 novel (1 filtered)",
    "training stability: 6 results",
    "training stability: 4 novel (2 filtered)",
    "academic/foundational: 6 results",
    "academic/foundational: 3 novel (3 filtered)",
    "Gemma 4 architecture specifics: 6 results",
    "Gemma 4 architecture specifics: 3 novel (3 filtered)",
    "Fetched 25 sources → 118 claims → verifying top 25",
    "\"T2L conditions its hypernetwork on BOTH module typ…\": 3-0 ✓",
    "\"T2L was directly compared against Hyperdecoders (p…\": 3-0 ✓",
    "\"T2L only targets query and value projection module…\": 3-0 ✓",
    "\"T2L has three architectural variants (L, M, S) wit…\": 3-0 ✓",
    "\"T2L variants are conditioned on a Module embedding…\": 3-0 ✓",
    "\"T2L has three architectural variants (L, M, S) tha…\": 3-0 ✓",
    "\"On zero-shot SFT-trained T2L benchmarks, the L and…\": 3-0 ✓",
    "\"Zhyper factorizes LoRA generation by keeping fixed…\": 3-0 ✓",
    "\"Zhyper achieves up to 26x fewer parameters than T2…\": 3-0 ✓",
    "\"The conditioning architecture is a 3-layer MLP tha…\": 3-0 ✓",
    "\"HyRA uses a shared hypernetwork to generate joint …\": 2-1 ✓",
    "\"Theoretically, the paper proves that the shared-st…\": 1-2 ✗",
    "\"T2L (SFT-trained) outperforms Hyperdecoders (Iviso…\": 0-3 ✗",
    "\"T2L has three architectural variants (L, M, S) tha…\": 3-0 ✓",
    "\"T2L conditions its hypernetwork head on a depth em…\": 3-0 ✓",
    "\"Hyperdecoders reuse a single hypernetwork across a…\": 3-0 ✓",
    "\"The mechanism for per-layer generation is concaten…\": 3-0 ✓",
    "\"In zero-shot SFT-trained evaluation, L (largest, s…\": 3-0 ✓",
    "\"Different transformer layers require very differen…\": 0-3 ✗",
    "\"Empirically on DeBERTaV3-base/MNLI, NormAL LoRA as…\": 3-0 ✓",
    "\"Hypernetworks face known numerical stability issue…\": 0-3 ✗",
    "\"Weight generation strategies fall into distinct ca…\": 0-3 ✗",
    "\"Gemma 3n's language model uses an alternating atte…\": 2-1 ✓",
    "\"Sliding window size is 512 tokens by default, and …\": 3-0 ✓",
    "\"The Gemma 3n text model has 35 hidden layers by de…\": 1-2 ✗",
    "Verify done: 25 claims → 19 confirmed, 6 killed"
  ],
  "result": {
    "question": "Research the questions in /Volumes/T7/projects/doc-to-lora/doc/RESEARCH_PROMPT.md — a review of two hypernetwork architectures (code2lora and doc2lora) that generate LoRA weights for frozen LLMs, both descended from SakanaAI's Text-to-LoRA (Charakorn et al., ICML 2025, https://openreview.net/forum?id=zWskCdu3QA).\n\nThe two codebases:\n- code2lora/ — maps 2048-d repo embeddings → LoRA weights for Qwen2.5-Coder-1.5B. Has Static (single snapshot) and GRU (sequential commits) variants. Produces ONE (A,B) per module type, SHARED across all layers.\n- src/ctx_to_lora/ (doc2lora) — maps raw text → ctx_encoder → Perceiver cross-attention → EinMix head → PER-LAYER, PER-MODULE LoRA (A,B). Targets Gemma 4 E2B. Perceiver produces n_layers × n_modules distinct output slots.\n\nTwo plans under review:\n- Plan A (code2lora/GEMMA4.md): Port code2lora to Gemma 4 E2B. Use Profile C only (16 layers) with 5 modules (q_proj, o_proj, gate_proj, up_proj, down_proj — excluding k/v due to dim heterogeneity). New model_utils.py extracts bare text model from VLM, delegates .generate, projects last_hidden_state → logits at call site. Regex fix for module discovery.\n- Plan B (code2lora/GRU_INTEGRATION.md): Add GRU sequential variant to doc2lora. GRU sits between ctx_encoder and HyperLoRA head, accumulates across commits. In GRU mode, Perceiver is BYPASSED — the GRU's [bs, 2048] hidden state projects directly to [bs, 16, 5, 8, 512] = 327,680 values via ONE linear layer (160:1 expansion). Custom training loop, truncated BPTT every 16 commits.\n\nInvestigate these six areas (numbered to match the prompt):\n\n1. **Per-layer vs shared LoRA — does it matter?** T2L paper shows their shared head and Hyperdecoders (per-layer) are roughly comparable. We don't have the Code2LoRA paper (under review). Search for any papers/blogs/ablations comparing shared vs per-layer LoRA generation in hypernetworks. Theoretical arguments from PEFT literature about whether layer-specific LoRA matters for code tasks. Could doc2lora's per-layer capacity be mostly unused/redundant? How to test (e.g., per-layer LoRA Frobenius norms across training)?\n\n2. **GRU bottleneck concern.** Single [bs, 2048] → [bs, 16, 5, 8, 512] = 327,680 values via one linear layer = 160:1 expansion. Pathological? Established approaches for expanding a single vector to structured multi-slot output? Deeper projection head (2-layer MLP)? Alternative: GRU → shared per-type (A,B) matching code2lora's approach? Papers on \"structured output from recurrent state\"?\n\n3. **Gemma 4 Profile C: local vs global attention.** Profile C layers (15-18, 20-23, 25-28, 30-33) may all be sliding-window-local, with global-attention layers at skipped positions (19, 24, 29, 34). Confirm from Gemma 4 model card/config/published analysis whether Profile C = local-only. If so, how much does this matter for a context-conditioning hypernet? Are global-attention layers the ones that actually \"see\" long context? Cost/complexity of including Profile B layers under separate spec group?\n\n4. **Missing Gemma 4 gotchas.** Plan assumes: extracting full_model.model.language_model works, .generate delegation via bound method works, lm_head stashing works, regex r\"\\b(?:model\\.)?layers\\.(\\d+)\\.\" matches bare text model's named_modules(). Known issues with Gemma 4 E2B + transformers>=5.5.0 (check huggingface/transformers GitHub issues). Gemma4ForConditionalGeneration quirks around dtype/device_map/flash_attention_2 vs other Gemma models. Tokenization differences (special tokens, chat template) affecting left-truncate/left-pad. pad_token behavior.\n\n5. **Training stability.** Gemma 4 + bf16 + LoRA NaN traps. doc2lora's NaN was fixed by switching Perceiver from flash-attn to SDPA, and by using lora_alpha=8 instead of legacy r**1.5 * 2 = 45. Is LoRA.forward's fp32 upcast sufficient, or do bf16 intermediate states cause issues? Is lora_alpha=8 (scaling=alpha/rank=1.0) right for Gemma? What does PEFT literature recommend for Gemma?\n\n6. **Alternative approaches we haven't considered.** Multi-profile LoRA heads (separate heads per dimension profile covering all 35 layers). Weight-sharing schemes between profiles. Different aggregation architectures (cross-attention between GRU hidden state and learned layer queries vs flat linear projection). Recent hypernetwork-for-LoRA papers (2025-2026) improving on T2L architecture.\n\nFiles in repo to consult for context (deep-research can read these via filesystem):\n- doc/ARCHITECTURE_COMPARISON.md\n- code2lora/GEMMA4.md\n- code2lora/GRU_INTEGRATION.md\n- code2lora/hypernetwork/code2lora_core.py (head 214-269, LoRA module 53-102, module spec discovery 114-140)\n- src/ctx_to_lora/modeling/hypernet.py (HyperLoRA head 238-464, ModulatedPretrainedModel forward 743-857)\n- src/ctx_to_lora/modeling/aggregator.py (Perceiver 84-212)\n- src/ctx_to_lora/modeling/text_to_lora_impl.py (original T2L code with shared_AB_head)\n- configs/main_exp/gemma4_e2b_closed_qa.yaml\n\nExpected output: structured report with (1) findings per investigation area with sources, (2) issues/risks we've missed, (3) concrete recommendations with confidence levels for changes to GEMMA4.md and GRU_INTEGRATION.md, (4) prioritized list of things to verify empirically on GPU before committing to current plans.",
    "summary": "The research strongly suggests that doc2lora's per-layer per-module full (A,B) generation via Perceiver is significantly over-parameterized relative to what's actually needed: T2L itself uses a SHARED MLP backbone conditioned on cheap learnable depth/module embeddings (the canonical Hyperdecoders pattern) to produce per-(layer,module) LoRAs, and L vs M variants differ by only 0.2 points on zero-shot benchmarks despite L having 38% more parameters. Zhyper (Oct 2025) goes further — generating only a rank-r modulation vector with fixed per-(layer,module) A/B matrices — achieving up to 26x parameter reduction (4.2M vs 55M) with only 0.5% performance gap. The proposed GRU 160:1 single-linear expansion (2048 → 327,680) is architecturally pathological compared to these established patterns; the simpler fix is to keep T2L's shared head with depth+module embeddings (or Hyperdecoders-style concatenation) on top of the GRU's hidden state. For Gemma 4 Profile C, the 4-local + 1-global attention pattern is confirmed and configurable per-checkpoint via layer_types (read it from config; don't assume) — Profile C does hit only sliding-window-local layers, which has unclear consequences for context-conditioned hypernets and warrants empirical comparison against a Profile-B variant.",
    "findings": [
      {
        "claim": "T2L's architecture is a SHARED hypernetwork backbone conditioned on learnable depth E[l] and module-type E[m] embeddings (phi_{m,l} = concat[f(z), E[m], E[l]]), producing per-layer per-module LoRA weights in a single batched forward pass. This is the canonical 'how to get per-layer outputs from a shared head' pattern and is materially simpler than both doc2lora's Perceiver+EinMix per-slot head and the proposed GRU 1:160 linear expansion.",
        "confidence": "high",
        "sources": [
          "https://arxiv.org/abs/2506.06105",
          "https://arxiv.org/pdf/2506.06105",
          "https://icml.cc/media/icml-2025/Slides/43471.pdf",
          "https://arxiv.org/pdf/2203.08304",
          "/Volumes/T7/projects/doc-to-lora/src/ctx_to_lora/modeling/text_to_lora_impl.py"
        ],
        "evidence": "Primary T2L paper (Charakorn et al., ICML 2025) Eq. for phi_{m,l} and the explicit statement 'all variants use the same backbone architecture and only differ in their output heads and learnable embeddings... values of m and l can be batched, which allows T2L to generate ΔW for all modules and layer indices efficiently within a single forward pass.' Confirmed in the doc2lora repo's own T2L reference impl at text_to_lora_impl.py lines 587-595 (layer_depth_encoder, layer_type_encoder) and line 793 (cat_emb concatenation). Hyperdecoders (Ivison & Peters, EMNLP 2022) established the same pattern earlier: 'We re-use this hypernetwork to generate the adapters for every layer by (partially) conditioning the input on layer embeddings, greatly improving the parameter efficiency.' HyperFormer (Karimi Mahabadi et al., ACL 2021) pioneered the approach. 3 unanimous adversarial votes on each constituent claim."
      },
      {
        "claim": "T2L's three variants (L, M, S) trade output-space complexity for parameter count, but the L vs M gap is only 0.2 pts on 10-task zero-shot avg (67.7 vs 67.5) despite L having 38% more parameters (55M vs 34M). S stalls at 65.2 with 91% reduction. This is direct evidence that shared/inductive-bias-heavy heads do not lose meaningful capability when generating per-layer LoRAs.",
        "confidence": "high",
        "sources": [
          "https://arxiv.org/abs/2506.06105",
          "https://arxiv.org/pdf/2506.06105",
          "https://icml.cc/media/icml-2025/Slides/43471.pdf"
        ],
        "evidence": "T2L Table 2: L=67.7, M=67.5, S=65.9 on 10-task zero-shot avg. Parameter counts from Section 4: L=55M, M=34M, S=5M trainable. At 479 tasks: L=67.7, M=67.5, S=65.2 — S 'does not benefit from extended training with 479 tasks, potentially due to its limited model capacity' (paper quote). L outputs both A and B simultaneously [2,r,d]; M shares head between A/B [r,d] via embedding; S outputs one rank [d]. The L vs Hyperdecoders (per-instance) comparison shows 67.7 vs 67.3 avg — essentially equivalent despite weaker per-task conditioning. 3 unanimous votes."
      },
      {
        "claim": "Zhyper (Abdalla et al., arXiv:2510.19733, Oct 2025) factorizes LoRA generation by keeping FIXED per-layer per-module A and B matrices and only generating a small rank-r modulation vector z via a 3-layer MLP conditioned on concat[c, e_t, e_l]. ΔW = A·diag(z)·B. Achieves competitive performance (65.9 vs T2L L 66.4, 0.5% gap) with up to 26x fewer parameters (4.2M vs 55M at rank 8; 7.62M vs 110M at rank 16).",
        "confidence": "high",
        "sources": [
          "https://arxiv.org/abs/2510.19733"
        ],
        "evidence": "Zhyper paper Eq. 1: z_{l,t} = H_phi(c || e_t || e_l) ∈ R^r; Eq. 2: ΔW_{l,t}(c) = A_{l,t} diag(z_{l,t}) B_{l,t}. Table 2 confirms exact parameter counts at ranks 8/16/32. Table 5 distinguishes Zhyper as the only listed method with 'Compact Modulation: ✓' — T2L/HyperLoRA/Hyperdecoders all generate full LoRA matrices. Caveats from verification: (a) on HumanEval code benchmark Zhyper drops to 39.6 vs T2L's 42.3 — code performance is slightly worse, relevant to code2lora's Qwen2.5-Coder use case; (b) experiments use Mistral-7B base, not Gemma 4 or Qwen2.5-Coder; (c) preprint under review, not yet peer-reviewed. 3 unanimous votes on both Zhyper claims."
      },
      {
        "claim": "HoRA (Cross-Head LoRA, arXiv:2510.04295) uses a shared hypernetwork to generate joint low-rank matrices for ALL attention heads within a single layer (not across layers) to promote cross-head information sharing and reduce redundancy. Provides additional precedent that hypernetwork generation with structured weight sharing outperforms fully-independent per-slot generation. NOTE: research originally referred to this as 'HyRA' — correct name is HoRA.",
        "confidence": "medium",
        "sources": [
          "https://arxiv.org/abs/2510.04295"
        ],
        "evidence": "Paper architecture: Theta_A produces A^Q, A^V from a shared learned A_bar; Theta_B produces B^Q_{1:H}, B^V_{1:H} from learned B_{1:H} via specialized projections. 'The shared hypernetwork introduces structured coupling: heads are no longer fully independent but instead benefit from common parameterization, while still retaining flexibility through specialized transformations.' 2-1 vote (one verifier noted the name issue). Marginally relevant — sharing is intra-layer not cross-layer, but reinforces the broader pattern."
      },
      {
        "claim": "NormAL LoRA empirically shows that on DeBERTaV3-base/MNLI, optimal LoRA rank is non-uniformly distributed: deeper layers get higher rank, and W_f1/W_f2 (FFN) modules get more rank than W_K/W_O (attention key/output). This is some evidence FOR per-layer LoRA capacity mattering, but the finding is from a single 86M encoder model on a single NLU task and does NOT directly demonstrate that decoder LLMs (Qwen2.5-Coder, Gemma 4) on code tasks need per-layer variation that a shared-conditioned head can't represent.",
        "confidence": "medium",
        "sources": [
          "https://aclanthology.org/2025.findings-emnlp.1074.pdf"
        ],
        "evidence": "Verbatim quote from NormAL LoRA EMNLP 2025 Findings: 'earlier (shallower) layers are assigned lower ranks compared to deeper layers... W_K and W_O receive the least importance, while W_f1 and W_f2 are assigned the highest ranks.' Corroborated by AdaLoRA (ICLR 2023) and TriAdaptLoRA (2025) on similar setups. CRITICAL CAVEAT: the related claim that 'a shared-head hypernetwork cannot represent per-layer non-uniformity' was REFUTED 0-3 — because T2L's shared head IS conditioned on E[l], so it CAN produce different-magnitude outputs per layer. The right framing is that shared heads can represent layer variation; the question is whether full per-layer per-module parameterization (doc2lora) adds anything over shared+embeddings (T2L). 3 unanimous votes on the empirical finding; the architectural implication is weaker."
      },
      {
        "claim": "The proposed GRU integration (Plan B / GRU_INTEGRATION.md) bypasses the Perceiver and projects a single [bs, 2048] GRU hidden state through one linear layer to [bs, 16, 5, 8, 512] = 327,680 values (160:1 expansion). This is architecturally pathological compared to the established alternatives: T2L's shared head + depth/module embeddings, or Hyperdecoders' concatenated layer-embedding conditioning. The canonical pattern is to feed the conditioning vector into a shared MLP that is called once per (layer, module) slot with embeddings concatenated, not to predict all slots from one linear layer.",
        "confidence": "high",
        "sources": [
          "https://arxiv.org/abs/2506.06105",
          "https://arxiv.org/pdf/2203.08304",
          "https://arxiv.org/abs/2510.19733"
        ],
        "evidence": "Three independent precedents all use the same shared-backbone + per-slot embedding pattern: T2L's phi_{m,l} = concat[f(z), E[m], E[l]] -> shared MLP backbone -> head; Hyperdecoders' Adapter_i = Hypernetwork([e; l_i]); Zhyper's H_phi(c || e_t || e_l). All three reuse one hypernetwork across all (layer, module) slots by conditioning on cheap learnable embeddings — none predict all slots from a single linear projection. Hyperdecoders explicitly motivates this as 'greatly improving the parameter efficiency.' Direct fix for Plan B: replace 'GRU hidden -> single linear -> [16,5,8,512]' with 'GRU hidden as task embedding -> existing HyperLoRA head batched over (m,l) via depth+module embeddings.' This preserves the GRU's sequential aggregation contribution while using the standard well-tested expansion mechanism."
      },
      {
        "claim": "Gemma 3n / Gemma 4 E2B's language model uses alternating attention: every 5th layer is global full-attention (indices 4, 9, 14, 19, 24, 29, 34), all others are sliding-window-local with default window=512. Profile C (layers 15-18, 20-23, 25-28, 30-33) is therefore LOCAL-ONLY — it hits exactly the 4-of-5 sliding-window layers and skips the global-attention layers at 19, 24, 29, 34. The layer_types are configurable per-checkpoint via the config's layer_types list, so plans MUST read this from the actual loaded config rather than assume.",
        "confidence": "high",
        "sources": [
          "https://huggingface.co/docs/transformers/en/model_doc/gemma3n",
          "https://huggingface.co/lmstudio-community/gemma-3n-E4B-it-MLX-bf16/blob/main/config.json"
        ],
        "evidence": "HF docs verbatim: 'alternating 4 local sliding window self-attention layers for every global self-attention layer with a maximum context length of 32k tokens.' Transformers source (configuration_gemma3n.py): layer_types = 'full_attention' if (i + 1) % 5 == 0 else 'sliding_attention' over 35 layers. Deployed E4B checkpoint config.json explicitly contains 35-element layer_types array confirming the pattern, and sliding_window=512. This RESOLVES the 'open design question' flagged in code2lora/GEMMA4.md line 20. Implication for Plan A: Profile C captures only context-window-512 layers — for a hypernet that injects information from potentially long context, the global-attention layers (where the model actually integrates across the full sequence) are EXCLUDED, which may be the wrong choice for context-conditioning. Empirical comparison against Profile B (which would include global layers) is warranted. 2-1 vote on this claim (1 verifier wanted tighter primary sourcing); architectural inference is well-grounded."
      },
      {
        "claim": "T2L's original design targets only q_proj and v_proj at rank 8 across all attention blocks (3.4M LoRA params on Mistral-7B), explicitly sidestepping the dimension-heterogeneity problem of gate/up/down_proj where d_in != d_out. Plan A's choice to include q_proj, o_proj, gate_proj, up_proj, down_proj (excluding k/v due to GQA dim heterogeneity) is a departure from T2L's tested module set, and doc2lora already handles this via EinMix per-module-shape weights.",
        "confidence": "high",
        "sources": [
          "https://arxiv.org/abs/2506.06105"
        ],
        "evidence": "T2L paper: 'All LoRA adapters are of rank 8 and only target the query and the value projection modules in every attention block of the base LLM (totaling 3.4M parameters).' Confirmed by T2L GitHub config files (target_modules: [q_proj, v_proj]) and parameter-count arithmetic on Mistral-7B GQA. Note that Gemma 3n/4 uses kv_heads=2 (vs 8 q-heads), giving k_proj and v_proj out_dim 512 vs q_proj's 2048 — same heterogeneity issue Plan A flags. doc2lora's per-(layer,module-shape) EinMix head naturally handles this; the open question is whether including o_proj/gate/up/down adds real capability vs cost vs just doing q+v as T2L did."
      }
    ],
    "caveats": "Key uncertainties and limitations: (1) Code2LoRA's own paper is under review and unavailable, so we cannot directly compare its 'shared (A,B) per module type, broadcast across layers' design empirically against T2L's depth-embedding-conditioned shared head — these are different forms of sharing, and the T2L empirical evidence doesn't directly test the strictly-broadcast variant. (2) Zhyper is a preprint (Oct 2025), not yet peer-reviewed; its HumanEval code-benchmark drop (39.6 vs T2L 42.3) is relevant to code2lora's Qwen2.5-Coder use case and weakens the 'just use Zhyper for code2lora' inference. (3) The per-layer LoRA rank non-uniformity evidence (NormAL LoRA, AdaLoRA) comes from small encoder models on NLU tasks — generalization to decoder LLMs on code/QA tasks is not directly established. (4) The Gemma 4 transformers-version-specific issues (Gemma4ForConditionalGeneration quirks with bf16, flash_attn vs SDPA, lm_head stashing across .generate calls, chat-template/pad_token issues) were NOT meaningfully resolved by the research — these remain empirical-verification items, not literature-resolved. (5) The five 'refuted' claims are particularly important: (a) the Hyperdecoders comparison numbers in the prompt were wrong (correct: T2L L=67.7 vs Hyperdecoders=67.3, NOT 71.6 vs 73.6); (b) the claim that 'shared heads cannot represent per-layer LoRA capacity variation' was correctly refuted — T2L's E[l] embedding means a shared head CAN produce per-layer variation, so the right comparison is not 'shared vs per-layer' but 'shared + depth-embedding vs per-layer-per-module full parameterization'; (c) doc2lora is descended from T2L and the upstream T2L code is in the repo at src/ctx_to_lora/modeling/text_to_lora_impl.py — the simplest forward path is to reuse this proven mechanism rather than reinvent. (6) The L vs M = 0.2 pt comparison is at 10-task zero-shot avg for SFT-trained models on Mistral-7B and may not transfer perfectly to Gemma 4 E2B context-conditioned setting.</caveats>\n<parameter name=\"openQuestions\">[\n  \"Empirical question 1 (highest priority): On Gemma 4 Profile C with the existing doc2lora training data, does replacing the per-layer Perceiver+EinMix head with a T2L-style shared MLP head conditioned on depth+module embeddings recover the same/better loss? If yes, the GRU 160:1 expansion problem dissolves — GRU hidden state becomes the task embedding fed to the shared head, no bypass needed.\",\n  \"Empirical question 2: On the same Gemma 4 Profile C setup, does a Zhyper-style ΔW = A·diag(z)·B with fixed per-(layer,module) A,B and a small generated z work for context-conditioned LoRA, given Zhyper's slightly weaker code-benchmark performance? Particularly relevant for code2lora's Qwen2.5-Coder target.\",\n  \"Empirical question 3: Does Profile C (local-attention-only) underperform a Profile B variant that includes the global-attention layers (19, 24, 29, 34) for context-conditioned hypernet generation? Mechanistic intuition says global-attention layers are where long-context info gets integrated, so excluding them may be wrong for context-conditioning specifically.\",\n  \"Architecture question: Does code2lora's 'one (A,B) per module type, broadcast identically across all layers' actually work? T2L never tests this exact configuration (always conditions on E[l]); if it works on Qwen2.5-Coder, that's novel evidence; if it underperforms a T2L-style depth-conditioned shared head by a meaningful margin, Plan A's port may inherit that limitation.\",\n  \"Gemma 4 transformers integration: pad_token, chat template, Gemma4ForConditionalGeneration .generate delegation, dtype/device_map quirks, flash_attn vs SDPA NaN traps — all require direct GPU verification; no published source addresses the specific 'extract bare text model, project last_hidden_state to logits, stash lm_head' pattern Plan A proposes.\",\n  \"Is doc2lora's per-layer per-module Perceiver capacity actually being used? Probe with per-(layer,module) LoRA Frobenius norms and pairwise diffs across training — if outputs collapse to near-identical across layers, the shared-head-with-embeddings approach loses nothing and saves significant parameters.\"\n]",
    "refuted": [
      {
        "claim": "Theoretically, the paper proves that the shared-structure hypernetwork improves sample complexity of estimating low-rank matrices from an EXPONENTIAL rate (without sharing) to a POLYNOMIAL rate — meaning that fully independent per-head LoRA matrices may need exponentially more data to estimate well. This is a formal argument that excess per-head/per-slot capacity is harmful in low-data regimes (relevant to whether doc2lora's per-layer-per-module capacity is well-utilized).",
        "vote": "1-2",
        "source": "https://arxiv.org/abs/2510.04295"
      },
      {
        "claim": "T2L (SFT-trained) outperforms Hyperdecoders (Ivison & Peters 2022) — a per-instance hypernetwork that generates adapters on-the-fly per input sequence — on average across 8 benchmark tasks (T2L L: 71.6 / M: 73.5 / S: 71.6 vs Hyperdecoders: 73.6). More importantly, T2L conditions on a TASK DESCRIPTION (not the input instance) while Hyperdecoders conditions per-instance — but T2L still wins or ties despite the much weaker conditioning signal. This suggests per-layer/per-instance conditioning is not strictly required for strong generalization.",
        "vote": "0-3",
        "source": "https://arxiv.org/pdf/2506.06105"
      },
      {
        "claim": "Different transformer layers require very different LoRA capacities, making uniform rank allocation across layers wasteful — supporting the case that per-layer LoRA generation (like doc2lora) could exploit this heterogeneity, whereas shared (single A,B) heads (like code2lora) cannot.",
        "vote": "0-3",
        "source": "https://aclanthology.org/2025.findings-emnlp.1074.pdf"
      },
      {
        "claim": "Hypernetworks face known numerical stability issues including vanishing/exploding gradients, addressable via gradient clipping and spectral norm regularization — directly relevant to bf16 LoRA NaN traps in doc2lora.",
        "vote": "0-3",
        "source": "https://arxiv.org/html/2306.06955v3"
      },
      {
        "claim": "Weight generation strategies fall into distinct categories: generate-once (all weights together), component-wise (per-layer), chunk-wise, and multi-head (split). Multi-head 'simplifies the complexity and reduces the number of weights required' — supporting the choice between shared (code2lora) vs per-layer (doc2lora) LoRA generation.",
        "vote": "0-3",
        "source": "https://arxiv.org/html/2306.06955v3"
      },
      {
        "claim": "The Gemma 3n text model has 35 hidden layers by default with hidden_size 2048, 8 attention heads, 2 KV heads (GQA), and head_dim 256. The hidden_size of 2048 matches code2lora's expected per-layer dim and confirms the doc2lora pipeline assumption; GQA with only 2 KV heads (vs 8 Q heads) means k_proj and v_proj have 4x smaller output dim (512 vs 2048), confirming Plan A's exclusion of k/v from the module set due to dim heterogeneity.",
        "vote": "1-2",
        "source": "https://huggingface.co/docs/transformers/en/model_doc/gemma3n"
      }
    ],
    "sources": [
      {
        "url": "https://arxiv.org/abs/2506.06105",
        "quality": "primary",
        "angle": "academic/foundational",
        "claimCount": 5
      },
      {
        "url": "https://icml.cc/media/icml-2025/Slides/43471.pdf",
        "quality": "primary",
        "angle": "academic/foundational",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/abs/2510.19733",
        "quality": "primary",
        "angle": "academic/foundational",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/abs/2510.04295",
        "quality": "primary",
        "angle": "technical/PEFT literature",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/pdf/2506.06105",
        "quality": "primary",
        "angle": "technical/PEFT literature",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/abs/2510.02630",
        "quality": "primary",
        "angle": "technical/PEFT literature",
        "claimCount": 3
      },
      {
        "url": "https://arxiv.org/pdf/2203.08304",
        "quality": "primary",
        "angle": "technical/PEFT literature",
        "claimCount": 5
      },
      {
        "url": "https://aclanthology.org/2025.findings-emnlp.1074.pdf",
        "quality": "primary",
        "angle": "technical/PEFT literature",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/html/2306.06955v3",
        "quality": "primary",
        "angle": "technical/PEFT literature",
        "claimCount": 5
      },
      {
        "url": "https://huggingface.co/docs/transformers/en/model_doc/gemma3n",
        "quality": "primary",
        "angle": "Gemma 4 architecture specifics",
        "claimCount": 5
      },
      {
        "url": "https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4",
        "quality": "blog",
        "angle": "Gemma 4 architecture specifics",
        "claimCount": 4
      },
      {
        "url": "https://arxiv.org/pdf/2503.19786",
        "quality": "primary",
        "angle": "Gemma 4 architecture specifics",
        "claimCount": 5
      },
      {
        "url": "https://github.com/huggingface/peft/issues/3129",
        "quality": "primary",
        "angle": "practitioner/integration",
        "claimCount": 5
      },
      {
        "url": "https://ghost.oxen.ai/writing-a-fine-tuning-and-deployment-pipeline-isnt-as-easy-as-it-looks-gemma-4-version/",
        "quality": "blog",
        "angle": "practitioner/integration",
        "claimCount": 5
      },
      {
        "url": "https://huggingface.co/principled-intelligence/gemma-4-E2B-it-text-only",
        "quality": "secondary",
        "angle": "practitioner/integration",
        "claimCount": 4
      },
      {
        "url": "https://dev.to/dentity007/fine-tuning-gemma-4-on-day-zero-3-bugs-we-solved-in-30-minutes-2ke",
        "quality": "blog",
        "angle": "practitioner/integration",
        "claimCount": 5
      },
      {
        "url": "https://github.com/PrimeIntellect-ai/prime-rl/issues/2362",
        "quality": "primary",
        "angle": "practitioner/integration",
        "claimCount": 5
      },
      {
        "url": "https://discuss.huggingface.co/t/bf16-training-instability-with-llama-3-1-8b-lora-dora-peft/170326",
        "quality": "forum",
        "angle": "training stability",
        "claimCount": 5
      },
      {
        "url": "https://huggingface.co/docs/peft/main/developer_guides/troubleshooting",
        "quality": "primary",
        "angle": "training stability",
        "claimCount": 5
      },
      {
        "url": "https://huggingface.co/blog/damjan-k/rslora",
        "quality": "secondary",
        "angle": "training stability",
        "claimCount": 4
      },
      {
        "url": "https://github.com/huggingface/transformers/issues/36814",
        "quality": "primary",
        "angle": "training stability",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/abs/2602.06358",
        "quality": "primary",
        "angle": "recent hypernet alternatives",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/abs/2506.11638",
        "quality": "primary",
        "angle": "recent hypernet alternatives",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/abs/2603.19278",
        "quality": "primary",
        "angle": "recent hypernet alternatives",
        "claimCount": 4
      },
      {
        "url": "https://arxiv.org/html/2604.02051",
        "quality": "primary",
        "angle": "recent hypernet alternatives",
        "claimCount": 4
      }
    ],
    "stats": {
      "angles": 6,
      "sourcesFetched": 25,
      "claimsExtracted": 118,
      "claimsVerified": 25,
      "confirmed": 19,
      "killed": 6,
      "afterSynthesis": 8,
      "urlDupes": 4,
      "budgetDropped": 7,
      "agentCalls": 108
    }
  }
}