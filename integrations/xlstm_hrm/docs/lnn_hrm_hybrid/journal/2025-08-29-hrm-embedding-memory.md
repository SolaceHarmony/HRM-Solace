# E-HRM-EMB-MEM — HRM Embedding & Memory Deep-Dive (ACT v1)

- Date: 2025-08-29
- Repo: HRM-Solace (fork of sapientinc/HRM), models/hrm/hrm_act_v1.py
- Goal: Understand HRM’s embedding and memory/carry; map to our HRM+/xLSTM invariants and identify integration paths.

## Findings (code-level)
- Token Embedding & Scale:
  - CastedEmbedding(vocab_size, hidden_size) with truncated LeCun normal init; scaled by √hidden (embed_scale).
  - Optional learned positional encodings (`learned`) or RoPE (`rope`).
- Puzzle (Task) Embedding:
  - Optional CastedSparseEmbedding(num_puzzle_identifiers, puzzle_emb_ndim): zero-initialized; loaded into a per-batch `local_weights` and trained with a custom sparse SignSGD (distributed) optimizer that merges gradients by unique IDs.
  - Puzzle embedding is prepended to tokens (ceil division into full hidden-size rows), acting as conditioning memory per instance.
- Hierarchical Carry (Two-Tier States):
  - Inner carry = { z_H, z_L } of shape (B, seq_len + puzzle_len, hidden_size), dtype forward_dtype (bf16 by default).
  - Reasoning loop uses cycles: for each H cycle, run L cycles updating z_L with input injection from z_H + input_embeddings (RoPE or learned PE), then update z_H from z_L. Final pass computes a 1-step gradient update for both.
  - H_level/L_level are transformer blocks: self-attn + SwiGLU MLP with post-norm RMSNorm. Attention uses RoPE if configured.
- Halting (ACT wrapper with Q-head):
  - Q-head on z_H[:, 0] produces logits for {halt, continue}. During training, halting is thresholded by comparing Q_halt vs Q_continue; exploration uses ε-step random minimum halting step; target Q is bootstrapped by running inner once more (no replay buffer, no target nets).
  - Carry reset per sample when halted. Steps increment; max steps cap enforced.

## Mapping to HRM+/xLSTM Invariants
- Preserve→Gate: HRM uses residual post-norm blocks; inner carry is value-preserving across steps until explicit updates; halting gates compute time per sequence.
- Time Writes The Story: H/L cycles correspond to slow/fast phases; ACT imposes discrete step budgets akin to our Z5 scheduling (commit only after budget windows).
- Memory Types:
  - (1) Per-instance puzzle embedding (external key/value conditioning).
  - (2) Hierarchical internal carry (z_H, z_L) over token positions — a dual-level state memory evolved over cycles.
- Telemetry: Q-halt/continue logits provide halting telemetry; we can add energy/α-style audits analogously.

## Integration Ideas (xLSTM)
- New Memory Type(s):
  - HRMStateMemory: add a dual-level state (H/L) per token alongside sLSTM/mLSTM; expose a small ACT head to modulate compute or layer skipping in xLSTM.
  - Puzzle/task conditioning: add a sparse per-instance embedding (zero-init + SignSGD) as a slot in the embedding table; prepend rows (ceil-div style).
- Replacement/Augmentation:
  - Replace sLSTM block(s) in the stack with an HRMReasoning block (attention+SwiGLU with RoPE) and ACT halting on segment boundaries; train end-to-end with ponder.
  - Or treat HRM inner as a residual adapter around xLSTM outputs (post-stack), mirroring our wrapper, to minimize disruption.
- SSM Angle:
  - Re-interpret H/L cycles as a 2-timescale SSM kernel over positions (z_H, z_L as state); prototype an S4/S5-like kernel that mimics the H→L→H update algebra per step; compare to sLSTM.

## Hypotheses
- Adding puzzle/task conditioning improves sample efficiency and acts like a per-instance memory key.
- Dual-level state added to xLSTM (HRMStateMemory) stabilizes multi-step reasoning and enables ACT-driven compute budgets without hurting throughput.
- An SSM approximation of the H/L cycle may compress compute and improve long-range carry without full attention cost.

## Next
- Draft an adapter API in xLSTM to register a “memory type” with (init_state, update, readout), then plug HRMStateMemory as a prototype.
- Prototype a tiny HRMReasoning block inside a BlockStack slot and compare against sLSTM on a toy task; log ponder/energy.
- Journal entries with figures/tables; open issues for adapter API and HRMStateMemory.

## Links
- Code: models/hrm/hrm_act_v1.py; models/layers.py; models/sparse_embedding.py
- Issues: xLSTM-Metal #2 (per-block gating), #5 (energy), #6 (LM trainer), #7 (probe figures), #8 (modulators), #11 (LLR mask in HRM)
