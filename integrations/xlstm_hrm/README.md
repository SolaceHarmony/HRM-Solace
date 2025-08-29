# HRM + xLSTM Integration (MPS-first)

This integration brings the HRM layer, telemetry, and probes into the HRM-Solace repo for experiments with the official xLSTM stack.

- Code: `src/lnn_hrm/*`
- Docs: `docs/lnn_hrm_hybrid/*` (origin paper, research journal, citation sea, guides)
- Examples: `examples/*` (wrapper demo, tiny trainer, 5‑HT sweep)
- Tools: `tools/telem`, `tools/harvest`, `tools/journal`
- Tests: `tests/*` (CPU + MPS split)
- Scripts: `scripts/mps_smoke.sh`
- Lab: `lab/dendritic_comb_probe.py`

Notes
- MPS required (Apple). No CUDA fallback.
- Set `PYTHONPATH` to repo root or run via `pip install -e .` if packaged.

