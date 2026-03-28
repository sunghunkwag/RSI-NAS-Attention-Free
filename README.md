# RSI-NAS: Attention-Free Neural Architecture Search via Recursive Self-Improvement

**Unified integration of two independent research tracks into a single self-improving system.**

## What This Is

A system that **evolves attention-free neural architectures** using a three-layer recursive self-improvement engine:

| Layer | Component | Role |
|-------|-----------|------|
| 1 | `ModuleRegistry` | Primitive neural modules (NCA, GatedShiftMixer, FractalGNN, CoarseNCA, SqueezeExcite, GatedFFN) |
| 2 | `ArchitectureGrammar` | Composition rules: swap, add, remove, tweak, crossover |
| 3 | `ArchitectureMeta` | Meta-rules: library extraction, sequential composition, hyperparameter specialization |

Fitness is measured by **actual SGD training** on character-level language modeling (bits-per-character), not proxy metrics. MAP-Elites archive provides quality-diversity search over (param_count, depth) behavior space.

## Architecture Primitives (Zero Attention)

All modules have uniform `(B, L, D) → (B, L, D)` interface, O(L) complexity:

- **NCAStep** — Perceive-react-diffuse cellular automaton cycle
- **GatedShiftMixer** — Content-preserving long-range via fixed-offset gating (no Q/K/V, no softmax)
- **FractalGNNBlock** — Chunk-pool → graph convolution → gated broadcast
- **CoarseNCA** — Multi-scale: downsample → NCA at coarse resolution → upsample
- **SqueezeExcite** — Global channel recalibration
- **GatedFFN** — SwiGLU feed-forward

## Key Mechanism: Library Learning for Neural Architectures

DreamCoder-style pattern extraction applied to **layer topology** instead of program syntax:

1. Scan elite architectures for recurring 2-3 layer sequences
2. If a pattern appears in multiple high-fitness elites, extract it as a new single module
3. Register it in the ModuleRegistry — the evolutionary loop can now use it as a primitive
4. Prune unused generated modules to prevent vocabulary bloat

## Ablation Results (CPU, 5 seeds × 8 generations)

| Condition | BPC (mean ± std) | Generated Modules |
|-----------|-------------------|-------------------|
| FROZEN (no RSI) | 6.339 ± 0.151 | 0 |
| SELF-MODIFY (RSI active) | 6.316 ± 0.145 | 1-2 per seed |

- **Delta = +0.023 BPC** (SELF-MODIFY better), Cohen's d = 0.50 (medium effect)
- SELF-MODIFY wins 1/5 seeds, ties 4/5, **never loses**
- Library extraction **fires in all 5 seeds** — mechanism is live, not dead code
- Critical finding: **pruning-propagation race condition** — generated modules get pruned before they can propagate to elites

### Key Difference from Symbolic RSI

In the symbolic RSI system (`main.py`), self_encode and PolymorphicOp were **DEAD_CODE** — never reached by the evolutionary loop. Here, library extraction is **ACTIVE** in every seed. The bottleneck is operational (pruning timing), not structural (unreachability).

## Usage

```bash
# Quick demo (15 generations)
python rsi_nas.py

# Controlled ablation: FROZEN vs SELF-MODIFY
python rsi_nas.py ablation
```

## Requirements

```
torch>=2.0
numpy
```

## Source Lineage

| File | Origin |
|------|--------|
| NCAStep, GatedShiftMixer, CoarseNCA | `afn3.py` (Adaptive Field Network v3) |
| FractalGNNBlock | `fractal_gnn.py` |
| PerceptionFilter, ReactionGate | `nca_lm.py` (Neural Cellular Automata LM) |
| RSI 3-layer framework pattern | `main.py` (RSI-Exploration) |

## Next Steps (GPU required)

1. Fix pruning grace period (min 3× expansion_interval)
2. Scale: d_model=64, train_steps=200, 30+ generations, 10 seeds
3. Baseline comparison: Transformer (same param budget) vs best RSI-NAS architecture
4. That comparison table changes everything.

## Architecture

Built entirely via **no-code architect methodology**: design → instruct → delegate to AI → verify → correct. No direct coding. 820 lines, self-contained.
