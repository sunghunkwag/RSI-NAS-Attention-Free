# RSI-NAS: Attention-Free Neural Architecture Search with EIE/AFIRSI

This repository implements a recursive self-improvement loop for attention-free
neural architecture search. The system does not optimize a proxy score only: each
candidate architecture is built, trained with SGD on character-level language
modeling, and inserted into a MAP-Elites archive when it improves a behavior
cell.

The current implementation extends the original three-layer RSI-NAS system with
EIE/AFIRSI: Epistemic Instrument Evolution for generated neural modules. This was
added to address the pruning-propagation race observed in the previous ablation:
generated modules could be removed before the search loop had enough time to
sample, train, and archive architectures that used them.

## Architecture

| Layer | Component | Role |
| --- | --- | --- |
| 1 | `ModuleRegistry` | Stores primitive and generated attention-free modules with lifecycle evidence |
| 2 | `ArchitectureGrammar` | Builds and mutates architecture genomes |
| 3 | `ArchitectureMeta` | Creates new modules through library extraction, composition, and specialization |
| 4 | `EpistemicInstrumentEvolver` | Detects instrument failure residue and mutates pruning/probing policy |
| Archive | `ArchitectureArchive` | MAP-Elites quality-diversity archive over parameter count and depth |

## Attention-Free Primitives

All primitives expose a uniform `(B, L, D) -> (B, L, D)` interface and avoid
attention softmax:

- `NCAStep`: perceive-react-diffuse cellular automaton block
- `GatedShiftMixer`: fixed-offset gated sequence mixing
- `FractalGNNBlock`: chunk pooling, graph convolution, and gated broadcast
- `CoarseNCA`: downsample, coarse NCA update, and upsample
- `SqueezeExcite`: global channel recalibration
- `GatedFFN`: SwiGLU-style feed-forward block

## Recursive Self-Improvement Loop

Each generation performs:

1. Generate candidate architectures through mutation or crossover.
2. Optionally instrument candidates with under-tested generated modules when EIE
   has activated probing.
3. Train each candidate with SGD and compute bits-per-character.
4. Insert improvements into the MAP-Elites archive.
5. Periodically expand the module vocabulary through meta-grammar actions.
6. Periodically prune generated modules, unless EIE detects insufficient evidence.

The EIE/AFIRSI path is not a standalone simulator. It is wired into the actual
search loop and changes the live instruments used by the loop.

## EIE/AFIRSI Mechanism

The ChatGPT brainstorming thread produced the key design constraint: real RSI
must not only generate new candidates, it must also rewrite the instruments that
observe, judge, and preserve evidence.

This implementation turns that idea into code:

- `GeneratedModuleRecord` tracks birth generation, source action, parents,
  evaluations, archive insertions, elite usage, best fitness, and pruning
  attempts.
- `FailureResidue` records a concrete epistemic failure:
  `PRUNING_PROPAGATION_RACE`.
- `EpistemicInstrumentEvolver` mutates `PruningPolicy` when a generated module
  is about to be pruned before the evidence window is satisfied.
- The mutated policy applies a grace window of at least `3 * expansion_interval`,
  requires a minimum generated-module evaluation count, and enables generated
  module probing.
- `ArchitectureMeta.instrument_candidate()` injects under-tested generated
  modules into real candidate genomes so they receive actual SGD evaluation.
- First evaluations use a clean generated-module probe scaffold, so the system
  changes the experiment unit instead of only inserting modules into noisy random
  genomes.
- `ArchitectureMeta.refresh_meta_operator_policy()` updates the meta-layer's own
  `library`, `compose`, and `specialize` operator weights from generated-module
  evaluation and archive evidence.

This creates a closed loop:

`failure residue -> instrument mutation -> changed experiment unit -> real
evaluation evidence -> meta-operator policy update -> archive or prune decision`.

## Usage

```bash
python rsi_nas.py
```

Run the controlled ablation:

```bash
python rsi_nas.py ablation
```

The ablation now compares:

- `FROZEN`: no design-space expansion
- `SELF-MODIFY`: baseline meta-grammar expansion without EIE
- `AFIRSI-EIE`: meta-grammar expansion plus epistemic instrument evolution

Run the focused EIE mechanism validation:

```bash
python validate_eie.py
```

This checks a bounded claim only: generated modules that baseline
self-modification prunes immediately are protected, probed, and evaluated under
AFIRSI-EIE. It is not a proof of open-ended RSI or consistent BPC improvement.

## Validation

Run the test suite:

```bash
python -m pytest test_rsi_nas.py -q
```

The suite covers primitive modules, registry behavior, genome construction,
network building, SGD fitness evaluation, grammar mutation, meta-grammar
expansion, MAP-Elites insertion, loop integration, and EIE/AFIRSI behavior.

Current local validation:

```text
50 passed
```

Focused EIE validation on five CPU seeds:

```text
SELF-MODIFY generated evaluations: 0
AFIRSI-EIE generated evaluations: 45
AFIRSI-EIE generated archive insertions: 21
AFIRSI-EIE meta-operator policy updates: 25
SELF-MODIFY mean best BPC: 7.6982
AFIRSI-EIE mean best BPC: 7.6757
Mean BPC delta (SELF-MODIFY - AFIRSI-EIE): +0.0225
AFIRSI-EIE seed wins: 3 / 5
Mechanism valid: true
```

## Source Lineage

| Source | Role |
| --- | --- |
| `afn3.py` | GatedShiftMixer, NCAStep, CoarseNCA, SqueezeExcite, GatedFFN lineage |
| `fractal_gnn.py` | FractalGNNBlock lineage |
| `nca_lm.py` | PerceptionFilter and ReactionGate lineage |
| `main.py` | Three-layer RSI framework pattern |
| ChatGPT RSI/EIE discussion | AFIRSI framing: failure residue drives instrument evolution |

## Next Research Checks

- Run longer GPU ablations with `d_model=64`, `train_steps=200`, 30+ generations,
  and multiple seeds.
- Compare `AFIRSI-EIE` against `SELF-MODIFY` and `FROZEN` on final BPC, archive
  coverage, generated-module survival, and generated-module archive insertions.
- Add a fixed-budget Transformer baseline with matched parameter counts.
