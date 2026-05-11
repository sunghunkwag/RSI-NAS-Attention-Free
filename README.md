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
| 3a | `EIEConfig` | Mutable instrument policy optimized by the outer RSI loop |
| 3b | `omega_adapter` | AFIRSI residue export and OMEGA-style instrument synthesis |
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

## Outer Recursive Policy Search

`open_ended_rsi.py` adds an outer loop over the EIE/AFIRSI policy itself. The
inner RSI loop still performs real architecture search and SGD evaluation. The
outer loop then mutates the policy that controls pruning grace, probing,
generated-module experiment scaffolds, and meta-operator evidence weighting.

The outer loop accepts a policy mutation only when it improves over the previous
champion on paired seeds and keeps the mechanism active:

- generated modules receive real evaluations,
- generated modules enter the MAP-Elites archive,
- failure residues trigger instrument mutations,
- meta-operator weights update from generated-module evidence,
- mean BPC beats the non-EIE `SELF-MODIFY` baseline,
- the mutated EIE policy scores above the previous champion policy.

This is a bounded open-ended-RSI proxy: it demonstrates recursive
self-modification of the RSI instrument policy across validation cycles. It is
not mathematical proof of unbounded open-ended RSI.

## AFIRSI-OMEGA Integration Prototype

`open_ended_rsi_omega.py` replaces the fixed hand-designed
`mutate_candidate()` outer loop with a residue-conditioned instrument synthesis
loop inspired by the OMEGA-THDSE pattern: deterministic symbolic interpretation,
explicit mutation contracts, and causal provenance.

The integration path is:

`FailureResidue -> StructuredAFIRSIResidue -> missing-instrument constraints ->
InstrumentPatch -> EIEConfig policy -> paired-seed validation -> holdout-seed
validation -> accepted parent policy`.

The adapter is intentionally local and does not call external APIs or hidden
services. It uses OMEGA-style deterministic synthesis rather than importing a
runtime dependency from the separate `sunghunkwag/OMEGA-THDSE` repository.

The integration layer contains:

- `omega_adapter/schemas.py`: JSON schema for residues, instrument patches, and
  the mutation contract.
- `omega_adapter/residue_export.py`: exporter for real residues produced by
  completed RSI-NAS runs.
- `omega_adapter/instrument_generator.py`: residue-conditioned OMEGA-style
  patch generator.
- `omega_adapter/policy_import.py`: validated conversion from instrument patch
  to executable `EIEConfig`.
- `open_ended_rsi_omega.py`: paired-seed plus holdout-seed outer loop with JSON
  provenance report.

The mutation contract permits EIE policy fields, generated-module evaluation
budget, candidate scoring coefficients, evaluator terms, archive priority, and
scaffold strategy. It forbids fake BPC values, direct success flags, bypassing
real build/train/evaluate, and accepting without paired and holdout validation.

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

Run the outer recursive policy-improvement validation:

```bash
python open_ended_rsi.py
```

Use `--cycles`, `--seeds`, `--generations`, and `--train-steps` to increase the
validation budget. The default run is intentionally CPU-sized.

Run the AFIRSI-OMEGA residue-conditioned validation:

```bash
python open_ended_rsi_omega.py --json
```

The default paired seeds are `7,11,19`; the default holdout seeds are `23,29`.
For an explicit CPU-sized run:

```bash
python open_ended_rsi_omega.py --seeds 7,11,19 --holdout-seeds 23,29 --cycles 2 --generations 5 --train-steps 2 --json
```

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
51 passed
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

Outer recursive policy-improvement validation on three CPU seeds:

```text
cycle=0 accepted=False champion=seed_policy score=0.0757 delta=+0.0555 archive=5 evals=17
cycle=1 accepted=True champion=cycle1_clean_probe_archive score=0.0805 delta=+0.0553 archive=9 evals=19
cycle=2 accepted=False champion=cycle1_clean_probe_archive score=0.0805 delta=+0.0553 archive=9 evals=19
open_ended_proxy_valid: true
accepted_improvements: 1
```

AFIRSI-OMEGA residue-conditioned validation on paired seeds `7,11,19` and
holdout seeds `23,29`:

```text
omega_validation_valid: true
accepted_improvements: 1
accepted candidate: cycle1_residue_pressure_disagreement_9f2fe86984
paired mean BPC delta: +0.0781
holdout mean BPC delta: +0.1034
generated evaluations: 46
generated archive insertions: 16
meta-operator policy updates: 12
source residues used: 22
```

Boundary:

```text
This demonstrates residue-conditioned recursive improvement of the EIE/AFIRSI instrument policy across validation cycles. It is not mathematical proof of unbounded open-ended RSI, AGI, or autonomous self-improvement in the real world.
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
