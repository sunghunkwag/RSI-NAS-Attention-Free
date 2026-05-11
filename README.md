# RSI-NAS with AFIRSI/EIE

Attention-free neural architecture search with bounded, residue-conditioned
instrument-policy improvement.

This repository runs real RSI-NAS executions: candidate architectures are built,
trained with SGD on character-level language modeling, scored by BPC, and inserted
into a MAP-Elites archive only through real evaluation evidence.

## What This Implements

- Attention-free NAS over NCA, gated shift mixing, fractal GNN, coarse NCA,
  squeeze-excite, and gated FFN primitives.
- EIE/AFIRSI instrumentation for generated modules that might otherwise be
  pruned before enough evidence is collected.
- A first-class AFIRSI core:
  `FailureResidueLedger`, `ObservationChannel`, `Evaluator`, `ExperimentUnit`,
  `OperatorGenerator`, `InstrumentMutationContract`, and
  `ProblemSpaceVersionGraph`.
- A bounded recursive policy kernel in `bounded_rsi.py`.

## Bounded Recursive Loop

The bounded kernel executes:

```text
P0 policy
-> real RSI-NAS execution
-> real AFIRSI residues
-> residue-conditioned patch generation
-> mutation-contract validation
-> paired-seed validation
-> holdout-seed validation
-> accepted P1
-> P1 becomes the parent for cycle 2
-> real execution under P1
-> P1-derived residues
-> P2 candidate generation
-> P2 validation against P1
```

Success is strict:

- `P0 -> accepted P1`: one bounded self-improvement step.
- `P0 -> accepted P1 -> P2 attempt`: minimal recursive attempt.
- `P0 -> accepted P1 -> accepted P2`: bounded recursive success.

## Second-Order Residues

Beyond `PRUNING_PROPAGATION_RACE`, the kernel now diagnoses:

- `ARCHIVE_STAGNATION`
- `OPERATOR_GENERATOR_MODE_COLLAPSE`
- `EVALUATOR_NOISE_OR_OVERFIT`
- `POLICY_SATURATION`
- `SCAFFOLD_BIAS`
- `META_OPERATOR_IMBALANCE`
- `PATCH_EFFECTIVENESS_FAILURE`

These residues are derived from live execution summaries, validation results,
generated-module records, archive activity, and policy lineage. They are not
synthetic success flags.

## Latest Default Result

Default CPU-sized bounded run:

```text
P1 became parent of cycle 2: true
P2 came from P1-derived residues: true
P2 was compared against P1: true
bounded_recursive_success: true
partial_recursive_success: false
accepted_policy_count: 2
```

Cycle-2 residue distribution:

```text
ARCHIVE_STAGNATION: 1
POLICY_SATURATION: 1
PRUNING_PROPAGATION_RACE: 27
```

Accepted P2 patch family:

```text
policy_saturation_next_bottleneck
```

Rejected P2 candidates were also diagnosed structurally, including
`SCAFFOLD_BIAS`.

## Run

```bash
python rsi_nas.py
python validate_eie.py --json
python open_ended_rsi.py --json
python open_ended_rsi_omega.py --json
python bounded_rsi.py --json
```

Useful bounded-kernel options:

```bash
python bounded_rsi.py \
  --cycles 2 \
  --seeds 7,11,19 \
  --holdout-seeds 23,29 \
  --generations 5 \
  --population-size 3 \
  --train-steps 2 \
  --json
```

## Test

```bash
python -m pytest -q
python -m compileall afirsi_core
git diff --check
```

Latest local validation:

```text
112 passed
validate_eie.py --json: mechanism_valid true
open_ended_rsi.py --json: open_ended_proxy_valid true
open_ended_rsi_omega.py --json: omega_validation_valid true
bounded_rsi.py --json: bounded_recursive_success true
```

## Anti-Shortcut Boundary

The implementation forbids fake BPC, fake generated evaluations, fake archive
insertions, direct success flags, skipped build/train/evaluate, seed-specific
hardcoding, residue deletion, unvalidated policy acceptance, and accepted
problem-space versions for rejected patches.

Boundary statement:

```text
This validates a bounded recursive self-improvement kernel over AFIRSI/EIE instrument policy inside RSI-NAS. It is not proof of unbounded open-ended RSI, AGI, ASI, or real-world autonomous self-improvement.
```
