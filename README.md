# Residue-Evolved Generalization Harness

Suggested repository slug: `residue-evolved-generalization-harness`

This is a bounded research harness for testing residue-driven evaluator
evolution, learner-program synthesis, runtime primitive-type invention, and
held-out validation on local real-data tasks.

It is not a verified AGI system. It does not claim open-ended autonomy,
human-level intelligence, or external benchmark success.

## What Is Implemented

- Local Python-inventory architecture synthesis into RSI-NAS candidates.
- Cross-domain bounded program selection over algorithmic, causal, and grid
  tasks.
- Real-data learner selection using train / validation / held-out test splits.
- EIE-style evaluator evolution from validation residue.
- Self-generated validation goals from observed failure residue.
- Runtime learner-program invention using bounded feature / estimator programs.
- Runtime primitive-type invention through a residue-cited primitive registry.
- Cross-task transfer of accepted learner programs.
- Automatic discovery of one unrequested local real-data domain.
- A verified local action trace for multi-step execution.

## Main Files

```text
local_agi_architect.py
validate_agi_claim.py
adaptive_domain_programs.py
cross_domain_agi_system.py
real_world_generalization_benchmark.py
eie_real_world_instrument_evolver.py
emergent_generalization_system.py
```

Tests are in:

```text
tests/test_local_agi_architect.py
tests/test_adaptive_domain_programs.py
tests/test_cross_domain_agi_system.py
tests/test_real_world_generalization_benchmark.py
tests/test_eie_real_world_instrument_evolver.py
tests/test_emergent_generalization_system.py
```

## Latest Local Result

Command:

```bash
python emergent_generalization_system.py \
  --json-output emergent_generalization_report.json
```

Observed output:

```text
final_verdict=EMERGENT_GENERALIZATION_MECHANISMS_PASSED_LOCAL_GATES
mechanism_success=True
agi_claim_verified=False
self_generated_goal_count=5
invented_program_count=8
invented_primitive_type_count=4
transferred_program_count=3
```

Held-out task results:

```text
breast_cancer_diagnosis clean=0.977 stress_floor=0.958
wine_chemical_origin clean=0.978 stress_floor=0.978
handwritten_digit_recognition clean=0.989 stress_floor=0.989
iris_morphology_species open_world_clean=0.967 stress_floor=0.900
```

Gate summary:

```text
PASS self_generated_validation_goals
PASS residue_conditioned_learner_program_invention
PASS cross_task_program_transfer
PASS heldout_real_data_and_stress_success
PASS test_split_not_used_for_generation
PASS unbounded_open_world_domain_creation
PASS autonomous_long_horizon_tool_use
PASS recursive_capability_improvement_without_manual_primitives
FAIL external_frontier_benchmark_validation
```

## Validation

Current local checks:

```bash
python -m pytest -q
python -m compileall emergent_generalization_system.py
git diff --check
```

Observed:

```text
132 passed
compileall passed
git diff --check: README CRLF warning only
```

## Evidence Boundaries

This repository provides evidence for a local mechanism stack:

```text
residue -> evaluator mutation -> self-generated goals
residue -> learner-program synthesis -> primitive-type invention
accepted programs -> cross-task transfer -> held-out testing
```

It does not provide external AGI evidence. The current remaining failed gate is
independent frontier validation, such as ARC-AGI, GAIA, SWE-bench, or
METR-style tasks.

## Run Additional Checks

```bash
python real_world_generalization_benchmark.py \
  --json-output real_world_generalization_report.json

python eie_real_world_instrument_evolver.py \
  --json-output eie_real_world_report.json

python cross_domain_agi_system.py \
  --json-output cross_domain_agi_report.json

python validate_agi_claim.py
```

## Claim

Supported claim:

```text
A bounded local generalization harness implements residue-driven evaluator
evolution, learner-program synthesis, runtime primitive-type invention,
cross-task transfer, open-world local dataset discovery, and held-out
validation.
```

Unsupported claim:

```text
AGI has been achieved.
```
