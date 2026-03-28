"""
PROPER ABLATION STUDY — No tricks, no scaffolding.

Remaining code changes from baseline:
  FIX 1: Depth enforcement (reject mutated trees > max_depth)
  FIX 2: Adaptive min_frequency (lower extraction threshold when archive < 20)
  FIX 3: OpFitnessTracker (fitness-driven sampling weights + pruning)
  FIX 4: Fitness-weighted library extraction (score by elite fitness, not just freq)

Conditions:
  A) FROZEN        — no expansion, no library learning (baseline)
  B) LIB-ONLY      — library learning ON, tracker OFF (pure freq-based extraction)
  C) LIB+TRACKER   — library learning ON, tracker ON (fitness-driven extraction)

Targets:
  1) quintic (x^5) at depth=2 — requires depth amplification
  2) septic (x^7) at depth=2  — harder, requires more amplification
  3) sin(x) at depth=3        — original target, must NOT regress

All conditions use depth enforcement (FIX 1) — this is a bug fix, not a feature.
"""
import sys, random, numpy as np
sys.path.insert(0, '/sessions/exciting-compassionate-cray/repos/RSI-Exploration-20260324')
from main import (
    VocabularyLayer, GrammarLayer, MetaGrammarLayer, MAPElitesArchive,
    ResourceBudget, CostGroundingLoop, SelfImprovementEngine,
    LibraryLearner, OpFitnessTracker,
    quintic_fitness, septic_fitness, symbolic_regression_fitness
)

NUM_SEEDS = 10
POP = 30


def build(seed, mode, fitness_fn, max_depth, gens):
    random.seed(seed); np.random.seed(seed)
    vocab = VocabularyLayer()
    grammar = GrammarLayer(vocab, max_depth=max_depth)

    tracker = None
    lib = None

    if mode in ("lib-only", "lib+tracker"):
        lib = LibraryLearner(
            vocab=vocab, min_subtree_depth=2, min_frequency=2,
            max_library_additions=2,
            fitness_tracker=None  # no tracker for lib-only
        )

    if mode == "lib+tracker":
        tracker = OpFitnessTracker()
        vocab.set_fitness_tracker(tracker)
        lib.fitness_tracker = tracker

    meta = MetaGrammarLayer(vocab, grammar, library_learner=lib)
    budget = ResourceBudget(max_compute_ops=100_000, max_wall_seconds=60)
    cost = CostGroundingLoop(budget)
    archive = MAPElitesArchive(dims=[6, 10])
    interval = 10 if mode != "frozen" else 999999

    return SelfImprovementEngine(
        vocab=vocab, grammar=grammar, meta_grammar=meta,
        archive=archive, cost_loop=cost, fitness_fn=fitness_fn,
        expansion_interval=interval, fitness_tracker=tracker
    )


def run_condition(label, mode, fitness_fn, max_depth, gens, seeds):
    results = []
    for s in seeds:
        e = build(s, mode, fitness_fn, max_depth, gens)
        h = e.run(gens, POP)
        f = h[-1]["archive_best"]
        v = h[-1]["vocab_size"]
        results.append(f)
    return results


def ablation(target_name, fitness_fn, max_depth, gens):
    seeds = list(range(NUM_SEEDS))
    print(f"\n{'='*72}")
    print(f"  TARGET: {target_name}  |  max_depth={max_depth}  |  gens={gens}")
    print(f"{'='*72}")

    conditions = [
        ("A) FROZEN",      "frozen"),
        ("B) LIB-ONLY",    "lib-only"),
        ("C) LIB+TRACKER", "lib+tracker"),
    ]

    all_results = {}
    for label, mode in conditions:
        res = run_condition(label, mode, fitness_fn, max_depth, gens, seeds)
        all_results[label] = res
        m, s = np.mean(res), np.std(res)
        print(f"  {label:20s}  mean={m:.6f} ± {s:.6f}  "
              f"min={min(res):.4f}  max={max(res):.4f}")

    # Pairwise comparisons
    a = all_results["A) FROZEN"]
    b = all_results["B) LIB-ONLY"]
    c = all_results["C) LIB+TRACKER"]

    print(f"\n  Deltas:")
    print(f"    B-A (library extraction effect): {np.mean(b)-np.mean(a):+.6f}")
    print(f"    C-A (full RSI effect):           {np.mean(c)-np.mean(a):+.6f}")
    print(f"    C-B (tracker added value):       {np.mean(c)-np.mean(b):+.6f}")

    # Per-seed wins
    ba_wins = sum(1 for bi, ai in zip(b, a) if bi > ai + 0.01)
    ca_wins = sum(1 for ci, ai in zip(c, a) if ci > ai + 0.01)
    print(f"\n  Per-seed wins vs FROZEN (>0.01 threshold):")
    print(f"    B vs A: {ba_wins}/{NUM_SEEDS}")
    print(f"    C vs A: {ca_wins}/{NUM_SEEDS}")

    return all_results


print("=" * 72)
print("  ABLATION STUDY: Recursive Self-Improvement Verification")
print("  No trial mechanism. No library_insert mutation. No exploration bonus.")
print("  Only: depth enforcement + adaptive min_freq + fitness tracker.")
print("=" * 72)

# Test 1: quintic (the hard case)
r1 = ablation("x^5 (quintic)", quintic_fitness, max_depth=2, gens=200)

# Test 2: septic (even harder)
r2 = ablation("x^7 (septic)", septic_fitness, max_depth=2, gens=200)

# Test 3: sin(x) (original — must not regress)
r3 = ablation("sin(x) (original)", symbolic_regression_fitness, max_depth=3, gens=100)

# Final summary
print(f"\n{'='*72}")
print(f"  FINAL SUMMARY")
print(f"{'='*72}")
targets = [
    ("x^5", r1), ("x^7", r2), ("sin(x)", r3)
]
for name, r in targets:
    a_m = np.mean(r["A) FROZEN"])
    c_m = np.mean(r["C) LIB+TRACKER"])
    delta = c_m - a_m
    verdict = "RSI HELPS" if delta > 0.05 else ("NEUTRAL" if abs(delta) <= 0.05 else "RSI HURTS")
    print(f"  {name:8s}: FROZEN={a_m:.4f}  LIB+TRACKER={c_m:.4f}  "
          f"delta={delta:+.4f}  [{verdict}]")
