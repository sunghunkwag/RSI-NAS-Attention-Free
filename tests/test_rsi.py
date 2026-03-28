"""
RSI Verification Test #2: Recursive Self-Improvement on Hard Targets
=====================================================================

In the previous test, x^2+2x+1 was already solvable with base ops,
so the self-improvement advantage was not apparent.

This test:
1. Target: sin(x) -- cannot be exactly represented with base ops (add, mul, square)
   -> Self-modification must compose Taylor series terms to approximate
2. max_depth=3 -- forces shallow trees -> library learning's depth amplification required
3. 500 generations x 5 seeds x 2 conditions
"""

import random
import numpy as np
import math
import logging
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.INFO)

from main import (
    VocabularyLayer, GrammarLayer, MetaGrammarLayer,
    LibraryLearner, ResourceBudget, CostGroundingLoop,
    MAPElitesArchive, EliteEntry, SelfImprovementEngine,
    build_rsi_system, sine_approximation_fitness,
    _eval_tree, EvalContext, PolymorphicOp, ExprNode
)


def _collect_ops(node, ops_set):
    ops_set.add(node.op_name)
    for c in node.children:
        _collect_ops(c, ops_set)


def run_hard_experiment(seed, generations, pop_size, expansion_interval,
                        max_depth, fitness_fn, label):
    random.seed(seed)
    np.random.seed(seed)

    engine = build_rsi_system(
        fitness_fn=fitness_fn,
        max_depth=max_depth,
        budget_ops=100_000,
        budget_seconds=60.0,
        expansion_interval=expansion_interval,
        use_library_learning=True,
        library_min_depth=2,
        library_min_freq=2,
        library_max_additions=3,
        pruning_window=10,
        pruning_threshold=0.10,
    )

    base_ops = {op.name for op in engine.vocab.all_ops()}
    trajectory = []
    best_at = {0.2: None, 0.3: None, 0.4: None, 0.5: None, 0.6: None, 0.7: None}

    for gen in range(generations):
        record = engine.step(pop_size)

        current_ops = {op.name for op in engine.vocab.all_ops()}
        generated_ops = current_ops - base_ops
        poly_ops = {op.name for op in engine.vocab.all_ops()
                    if isinstance(op, PolymorphicOp)}

        elites = list(engine.archive._grid.values())
        elites_using_gen = sum(1 for e in elites
                              if _get_tree_ops(e.tree) & generated_ops)

        # Track convergence speed
        for thresh in best_at:
            if best_at[thresh] is None and record["archive_best"] >= thresh:
                best_at[thresh] = gen + 1

        if (gen + 1) % 50 == 0 or gen == 0:
            trajectory.append({
                "gen": gen + 1,
                "best": record["archive_best"],
                "coverage": record["archive_coverage"],
                "vocab": record["vocab_size"],
                "gen_ops": len(generated_ops),
                "poly": len(poly_ops),
                "util": elites_using_gen / max(len(elites), 1),
                "rules": record["grammar_rules"],
            })

    return {
        "label": label,
        "seed": seed,
        "trajectory": trajectory,
        "best_at": best_at,
        "final_best": trajectory[-1]["best"],
        "final_vocab": trajectory[-1]["vocab"],
        "final_gen_ops": trajectory[-1]["gen_ops"],
        "final_util": trajectory[-1]["util"],
    }


def _get_tree_ops(tree):
    ops = set()
    _collect_ops(tree, ops)
    return ops


def main():
    print("=" * 80)
    print("  RSI Verification #2: Hard Target (sin(x), depth=3)")
    print("  Harder Target: sin(x) approximation with shallow trees")
    print("=" * 80)

    SEEDS = [42, 123, 456, 789, 1024]
    GENERATIONS = 500
    POP_SIZE = 20
    MAX_DEPTH = 3  # Shallow tree -> library learning required

    # ===================================================================
    # Test 1: sin(x) with depth=3
    # ===================================================================
    print(f"\n>> Target: sin(x), max_depth={MAX_DEPTH}")
    print(f"  sin(x) cannot be exactly represented with add/mul/square.")
    print(f"  depth=3 is insufficient to directly construct Taylor series terms.")
    print(f"  -> Library learning must compress deep subtrees into single ops")
    print(f"     to expand effective depth.\n")

    # Condition A: FROZEN
    print("  Condition A: FROZEN")
    frozen = []
    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        r = run_hard_experiment(seed, GENERATIONS, POP_SIZE, 999999,
                               MAX_DEPTH, sine_approximation_fitness,
                               f"FROZEN_sin_d{MAX_DEPTH}")
        elapsed = time.time() - t0
        print(f"    Seed {seed}: best={r['final_best']:.4f} vocab={r['final_vocab']} "
              f"time={elapsed:.1f}s")
        frozen.append(r)

    # Condition B: SELF-MODIFY
    print("\n  Condition B: SELF-MODIFY (expansion every 5 gen)")
    modify = []
    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        r = run_hard_experiment(seed, GENERATIONS, POP_SIZE, 5,
                               MAX_DEPTH, sine_approximation_fitness,
                               f"MODIFY_sin_d{MAX_DEPTH}")
        elapsed = time.time() - t0
        print(f"    Seed {seed}: best={r['final_best']:.4f} vocab={r['final_vocab']} "
              f"gen_ops={r['final_gen_ops']} util={r['final_util']*100:.0f}% "
              f"time={elapsed:.1f}s")
        modify.append(r)

    # ===================================================================
    # Test 2: sin(x) with depth=5 (control -- easier for library learning)
    # ===================================================================
    print(f"\n>> Control: sin(x), max_depth=5")

    print("  Condition A: FROZEN (depth=5)")
    frozen5 = []
    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        r = run_hard_experiment(seed, GENERATIONS, POP_SIZE, 999999,
                               5, sine_approximation_fitness,
                               f"FROZEN_sin_d5")
        elapsed = time.time() - t0
        print(f"    Seed {seed}: best={r['final_best']:.4f} time={elapsed:.1f}s")
        frozen5.append(r)

    print("\n  Condition B: SELF-MODIFY (depth=5)")
    modify5 = []
    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        r = run_hard_experiment(seed, GENERATIONS, POP_SIZE, 5,
                               5, sine_approximation_fitness,
                               f"MODIFY_sin_d5")
        elapsed = time.time() - t0
        print(f"    Seed {seed}: best={r['final_best']:.4f} vocab={r['final_vocab']} "
              f"gen_ops={r['final_gen_ops']} util={r['final_util']*100:.0f}% "
              f"time={elapsed:.1f}s")
        modify5.append(r)

    # ===================================================================
    # Analysis
    # ===================================================================
    print("\n" + "=" * 80)
    print("  Analysis Results")
    print("=" * 80)

    for test_name, fr, mo, depth in [
        ("sin(x), depth=3", frozen, modify, 3),
        ("sin(x), depth=5", frozen5, modify5, 5)
    ]:
        fb = [r["final_best"] for r in fr]
        mb = [r["final_best"] for r in mo]
        delta = np.mean(mb) - np.mean(fb)

        print(f"\n  >> {test_name}")
        print(f"    FROZEN:      {np.mean(fb):.4f} +/- {np.std(fb):.4f}  {fb}")
        print(f"    SELF-MODIFY: {np.mean(mb):.4f} +/- {np.std(mb):.4f}  {mb}")
        print(f"    Delta = {delta:+.4f}")

        if delta > 0.01:
            print(f"    [PASS] Self-modification improves fitness (+{delta:.4f})")
        elif delta > 0.001:
            print(f"    [MARGINAL] Slight improvement (+{delta:.4f})")
        else:
            print(f"    [FAIL] No improvement (Delta = {delta:+.4f})")

        # Convergence speed
        print(f"\n    Convergence Speed:")
        for thresh in [0.3, 0.4, 0.5]:
            fr_gen = [r["best_at"].get(thresh, None) for r in fr]
            mo_gen = [r["best_at"].get(thresh, None) for r in mo]
            fr_avg = np.mean([g for g in fr_gen if g is not None]) if any(g is not None for g in fr_gen) else float('inf')
            mo_avg = np.mean([g for g in mo_gen if g is not None]) if any(g is not None for g in mo_gen) else float('inf')
            print(f"      -> {thresh:.1f} fitness: FROZEN={fr_avg:.0f}gen, MODIFY={mo_avg:.0f}gen "
                  f"(faster by {fr_avg-mo_avg:.0f} gen)")

        # Trajectory
        print(f"\n    Fitness Trajectory (FROZEN vs SELF-MODIFY):")
        print(f"    {'Gen':>5}  {'FROZEN':>8}  {'MODIFY':>8}  {'Delta':>8}  {'Util%':>6}")
        fr_traj_avg = {}
        mo_traj_avg = {}
        mo_util_avg = {}
        for r in fr:
            for t in r["trajectory"]:
                fr_traj_avg.setdefault(t["gen"], []).append(t["best"])
        for r in mo:
            for t in r["trajectory"]:
                mo_traj_avg.setdefault(t["gen"], []).append(t["best"])
                mo_util_avg.setdefault(t["gen"], []).append(t["util"])

        for gen in sorted(fr_traj_avg.keys()):
            if gen in mo_traj_avg:
                fa = np.mean(fr_traj_avg[gen])
                ma = np.mean(mo_traj_avg[gen])
                ua = np.mean(mo_util_avg.get(gen, [0]))
                print(f"    {gen:5d}  {fa:8.4f}  {ma:8.4f}  {ma-fa:+8.4f}  {ua*100:5.1f}%")

    # ===================================================================
    # Final Verdict
    # ===================================================================
    print("\n" + "=" * 80)
    print("  Final Verdict")
    print("=" * 80)

    d3_delta = np.mean([r["final_best"] for r in modify]) - np.mean([r["final_best"] for r in frozen])
    d5_delta = np.mean([r["final_best"] for r in modify5]) - np.mean([r["final_best"] for r in frozen5])
    d3_util = np.mean([r["final_util"] for r in modify])
    d5_util = np.mean([r["final_util"] for r in modify5])

    print(f"\n  sin(x) depth=3: Delta = {d3_delta:+.4f}, util = {d3_util*100:.1f}%")
    print(f"  sin(x) depth=5: Delta = {d5_delta:+.4f}, util = {d5_util*100:.1f}%")

    if d3_delta > 0.01 or d5_delta > 0.01:
        print(f"\n  VERDICT: RECURSIVE_SELF_IMPROVEMENT_CONFIRMED")
        print(f"  Self-modification provides measurable improvement on hard targets.")
        if d3_delta > d5_delta:
            print(f"  Effect is larger at depth=3 -> depth amplification is the key mechanism.")
    elif d3_delta > 0.001 or d5_delta > 0.001:
        print(f"\n  VERDICT: MARGINAL_IMPROVEMENT")
        print(f"  Marginal improvement. Recursive mechanism works but effect is small.")
    else:
        print(f"\n  VERDICT: SELF_MODIFICATION_WITHOUT_IMPROVEMENT")
        print(f"  Self-modification does not lead to improvement even on hard targets.")
        print(f"  Possible causes:")
        print(f"  1. Meta-grammar fails to create 'useful' ops (random composition)")
        print(f"  2. 500 generations insufficient (longer runs needed)")
        print(f"  3. Library learning's subtree extraction struggles to discover")
        print(f"     Taylor series coefficients needed for sin approximation")


if __name__ == "__main__":
    main()
