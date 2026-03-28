"""
RSI (Recursive Self-Improvement) Verification Test
====================================================

Three core questions:
1. Does the system autonomously generate new operations/rules? (Self-modification)
2. Do those discoveries lead to better performance? (Improvement)
3. Does better performance lead to even better discoveries? (Recursion)

Test design:
- Condition A (FROZEN): Meta-grammar expansion disabled (expansion_interval=999999)
- Condition B (SELF-MODIFY): Meta-grammar expansion enabled (expansion_interval=5)
- 5 seeds x 2 conditions x 500 generations
- Metrics: fitness trajectory, vocab growth, library extraction rate,
           new-op utilization rate, recursive loop depth
"""

import random
import copy
import numpy as np
import logging
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.disable(logging.INFO)  # suppress verbose logs

from main import (
    VocabularyLayer, GrammarLayer, MetaGrammarLayer,
    LibraryLearner, ResourceBudget, CostGroundingLoop,
    MAPElitesArchive, EnhancedMAPElitesArchive,
    EliteEntry, SelfImprovementEngine, build_rsi_system,
    symbolic_regression_fitness, _eval_tree, EvalContext,
    PolymorphicOp, ExprNode, PrimitiveOp
)


def _collect_ops(node, ops_set):
    """Collect all op names used in a tree."""
    ops_set.add(node.op_name)
    for c in node.children:
        _collect_ops(c, ops_set)


def _tree_uses_generated_ops(tree, base_ops):
    """Check if a tree uses any generated ops beyond the base ops."""
    used = set()
    _collect_ops(tree, used)
    return used - base_ops


def run_experiment(seed, generations, pop_size, expansion_interval, label):
    """Run a single experiment and return detailed trajectory."""
    random.seed(seed)
    np.random.seed(seed)

    engine = build_rsi_system(
        max_depth=5,
        budget_ops=100_000,
        budget_seconds=60.0,
        expansion_interval=expansion_interval,
        use_library_learning=True,
        library_min_depth=2,
        library_min_freq=2,
        library_max_additions=3,
        similarity_threshold=0.85,
    )

    # Record base op names (to distinguish from generated ops)
    base_ops = {op.name for op in engine.vocab.all_ops()}

    trajectory = []
    for gen in range(generations):
        record = engine.step(pop_size)

        # Count generated ops in current vocab
        current_ops = {op.name for op in engine.vocab.all_ops()}
        generated_ops = current_ops - base_ops
        poly_ops = {op.name for op in engine.vocab.all_ops()
                    if isinstance(op, PolymorphicOp)}

        # Ratio of elites using generated ops
        elites = list(engine.archive._grid.values())
        elites_using_generated = 0
        elites_using_self_encode = 0
        elites_using_poly = 0
        for e in elites:
            used = set()
            _collect_ops(e.tree, used)
            if used & generated_ops:
                elites_using_generated += 1
            if "self_encode" in used:
                elites_using_self_encode += 1
            if used & poly_ops:
                elites_using_poly += 1

        n_elites = len(elites)

        # "2nd-generation ops" -- detect ops generated from other generated ops
        # (e.g., lib_X containing lib_Y = evidence of recursive self-improvement)
        second_gen_ops = 0
        for op_name in generated_ops:
            if op_name.startswith("lib_"):
                # Library-extracted op: if its implementation uses
                # other generated ops, it's a 2nd-generation op
                pass  # analyzed separately below

        snapshot = {
            "gen": gen + 1,
            "best_fitness": record["archive_best"],
            "coverage": record["archive_coverage"],
            "vocab_size": record["vocab_size"],
            "generated_ops": len(generated_ops),
            "poly_ops": len(poly_ops),
            "grammar_rules": record["grammar_rules"],
            "meta_expansions": record["meta_expansions"],
            "n_elites": n_elites,
            "elites_using_generated": elites_using_generated,
            "elites_using_self_encode": elites_using_self_encode,
            "elites_using_poly": elites_using_poly,
            "utilization_rate": elites_using_generated / max(n_elites, 1),
        }
        trajectory.append(snapshot)

    # Final analysis: measure recursion depth
    # If generated ops appear in trees of other generated ops, it's 2nd-order recursion
    recursion_evidence = analyze_recursion_depth(engine, base_ops)

    return {
        "label": label,
        "seed": seed,
        "trajectory": trajectory,
        "final": trajectory[-1],
        "recursion": recursion_evidence,
    }


def analyze_recursion_depth(engine, base_ops):
    """
    Analyze the depth of recursive self-improvement.

    Depth 0: Only base ops used
    Depth 1: Meta-grammar generates new ops from base ops (1st generation)
    Depth 2: Elites use 1st-gen ops -> higher fitness
    Depth 3: Library learning extracts new patterns from high-fitness elites (2nd generation)
    Depth 4: 2nd-gen ops used in even better elites -> recursion complete
    """
    current_ops = {op.name for op in engine.vocab.all_ops()}
    generated_ops = current_ops - base_ops

    # Distinguish library-extracted ops (lib_*) from meta-grammar ops
    lib_ops = {n for n in generated_ops if n.startswith("lib_")}
    meta_ops = {n for n in generated_ops if not n.startswith("lib_") and not n.startswith("poly_")}
    poly_ops_set = {n for n in generated_ops if n.startswith("poly_")}

    # Depth 1: Were new ops generated?
    depth_1 = len(generated_ops) > 0

    # Depth 2: Do elites use generated ops?
    elites = list(engine.archive._grid.values())
    elites_with_gen = []
    for e in elites:
        used = set()
        _collect_ops(e.tree, used)
        gen_used = used & generated_ops
        if gen_used:
            elites_with_gen.append((e.grounded_fitness, gen_used))

    depth_2 = len(elites_with_gen) > 0

    # Depth 3: Did library learning extract patterns containing generated ops?
    # lib_* op names are fingerprints so we can't check directly -> indirect check:
    # If lib_* ops are used alongside other generated ops in elite trees, that's evidence
    depth_3_evidence = []
    for e in elites:
        used = set()
        _collect_ops(e.tree, used)
        lib_used = used & lib_ops
        other_gen_used = used & (meta_ops | poly_ops_set)
        if lib_used and other_gen_used:
            depth_3_evidence.append({
                "fitness": e.grounded_fitness,
                "lib_ops": lib_used,
                "meta_ops": other_gen_used,
            })

    depth_3 = len(depth_3_evidence) > 0

    # Depth 4: Do elites using 2nd-gen ops have higher fitness than those using only 1st-gen?
    fitness_with_lib = [f for f, _ in elites_with_gen if any(
        n.startswith("lib_") for n in _)] if elites_with_gen else []
    fitness_without_lib = [e.grounded_fitness for e in elites
                          if not any(n.startswith("lib_")
                          for n in _get_tree_ops(e.tree) & generated_ops)]

    depth_4 = False
    if fitness_with_lib and fitness_without_lib:
        avg_with = sum(fitness_with_lib) / len(fitness_with_lib)
        avg_without = sum(fitness_without_lib) / len(fitness_without_lib)
        depth_4 = avg_with > avg_without

    return {
        "depth_1_new_ops": depth_1,
        "depth_1_count": len(generated_ops),
        "depth_2_elites_use_ops": depth_2,
        "depth_2_count": len(elites_with_gen),
        "depth_3_compound_ops": depth_3,
        "depth_3_evidence_count": len(depth_3_evidence),
        "depth_4_recursive_benefit": depth_4,
        "lib_ops": len(lib_ops),
        "meta_ops": len(meta_ops),
        "poly_ops": len(poly_ops_set),
        "max_depth": (4 if depth_4 else 3 if depth_3 else 2 if depth_2
                      else 1 if depth_1 else 0),
    }


def _get_tree_ops(tree):
    ops = set()
    _collect_ops(tree, ops)
    return ops


def print_trajectory_summary(results, label, milestones=[50, 100, 200, 300, 500]):
    """Print trajectory summary."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"{'Gen':>5} {'Best':>8} {'Cov':>6} {'Vocab':>6} {'GenOps':>7} "
          f"{'Poly':>5} {'Util%':>6} {'Rules':>6}")
    print(f"{'-'*55}")

    for r in results:
        for t in r["trajectory"]:
            if t["gen"] in milestones or t["gen"] == 1:
                print(f"{t['gen']:5d} {t['best_fitness']:8.4f} {t['coverage']:6.4f} "
                      f"{t['vocab_size']:6d} {t['generated_ops']:7d} "
                      f"{t['poly_ops']:5d} {t['utilization_rate']*100:5.1f}% "
                      f"{t['grammar_rules']:6d}")
        print()


def main():
    print("=" * 80)
    print("  RSI (Recursive Self-Improvement) Verification Test")
    print("=" * 80)

    SEEDS = [42, 123, 456, 789, 1024]
    GENERATIONS = 500
    POP_SIZE = 20

    # ===================================================================
    # Condition A: FROZEN (meta-grammar disabled)
    # ===================================================================
    print("\n>> Condition A: FROZEN (no meta-grammar expansion)")
    frozen_results = []
    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        r = run_experiment(seed, GENERATIONS, POP_SIZE,
                          expansion_interval=999999, label=f"FROZEN_s{seed}")
        elapsed = time.time() - t0
        print(f"  Seed {seed} ({i+1}/5): best={r['final']['best_fitness']:.4f} "
              f"vocab={r['final']['vocab_size']} gen_ops={r['final']['generated_ops']} "
              f"time={elapsed:.1f}s")
        frozen_results.append(r)

    # ===================================================================
    # Condition B: SELF-MODIFY (meta-grammar enabled)
    # ===================================================================
    print("\n>> Condition B: SELF-MODIFY (meta-grammar expansion every 5 gen)")
    modify_results = []
    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        r = run_experiment(seed, GENERATIONS, POP_SIZE,
                          expansion_interval=5, label=f"MODIFY_s{seed}")
        elapsed = time.time() - t0
        print(f"  Seed {seed} ({i+1}/5): best={r['final']['best_fitness']:.4f} "
              f"vocab={r['final']['vocab_size']} gen_ops={r['final']['generated_ops']} "
              f"time={elapsed:.1f}s")
        modify_results.append(r)

    # ===================================================================
    # Comparative Analysis
    # ===================================================================
    print("\n" + "=" * 80)
    print("  Analysis Results")
    print("=" * 80)

    # 1. Fitness comparison
    frozen_bests = [r["final"]["best_fitness"] for r in frozen_results]
    modify_bests = [r["final"]["best_fitness"] for r in modify_results]

    print(f"\n  1. Final Fitness Comparison")
    print(f"     FROZEN:      {np.mean(frozen_bests):.4f} +/- {np.std(frozen_bests):.4f}")
    print(f"     SELF-MODIFY: {np.mean(modify_bests):.4f} +/- {np.std(modify_bests):.4f}")
    improvement = np.mean(modify_bests) - np.mean(frozen_bests)
    print(f"     Delta:       {improvement:+.4f}")
    if improvement > 0.01:
        print(f"     Verdict: Self-modification improves fitness")
    elif improvement > -0.01:
        print(f"     Verdict: No difference (self-modification has no effect on fitness)")
    else:
        print(f"     Verdict: Self-modification degrades performance")

    # 2. Coverage comparison
    frozen_covs = [r["final"]["coverage"] for r in frozen_results]
    modify_covs = [r["final"]["coverage"] for r in modify_results]

    print(f"\n  2. Coverage Comparison")
    print(f"     FROZEN:      {np.mean(frozen_covs):.4f} +/- {np.std(frozen_covs):.4f}")
    print(f"     SELF-MODIFY: {np.mean(modify_covs):.4f} +/- {np.std(modify_covs):.4f}")

    # 3. Vocab growth
    print(f"\n  3. Vocabulary Growth")
    for r in modify_results:
        traj = r["trajectory"]
        print(f"     Seed {r['seed']}: {traj[0]['vocab_size']} -> {traj[-1]['vocab_size']} "
              f"(+{traj[-1]['generated_ops']} generated, "
              f"{traj[-1]['poly_ops']} poly)")

    # 4. Generated op utilization trajectory
    print(f"\n  4. Generated Op Utilization Trajectory")
    print(f"     {'Gen':>5}  ", end="")
    for seed in SEEDS:
        print(f"  s{seed:>4}", end="")
    print()

    for gen_idx in [49, 99, 199, 299, 499]:  # gen 50, 100, 200, 300, 500
        print(f"     {gen_idx+1:5d}  ", end="")
        for r in modify_results:
            t = r["trajectory"][gen_idx]
            print(f"  {t['utilization_rate']*100:5.1f}%", end="")
        print()

    # 5. self_encode utilization
    print(f"\n  5. self_encode Utilization")
    for r in modify_results:
        t = r["trajectory"][-1]
        rate = t["elites_using_self_encode"] / max(t["n_elites"], 1) * 100
        print(f"     Seed {r['seed']}: {t['elites_using_self_encode']}/{t['n_elites']} "
              f"elites ({rate:.1f}%)")

    # 6. PolymorphicOp utilization
    print(f"\n  6. PolymorphicOp Utilization")
    for r in modify_results:
        t = r["trajectory"][-1]
        rate = t["elites_using_poly"] / max(t["n_elites"], 1) * 100
        print(f"     Seed {r['seed']}: {t['elites_using_poly']}/{t['n_elites']} "
              f"elites ({rate:.1f}%)")

    # 7. Recursion depth analysis
    print(f"\n  7. Recursion Depth Analysis")
    print(f"     Depth 0: Only base ops used")
    print(f"     Depth 1: Meta-grammar generates new ops")
    print(f"     Depth 2: Elites use generated ops -> higher fitness")
    print(f"     Depth 3: Library learning extracts patterns containing generated ops")
    print(f"     Depth 4: 2nd-gen ops provide fitness advantage")
    print()
    for r in modify_results:
        rec = r["recursion"]
        print(f"     Seed {r['seed']}: max depth = {rec['max_depth']}")
        print(f"       D1 (new op gen):        {'Y' if rec['depth_1_new_ops'] else 'N'} "
              f"({rec['depth_1_count']} ops)")
        print(f"       D2 (elite usage):       {'Y' if rec['depth_2_elites_use_ops'] else 'N'} "
              f"({rec['depth_2_count']} elites)")
        print(f"       D3 (compound extract):  {'Y' if rec['depth_3_compound_ops'] else 'N'} "
              f"({rec['depth_3_evidence_count']} evidence)")
        print(f"       D4 (recursive benefit): {'Y' if rec['depth_4_recursive_benefit'] else 'N'}")
        print(f"       Composition: lib={rec['lib_ops']} meta={rec['meta_ops']} poly={rec['poly_ops']}")
        print()

    # 8. Convergence speed comparison
    print(f"\n  8. Convergence Speed Comparison")
    print(f"     {'':>12} {'Gen->0.90':>10} {'Gen->0.95':>10} {'Gen->0.99':>10}")
    for label, results in [("FROZEN", frozen_results), ("SELF-MODIFY", modify_results)]:
        thresholds = {0.90: [], 0.95: [], 0.99: []}
        for r in results:
            for thresh in thresholds:
                found = None
                for t in r["trajectory"]:
                    if t["best_fitness"] >= thresh:
                        found = t["gen"]
                        break
                thresholds[thresh].append(found if found else GENERATIONS + 1)

        avgs = {k: np.mean([v for v in vals if v <= GENERATIONS])
                if any(v <= GENERATIONS for v in vals) else float('inf')
                for k, vals in thresholds.items()}

        print(f"     {label:<12} ", end="")
        for thresh in [0.90, 0.95, 0.99]:
            if avgs[thresh] == float('inf'):
                print(f"  {'never':>8}", end="")
            else:
                print(f"  {avgs[thresh]:8.1f}", end="")
        print()

    # ===================================================================
    # Final Verdict
    # ===================================================================
    print("\n" + "=" * 80)
    print("  Final Verdict")
    print("=" * 80)

    avg_depth = np.mean([r["recursion"]["max_depth"] for r in modify_results])
    any_depth_4 = any(r["recursion"]["depth_4_recursive_benefit"] for r in modify_results)
    any_depth_3 = any(r["recursion"]["depth_3_compound_ops"] for r in modify_results)
    avg_util = np.mean([r["final"]["utilization_rate"] for r in modify_results])

    print(f"\n  Average recursion depth: {avg_depth:.1f}")
    print(f"  Average generated op utilization: {avg_util*100:.1f}%")
    print(f"  Fitness improvement (vs FROZEN): {improvement:+.4f}")

    if any_depth_4 and improvement > 0.01:
        verdict = "GENUINE_RECURSIVE_SELF_IMPROVEMENT"
        desc = "Recursive self-improvement confirmed: the system uses self-generated tools\n" \
               "    to create better tools, which in turn improve performance \u2014 recursive loop active."
    elif any_depth_3 and improvement > 0.01:
        verdict = "PARTIAL_RSI_DEPTH_3"
        desc = "Partial self-improvement: the system extracts new patterns using generated ops,\n" \
               "    but whether 2nd-gen ops provide clear fitness advantage is uncertain."
    elif any_depth_3 and improvement > -0.01:
        verdict = "SELF_MODIFICATION_WITHOUT_IMPROVEMENT"
        desc = "Self-modification without improvement: the system modifies its structure\n" \
               "    but changes do not lead to performance gains."
    elif avg_depth >= 2 and improvement > 0.01:
        verdict = "ONE_SHOT_IMPROVEMENT"
        desc = "One-shot improvement: generated ops are used and fitness improves,\n" \
               "    but the recursive loop (generate->use->re-generate) is not confirmed."
    else:
        verdict = "NO_RSI"
        desc = "Recursive self-improvement not confirmed."

    print(f"\n  VERDICT: {verdict}")
    print(f"  Description: {desc}")

    # Save raw data
    print(f"\n  Raw data:")
    print(f"  FROZEN  seeds: {[r['final']['best_fitness'] for r in frozen_results]}")
    print(f"  MODIFY  seeds: {[r['final']['best_fitness'] for r in modify_results]}")
    print(f"  MODIFY depths: {[r['recursion']['max_depth'] for r in modify_results]}")


if __name__ == "__main__":
    main()
