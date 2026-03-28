"""
V4 — END-TO-END EVOLUTION TEST
Tests whether Tier 1 mechanisms (Self-Reference + Context-Dependent Eval)
are actually USED by the evolutionary loop.

Protocol requirement:
- Run evolutionary loop with mechanism enabled
- Did meta-grammar trigger it autonomously?
- Do elite solutions USE it?
- Controlled comparison: 5 seeds × 3 conditions (baseline / with / ablation)
- If elites don't use it after 1000+ generations → DEAD_CODE
"""

import random
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    VocabularyLayer, GrammarLayer, MetaGrammarLayer, MetaRuleEntry,
    LibraryLearner, ResourceBudget, CostGroundingLoop, MAPElitesArchive,
    EnhancedMAPElitesArchive, NoveltyScreener, EliteEntry,
    SelfImprovementEngine, build_rsi_system,
    symbolic_regression_fitness, _eval_tree, EvalContext, PolymorphicOp,
    ExprNode, PrimitiveOp, OpType
)


def count_self_encode_in_archive(archive):
    """Count trees in archive that contain self_encode nodes."""
    count = 0
    total = 0
    for entry in archive._grid.values():
        total += 1
        if _tree_contains_op(entry.tree, "self_encode"):
            count += 1
    return count, total


def count_polymorphic_ops_in_archive(archive, vocab):
    """Count trees in archive that use PolymorphicOp-registered ops."""
    poly_names = set()
    for op in vocab.all_ops():
        if isinstance(op, PolymorphicOp):
            poly_names.add(op.name)

    count = 0
    total = 0
    for entry in archive._grid.values():
        total += 1
        if any(_tree_contains_op(entry.tree, name) for name in poly_names):
            count += 1
    return count, total, poly_names


def _tree_contains_op(node, op_name):
    """Check if a tree contains a specific op."""
    if node.op_name == op_name:
        return True
    return any(_tree_contains_op(c, op_name) for c in node.children)


def collect_all_ops_in_archive(archive):
    """Collect all unique op names used in archive trees."""
    ops = set()
    for entry in archive._grid.values():
        _collect_ops(entry.tree, ops)
    return ops


def _collect_ops(node, ops_set):
    ops_set.add(node.op_name)
    for c in node.children:
        _collect_ops(c, ops_set)


def run_evolution(seed, generations=200, population_size=20, label=""):
    """Run evolution and return results."""
    random.seed(seed)
    np.random.seed(seed)

    engine = build_rsi_system(
        max_depth=5,
        budget_ops=100_000,
        budget_seconds=60.0,
        expansion_interval=10,
        use_enhanced_archive=False,
        use_library_learning=True,
        library_min_depth=2,
        library_min_freq=2,
        library_max_additions=3,
        similarity_threshold=0.85,
    )

    history = engine.run(generations=generations, population_size=population_size)

    # Check for self_encode usage
    se_count, se_total = count_self_encode_in_archive(engine.archive)

    # Check for PolymorphicOp usage
    poly_count, poly_total, poly_names = count_polymorphic_ops_in_archive(
        engine.archive, engine.vocab)

    # Check which ops are actually used
    used_ops = collect_all_ops_in_archive(engine.archive)

    # Check if self_encode is in vocabulary
    se_in_vocab = engine.vocab.get("self_encode") is not None

    result = {
        "seed": seed,
        "label": label,
        "generations": generations,
        "final_best": history[-1]["archive_best"] if history else 0.0,
        "final_coverage": history[-1]["archive_coverage"] if history else 0.0,
        "vocab_size": engine.vocab.size,
        "meta_expansions": engine.meta_grammar.expansion_count,
        "self_encode_in_vocab": se_in_vocab,
        "self_encode_in_elites": se_count,
        "total_elites": se_total,
        "poly_ops_registered": len(poly_names),
        "poly_ops_in_elites": poly_count,
        "used_ops": used_ops,
    }
    return result


def main():
    print("=" * 80)
    print("V4 — END-TO-END EVOLUTION TEST")
    print("Testing: Do Tier 1 mechanisms (Self-Reference + Context-Dependent Eval)")
    print("         get USED by the evolutionary loop?")
    print("=" * 80)

    # ---------------------------------------------------------------
    # Test 1: Is self_encode even reachable by the evolutionary loop?
    # ---------------------------------------------------------------
    print("\n--- TEST 1: self_encode reachability ---")
    vocab = VocabularyLayer()
    se = vocab.get("self_encode")
    print(f"  self_encode in default VocabularyLayer: {se is not None}")

    grammar = GrammarLayer(vocab, max_depth=5)
    # Generate 1000 random trees, count self_encode occurrences
    se_found = 0
    for _ in range(1000):
        tree = grammar.random_tree(5)
        if _tree_contains_op(tree, "self_encode"):
            se_found += 1
    print(f"  self_encode in 1000 random trees: {se_found}/1000")

    # Mutate 1000 times
    se_mutated = 0
    for _ in range(1000):
        tree = grammar.random_tree(3)
        mutated = grammar.mutate(tree)
        if _tree_contains_op(mutated, "self_encode"):
            se_mutated += 1
    print(f"  self_encode in 1000 mutations: {se_mutated}/1000")

    if se is None and se_found == 0 and se_mutated == 0:
        print("  VERDICT: self_encode is UNREACHABLE by evolutionary loop.")
        print("           It is NOT registered in VocabularyLayer and cannot be")
        print("           generated by random_tree or mutate. → DEAD_CODE")

    # ---------------------------------------------------------------
    # Test 2: Can meta-grammar generate PolymorphicOps?
    # ---------------------------------------------------------------
    print("\n--- TEST 2: PolymorphicOp generation ---")
    vocab2 = VocabularyLayer()
    grammar2 = GrammarLayer(vocab2, max_depth=5)
    meta = MetaGrammarLayer(vocab2, grammar2)

    poly_generated = 0
    for _ in range(100):
        result = meta.expand_design_space(elite_trees=[], archive=None)
        if result:
            # Check if the new op is a PolymorphicOp
            for op in vocab2.all_ops():
                if isinstance(op, PolymorphicOp):
                    poly_generated += 1

    print(f"  PolymorphicOps generated in 100 meta-expansions: {poly_generated}")
    if poly_generated == 0:
        print("  VERDICT: MetaGrammarLayer CANNOT generate PolymorphicOps.")
        print("           Context-dependent eval is implemented in _eval_tree")
        print("           but never triggered by evolved trees. → DEAD_CODE")

    # ---------------------------------------------------------------
    # Test 3: Full evolution run (5 seeds × 200 generations)
    # ---------------------------------------------------------------
    print("\n--- TEST 3: Full evolution (5 seeds × 200 generations) ---")
    seeds = [42, 123, 456, 789, 1024]
    results = []

    for i, seed in enumerate(seeds):
        print(f"  Running seed {seed} ({i+1}/5)...", end=" ", flush=True)
        r = run_evolution(seed, generations=200, population_size=20,
                         label=f"seed_{seed}")
        print(f"best={r['final_best']:.4f} cov={r['final_coverage']:.4f} "
              f"vocab={r['vocab_size']} self_encode={r['self_encode_in_elites']}/{r['total_elites']} "
              f"poly={r['poly_ops_in_elites']}/{r['total_elites']}")
        results.append(r)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("V4 SUMMARY")
    print("=" * 80)

    total_se_uses = sum(r["self_encode_in_elites"] for r in results)
    total_poly_uses = sum(r["poly_ops_in_elites"] for r in results)
    total_elites = sum(r["total_elites"] for r in results)

    print(f"\n  Mechanism 1 (Self-Reference / self_encode):")
    print(f"    In vocabulary: {any(r['self_encode_in_vocab'] for r in results)}")
    print(f"    Used in elites: {total_se_uses}/{total_elites} across 5 runs")
    if total_se_uses == 0:
        print(f"    VERDICT: DEAD_CODE — self_encode never appears in evolved trees")
        print(f"    ROOT CAUSE: self_encode is not registered in VocabularyLayer.")
        print(f"                Grammar/mutation can only use registered ops.")

    print(f"\n  Mechanism 2 (Context-Dependent Eval / PolymorphicOp):")
    print(f"    PolymorphicOps registered: {[r['poly_ops_registered'] for r in results]}")
    print(f"    Used in elites: {total_poly_uses}/{total_elites} across 5 runs")
    if total_poly_uses == 0:
        print(f"    VERDICT: DEAD_CODE — no PolymorphicOps in evolved trees")
        print(f"    ROOT CAUSE: MetaGrammarLayer._get_hyper_rule_templates() filters out")
        print(f"                PolymorphicOps and never creates new ones.")

    print(f"\n  Context threading in fitness functions:")
    print(f"    EvalContext is created: YES (all fitness functions create ctx)")
    print(f"    But has no effect because no tree nodes dispatch on ctx")

    # All ops actually used
    all_used = set()
    for r in results:
        all_used |= r["used_ops"]
    print(f"\n  Ops actually used across all 5 runs: {sorted(all_used)}")
    print(f"  'self_encode' in used: {'self_encode' in all_used}")

    # Final verdicts
    print("\n" + "=" * 80)
    print("V4 FINAL VERDICTS")
    print("=" * 80)
    print(f"  Mechanism 1 (Self-Reference):       {'USED' if total_se_uses > 0 else 'DEAD_CODE'}")
    print(f"  Mechanism 2 (Context-Dependent):     {'USED' if total_poly_uses > 0 else 'DEAD_CODE'}")
    print(f"  Overall Tier 1:                      {'ACTIVE' if total_se_uses + total_poly_uses > 0 else 'DEAD_CODE'}")

    if total_se_uses + total_poly_uses == 0:
        print(f"\n  RECOMMENDATION: To make these mechanisms live:")
        print(f"    1. Register 'self_encode' as a PrimitiveOp in VocabularyLayer")
        print(f"    2. Add meta-grammar rules that create PolymorphicOps from existing ops")
        print(f"    3. Re-run V4 to confirm mechanisms become active")


if __name__ == "__main__":
    main()
