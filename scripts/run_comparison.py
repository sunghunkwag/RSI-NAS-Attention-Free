import main
import logging

logging.getLogger().setLevel(logging.WARNING)

print("Running RSI Engine with New Mechanisms (Sub-tree Compression, Pruning, Parsimony, Error-guided)...")
engine = main.build_rsi_system(
    fitness_name="sine_approximation",
    max_depth=5,
    budget_ops=100000,
    budget_seconds=60.0,
    expansion_interval=5,
    use_enhanced_archive=True,
    use_library_learning=True
)

history = engine.run(generations=50, population_size=30)

print("\n=== Results ===")
print(f"Generation 1 Best Fitness: {history[0]['best_gen_fitness']:.4f}")
print(f"Generation 25 Best Fitness: {history[24]['best_gen_fitness']:.4f}")
print(f"Generation 50 Best Fitness: {history[-1]['archive_best']:.4f}")

final_vocab = [op.name for op in engine.vocab.all_ops()]
dynamic_ops = [op for op in final_vocab if op not in engine.vocab._default_op_names]

print(f"\nFinal Vocabulary Size: {len(final_vocab)} (Dynamically added & kept: {len(dynamic_ops)})")
if dynamic_ops:
    print("Kept Dynamic Operators:")
    for op in dynamic_ops:
        print(f" - {op}")

best_elite = max(engine.archive._grid.values(), key=lambda e: e.grounded_fitness)
print(f"\nBest Tree Size: {best_elite.tree.size()}")
print(f"Best Tree Depth: {best_elite.tree.depth()}")
