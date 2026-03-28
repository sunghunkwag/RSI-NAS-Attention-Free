import main
import logging
import sys

logging.getLogger().setLevel(logging.INFO)
engine = main.build_rsi_system(
    fitness_name="sine_approximation",
    max_depth=5,
    budget_ops=100000,
    budget_seconds=60.0,
    expansion_interval=5,
    use_enhanced_archive=True,
    use_library_learning=True
)

engine.run(generations=20, population_size=20)
print("\nFinal Dynamic operators:")
dynamic_ops = [op.name for op in engine.vocab.all_ops() if op.name not in engine.vocab._default_op_names]
for name in dynamic_ops:
    print(f" - {name}")
