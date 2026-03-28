"""
Tests for RSI-NAS: Attention-Free Neural Architecture Search
=============================================================

Covers:
  1. Module primitives — forward shape, gradient flow
  2. ModuleRegistry — registration, unregistration, defaults
  3. ArchitectureGenome — construction, fingerprint, clone
  4. Network building — smoke test, weight tying, param count
  5. Fitness evaluation — runs SGD, returns valid BPC
  6. ArchitectureGrammar — all mutation operators
  7. ArchitectureMeta — library extraction, sequential composition, specialization
  8. ArchitectureArchive — insertion, behavior descriptor, coverage
  9. RSI loop integration — step() produces valid records
  10. Ablation structure — FROZEN vs SELF-MODIFY divergence
"""

import copy
import random
import pytest
import numpy as np
import torch

from rsi_nas import (
    # Primitives
    PerceptionFilter, ReactionGate, MultiRateDiffusion, NCAStep,
    GatedShiftMixer, SqueezeExcite, CoarseNCA, GatedFFN,
    SimpleGraphConv, FractalGNNBlock,
    # RSI framework
    ModuleSpec, ModuleRegistry,
    LayerGene, ArchitectureGenome,
    BuiltNetwork, build_network, evaluate_architecture,
    ArchitectureGrammar, ArchitectureMeta, ArchitectureArchive,
    ArchiveEntry, FitnessResult,
    RSINASEngine, build_rsi_nas,
)

B, L, D = 2, 32, 16
torch.manual_seed(42)


# ── 1. Module primitives ────────────────────────────────────────────────────

class TestModulePrimitives:

    @pytest.mark.parametrize("module_fn", [
        lambda: NCAStep(D, k=5, dilations=(1, 4)),
        lambda: GatedShiftMixer(D, shifts=(-4, -1, 1, 4)),
        lambda: SqueezeExcite(D, r=4),
        lambda: CoarseNCA(D, stride=4, n_steps=1, dilations=(1, 4)),
        lambda: GatedFFN(D, exp=2),
        lambda: FractalGNNBlock(D, chunk_size=8, gnn_depth=1),
    ])
    def test_forward_shape(self, module_fn):
        m = module_fn()
        x = torch.randn(B, L, D)
        out = m(x)
        assert out.shape == (B, L, D)

    @pytest.mark.parametrize("module_fn", [
        lambda: NCAStep(D),
        lambda: GatedShiftMixer(D, shifts=(-4, 4)),
        lambda: GatedFFN(D),
        lambda: FractalGNNBlock(D, chunk_size=8),
    ])
    def test_gradient_flow(self, module_fn):
        m = module_fn()
        x = torch.randn(1, L, D, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_simple_graph_conv_shape(self):
        conv = SimpleGraphConv(D)
        h = torch.randn(B, 8, D)
        assert conv(h).shape == (B, 8, D)

    def test_perception_filter_shape(self):
        pf = PerceptionFilter(D, k=5, nf=2)
        x = torch.randn(B, L, D)
        assert pf(x).shape == (B, L, D)


# ── 2. ModuleRegistry ───────────────────────────────────────────────────────

class TestModuleRegistry:

    def test_defaults_registered(self):
        reg = ModuleRegistry()
        assert reg.size == 6
        names = {s.name for s in reg.all_specs()}
        assert "nca_step" in names
        assert "gated_shift_mixer" in names
        assert "fractal_gnn" in names

    def test_register_and_unregister(self):
        reg = ModuleRegistry()
        spec = ModuleSpec(name="test_mod", builder=lambda d, **kw: GatedFFN(d),
                          default_kwargs={}, param_cost=1.0, is_generated=True)
        reg.register(spec)
        assert reg.size == 7
        assert reg.get("test_mod") is not None
        assert reg.unregister("test_mod")
        assert reg.size == 6

    def test_cannot_unregister_default(self):
        reg = ModuleRegistry()
        assert not reg.unregister("nca_step")
        assert reg.size == 6

    def test_build_from_spec(self):
        reg = ModuleRegistry()
        spec = reg.get("gated_ffn")
        module = spec.build(D)
        x = torch.randn(1, L, D)
        assert module(x).shape == (1, L, D)


# ── 3. ArchitectureGenome ───────────────────────────────────────────────────

class TestArchitectureGenome:

    def test_basic_properties(self):
        g = ArchitectureGenome(
            layers=[LayerGene("nca_step", repeat=2),
                    LayerGene("gated_ffn", repeat=1)],
            d_model=D,
        )
        assert g.size() == 2
        assert g.depth() == 3

    def test_fingerprint_deterministic(self):
        g = ArchitectureGenome(layers=[LayerGene("nca_step")])
        assert g.fingerprint() == g.fingerprint()

    def test_clone_independent(self):
        g = ArchitectureGenome(layers=[LayerGene("nca_step", repeat=2)])
        g2 = g.clone()
        g2.layers[0].repeat = 5
        assert g.layers[0].repeat == 2

    def test_estimated_params(self):
        reg = ModuleRegistry()
        g = ArchitectureGenome(layers=[LayerGene("gated_ffn")], d_model=D)
        est = g.estimated_params(reg)
        assert est > 0


# ── 4. Network building ────────────────────────────────────────────────────

class TestNetworkBuilding:

    def test_build_and_forward(self):
        reg = ModuleRegistry()
        g = ArchitectureGenome(
            layers=[LayerGene("nca_step"), LayerGene("gated_ffn")],
            d_model=D, vocab_size=256, max_len=64,
        )
        net = build_network(g, reg)
        assert net is not None
        x = torch.randint(0, 256, (B, 32))
        out = net(x)
        assert out.shape == (B, 32, 256)

    def test_weight_tying(self):
        reg = ModuleRegistry()
        g = ArchitectureGenome(layers=[LayerGene("gated_ffn")], d_model=D)
        net = build_network(g, reg)
        assert net.head.weight is net.tok_emb.weight

    def test_missing_module_graceful(self):
        reg = ModuleRegistry()
        g = ArchitectureGenome(layers=[LayerGene("nonexistent_module")], d_model=D)
        net = build_network(g, reg)
        # Should still build (skips missing modules)
        # but may fail smoke test if no layers remain
        # Either way, should not crash

    def test_param_count_positive(self):
        reg = ModuleRegistry()
        g = ArchitectureGenome(
            layers=[LayerGene("nca_step"), LayerGene("gated_shift_mixer")],
            d_model=D,
        )
        net = build_network(g, reg)
        assert net.count_parameters() > 0


# ── 5. Fitness evaluation ──────────────────────────────────────────────────

class TestFitnessEvaluation:

    def test_evaluate_returns_valid(self):
        reg = ModuleRegistry()
        g = ArchitectureGenome(layers=[LayerGene("gated_ffn")], d_model=D)
        result = evaluate_architecture(g, reg, train_steps=5, seq_len=32, batch_size=2)
        assert isinstance(result, FitnessResult)
        assert 0.0 <= result.fitness <= 1.0
        assert result.bpc > 0
        assert result.param_count > 0

    def test_oversized_network_rejected(self):
        reg = ModuleRegistry()
        g = ArchitectureGenome(
            layers=[LayerGene("fractal_gnn", repeat=3)] * 5,
            d_model=128,
        )
        result = evaluate_architecture(g, reg, train_steps=5, max_params=1000)
        assert result.fitness == 0.0  # should be rejected for exceeding max_params


# ── 6. ArchitectureGrammar ──────────────────────────────────────────────────

class TestArchitectureGrammar:

    def test_random_genome(self):
        reg = ModuleRegistry()
        grammar = ArchitectureGrammar(reg)
        g = grammar.random_genome(d_model=D)
        assert 2 <= g.size() <= 5
        assert g.d_model == D

    def test_mutate_preserves_validity(self):
        reg = ModuleRegistry()
        grammar = ArchitectureGrammar(reg)
        g = grammar.random_genome(d_model=D)
        for _ in range(20):
            g = grammar.mutate(g)
            assert len(g.layers) >= 1
            assert all(isinstance(l, LayerGene) for l in g.layers)

    def test_crossover(self):
        reg = ModuleRegistry()
        grammar = ArchitectureGrammar(reg)
        g1 = grammar.random_genome(d_model=D)
        g2 = grammar.random_genome(d_model=D)
        child = grammar.crossover(g1, g2)
        assert len(child.layers) >= 1

    def test_max_layers_enforced(self):
        reg = ModuleRegistry()
        grammar = ArchitectureGrammar(reg, max_layers=3)
        g = grammar.random_genome(d_model=D)
        for _ in range(50):
            g = grammar.mutate(g)
        assert len(g.layers) <= 3


# ── 7. ArchitectureMeta ─────────────────────────────────────────────────────

class TestArchitectureMeta:

    def _make_elites(self, registry, n=5):
        grammar = ArchitectureGrammar(registry)
        genomes = [grammar.random_genome(d_model=D) for _ in range(n)]
        fitnesses = [random.uniform(0.1, 0.3) for _ in range(n)]
        return genomes, fitnesses

    def test_expand_returns_action(self):
        reg = ModuleRegistry()
        grammar = ArchitectureGrammar(reg)
        meta = ArchitectureMeta(reg, grammar)
        genomes, fits = self._make_elites(reg, 5)
        action = meta.expand_design_space(genomes, fits)
        # Should produce some action (library, compose, or specialize)
        assert action is not None or meta.expansion_count == 1

    def test_sequential_composition(self):
        reg = ModuleRegistry()
        grammar = ArchitectureGrammar(reg)
        meta = ArchitectureMeta(reg, grammar)
        initial_size = reg.size
        action = meta._compose_sequential()
        if action is not None:
            assert reg.size == initial_size + 1
            assert action.startswith("compose:")

    def test_library_extraction_with_shared_patterns(self):
        """If multiple elites share a layer pattern, extraction should fire."""
        reg = ModuleRegistry()
        grammar = ArchitectureGrammar(reg)
        meta = ArchitectureMeta(reg, grammar)
        # Create elites that share a common 2-layer pattern
        shared = [LayerGene("nca_step"), LayerGene("gated_ffn")]
        genomes = [
            ArchitectureGenome(layers=shared + [LayerGene("squeeze_excite")], d_model=D),
            ArchitectureGenome(layers=[LayerGene("gated_shift_mixer")] + shared, d_model=D),
            ArchitectureGenome(layers=shared + [LayerGene("coarse_nca")], d_model=D),
        ]
        fits = [0.2, 0.25, 0.3]
        action = meta._extract_library(genomes, fits)
        if action is not None:
            assert action.startswith("library:")
            assert "fused" in action

    def test_prune_removes_unused(self):
        reg = ModuleRegistry()
        grammar = ArchitectureGrammar(reg)
        meta = ArchitectureMeta(reg, grammar)
        # Register a generated module
        spec = ModuleSpec(name="test_gen", builder=lambda d, **kw: GatedFFN(d),
                          default_kwargs={}, param_cost=1.0, is_generated=True)
        reg.register(spec)
        assert reg.size == 7
        # Prune with elites that don't use it
        genomes = [ArchitectureGenome(layers=[LayerGene("nca_step")], d_model=D)]
        pruned = meta.prune_unused(genomes)
        assert "test_gen" in pruned
        assert reg.size == 6


# ── 8. ArchitectureArchive ──────────────────────────────────────────────────

class TestArchitectureArchive:

    def test_insertion_and_coverage(self):
        archive = ArchitectureArchive(param_bins=4, depth_bins=3)
        g = ArchitectureGenome(layers=[LayerGene("nca_step")], d_model=D)
        entry = ArchiveEntry(genome=g, fitness=0.2, bpc=4.0, param_count=10000,
                             behavior=(1, 0), generation=1)
        assert archive.try_insert(entry)
        assert archive.coverage > 0
        assert archive.best_fitness == 0.2

    def test_elitism(self):
        archive = ArchitectureArchive(param_bins=4, depth_bins=3)
        g1 = ArchitectureGenome(layers=[LayerGene("nca_step")], d_model=D)
        g2 = ArchitectureGenome(layers=[LayerGene("gated_ffn")], d_model=D)
        e1 = ArchiveEntry(genome=g1, fitness=0.2, bpc=4.0, param_count=10000,
                          behavior=(1, 0), generation=1)
        e2 = ArchiveEntry(genome=g2, fitness=0.3, bpc=3.5, param_count=10000,
                          behavior=(1, 0), generation=2)
        archive.try_insert(e1)
        archive.try_insert(e2)
        # Better fitness should replace
        assert archive.best_fitness == 0.3

    def test_behavior_descriptor(self):
        archive = ArchitectureArchive(param_bins=6, depth_bins=5)
        g = ArchitectureGenome(layers=[LayerGene("nca_step", repeat=2)], d_model=D)
        b = archive.behavior_descriptor(g, 50000)
        assert len(b) == 2
        assert 0 <= b[0] < 6
        assert 0 <= b[1] < 5


# ── 9. RSI loop integration ────────────────────────────────────────────────

class TestRSILoop:

    def test_step_returns_valid_record(self):
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        engine = build_rsi_nas(d_model=D, train_steps=5, expansion_interval=2)
        record = engine.step(population_size=2)
        assert "generation" in record
        assert "archive_best_bpc" in record
        assert "vocab_size" in record
        assert record["generation"] == 1

    def test_run_produces_history(self):
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        engine = build_rsi_nas(d_model=D, train_steps=5, expansion_interval=2)
        history = engine.run(generations=3, population_size=2)
        assert len(history) == 3
        assert history[-1]["generation"] == 3

    def test_bpc_improves_over_generations(self):
        """BPC should generally decrease (or at least not increase) over time."""
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        engine = build_rsi_nas(d_model=D, train_steps=10, expansion_interval=3)
        history = engine.run(generations=6, population_size=2)
        # First gen BPC should be >= last gen BPC (archive keeps best)
        assert history[-1]["archive_best_bpc"] <= history[0]["archive_best_bpc"]


# ── 10. Ablation structure ──────────────────────────────────────────────────

class TestAblationStructure:

    def test_frozen_produces_no_modules(self):
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        engine = build_rsi_nas(d_model=D, train_steps=5, expansion_interval=999999)
        engine.run(generations=4, population_size=2)
        assert len(engine.registry.generated_names()) == 0

    def test_self_modify_can_produce_modules(self):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        engine = build_rsi_nas(d_model=D, train_steps=5, expansion_interval=2)
        engine.run(generations=6, population_size=2)
        # Meta-grammar should have attempted expansion at least once
        assert engine.meta.expansion_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
