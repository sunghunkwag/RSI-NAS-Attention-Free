"""
Tests for omega_backend.py — Omega VM Backend Integration
==========================================================

Tests cover:
1. Instruction & ProgramGenome basics
2. VirtualMachine execution (arithmetic, control flow, memory, error cases)
3. ControlFlowGraph (trace→CFG, SCC detection, canonical hash)
4. ExprNodeCompiler (leaf compilation, unary/binary ops, full trees)
5. VMFitness (evaluate_tree, structural_score)
6. VM fitness registry functions
7. Integration: build_rsi_system with use_vm_backend=True
"""

import math
import unittest

from main import (
    ExprNode,
    VocabularyLayer,
    PrimitiveOp,
    EvalContext,
    OpType,
    build_rsi_system,
    _get_vm_fitness_registry,
)
from omega_backend import (
    Instruction,
    ProgramGenome,
    ExecutionState,
    VirtualMachine,
    ControlFlowGraph,
    ExprNodeCompiler,
    VMFitness,
    VM_FITNESS_REGISTRY,
    vm_symbolic_regression_fitness,
    vm_sine_approximation_fitness,
    vm_absolute_value_fitness,
    OPS,
    CONTROL_OPS,
    MEMORY_OPS,
)


# ---------------------------------------------------------------------------
# 1. Instruction & ProgramGenome
# ---------------------------------------------------------------------------

class TestInstruction(unittest.TestCase):
    def test_clone(self):
        inst = Instruction("ADD", 1, 2, 3)
        c = inst.clone()
        self.assertEqual(c.op, "ADD")
        self.assertEqual(c.a, 1)
        self.assertIsNot(inst, c)

    def test_to_tuple(self):
        inst = Instruction("SET", 5, 0, 2)
        t = inst.to_tuple()
        self.assertEqual(t, ("SET", 5, 0, 2))

    def test_default_values(self):
        inst = Instruction("HALT")
        self.assertEqual(inst.a, 0)
        self.assertEqual(inst.b, 0)
        self.assertEqual(inst.c, 0)


class TestProgramGenome(unittest.TestCase):
    def test_clone(self):
        g = ProgramGenome(gid="test", instructions=[Instruction("SET", 1, 0, 0), Instruction("HALT")])
        c = g.clone()
        self.assertEqual(c.gid, "test")
        self.assertEqual(len(c.instructions), 2)
        self.assertIsNot(g.instructions[0], c.instructions[0])

    def test_code_hash_deterministic(self):
        g = ProgramGenome(gid="test", instructions=[Instruction("ADD", 1, 2, 3)])
        h1 = g.code_hash()
        h2 = g.code_hash()
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_different_programs_different_hash(self):
        g1 = ProgramGenome(gid="a", instructions=[Instruction("ADD", 1, 2, 3)])
        g2 = ProgramGenome(gid="b", instructions=[Instruction("SUB", 1, 2, 3)])
        self.assertNotEqual(g1.code_hash(), g2.code_hash())


# ---------------------------------------------------------------------------
# 2. VirtualMachine
# ---------------------------------------------------------------------------

class TestVirtualMachine(unittest.TestCase):
    def setUp(self):
        self.vm = VirtualMachine(max_steps=200, memory_size=64)

    def test_set_and_halt(self):
        """SET R0=42, then HALT."""
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("SET", 42, 0, 0),
            Instruction("HALT"),
        ])
        st = self.vm.execute(genome, [])
        self.assertTrue(st.halted)
        self.assertTrue(st.halted_cleanly)
        self.assertEqual(st.regs[0], 42.0)

    def test_arithmetic(self):
        """SET R0=10, SET R1=3, ADD R0+R1->R2, SUB R0-R1->R3."""
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("SET", 10, 0, 0),
            Instruction("SET", 3, 0, 1),
            Instruction("ADD", 0, 1, 2),
            Instruction("SUB", 0, 1, 3),
            Instruction("HALT"),
        ])
        st = self.vm.execute(genome, [])
        self.assertEqual(st.regs[2], 13.0)
        self.assertEqual(st.regs[3], 7.0)

    def test_mul_div(self):
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("SET", 6, 0, 0),
            Instruction("SET", 3, 0, 1),
            Instruction("MUL", 0, 1, 2),
            Instruction("DIV", 0, 1, 3),
            Instruction("HALT"),
        ])
        st = self.vm.execute(genome, [])
        self.assertEqual(st.regs[2], 18.0)
        self.assertAlmostEqual(st.regs[3], 2.0)

    def test_div_by_zero(self):
        """DIV by zero should yield 0.0, not crash."""
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("SET", 10, 0, 0),
            Instruction("SET", 0, 0, 1),
            Instruction("DIV", 0, 1, 2),
            Instruction("HALT"),
        ])
        st = self.vm.execute(genome, [])
        self.assertEqual(st.regs[2], 0.0)
        self.assertTrue(st.halted_cleanly)

    def test_memory_load_store(self):
        """Store to memory, load back."""
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("SET", 99, 0, 0),
            Instruction("SET", 5, 0, 1),   # address 5
            Instruction("STORE", 1, 0, 0),  # mem[5] = R0 = 99
            Instruction("SET", 0, 0, 0),    # clear R0
            Instruction("LOAD", 1, 0, 2),   # R2 = mem[5]
            Instruction("HALT"),
        ])
        st = self.vm.execute(genome, [])
        self.assertEqual(st.regs[2], 99.0)
        self.assertGreater(st.memory_writes, 0)
        self.assertGreater(st.memory_reads, 0)

    def test_jump(self):
        """JMP skips an instruction."""
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("SET", 1, 0, 0),
            Instruction("JMP", 2, 0, 0),   # skip next
            Instruction("SET", 99, 0, 0),   # should be skipped
            Instruction("HALT"),
        ])
        st = self.vm.execute(genome, [])
        self.assertEqual(st.regs[0], 1.0)  # 99 was skipped

    def test_conditional_jz(self):
        """JZ: jump if R0 is zero."""
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("SET", 0, 0, 0),    # R0 = 0
            Instruction("JZ", 0, 2, 0),     # R0==0 → skip next
            Instruction("SET", 99, 0, 1),    # should be skipped
            Instruction("SET", 42, 0, 1),    # should execute
            Instruction("HALT"),
        ])
        st = self.vm.execute(genome, [])
        self.assertEqual(st.regs[1], 42.0)

    def test_call_ret(self):
        """CALL pushes return address, RET pops it."""
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("CALL", 2, 0, 0),   # call to pc+2=2
            Instruction("HALT"),              # return here after RET
            Instruction("SET", 77, 0, 0),    # subroutine body
            Instruction("RET"),               # return to pc=1
        ])
        st = self.vm.execute(genome, [])
        self.assertEqual(st.regs[0], 77.0)
        self.assertTrue(st.halted_cleanly)

    def test_stack_overflow(self):
        """Recursive CALL without RET should trigger stack overflow."""
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("CALL", 0, 0, 0),  # infinite recursion
        ])
        st = self.vm.execute(genome, [])
        self.assertTrue(st.halted)
        self.assertEqual(st.error, "STACK_OVERFLOW")

    def test_degenerate_loop_detection(self):
        """Infinite loop with no state change should be detected."""
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("JMP", 0, 0, 0),  # infinite: pc stays at 0
        ])
        st = self.vm.execute(genome, [])
        self.assertTrue(st.halted)
        self.assertEqual(st.error, "DEGENERATE_LOOP")

    def test_inputs_loaded_to_memory(self):
        """Inputs should be available in memory slots 0..n-1."""
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("LOAD", 0, 0, 0),   # R0 = mem[0] (should be first input)
            Instruction("HALT"),
        ])
        # R1 is loaded with len(inputs) in reset, but mem[0] = first input
        st = self.vm.execute(genome, [3.14, 2.71])
        self.assertAlmostEqual(st.regs[0], 3.14)
        self.assertAlmostEqual(st.memory[1], 2.71)

    def test_inc_dec(self):
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("SET", 5, 0, 0),
            Instruction("INC", 0, 0, 0),
            Instruction("DEC", 0, 0, 1),
            Instruction("HALT"),
        ])
        st = self.vm.execute(genome, [])
        self.assertEqual(st.regs[0], 6.0)
        self.assertEqual(st.regs[1], -1.0)

    def test_swap(self):
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("SET", 10, 0, 0),
            Instruction("SET", 20, 0, 1),
            Instruction("SWAP", 0, 1, 0),
            Instruction("HALT"),
        ])
        st = self.vm.execute(genome, [])
        self.assertEqual(st.regs[0], 20.0)
        self.assertEqual(st.regs[1], 10.0)

    def test_coverage_metric(self):
        genome = ProgramGenome(gid="t", instructions=[
            Instruction("SET", 1, 0, 0),
            Instruction("HALT"),
        ])
        st = self.vm.execute(genome, [])
        self.assertEqual(st.coverage(2), 1.0)  # visited both PCs 0 and 1


# ---------------------------------------------------------------------------
# 3. ControlFlowGraph
# ---------------------------------------------------------------------------

class TestControlFlowGraph(unittest.TestCase):
    def test_empty_trace(self):
        cfg = ControlFlowGraph.from_trace([], 5)
        self.assertEqual(cfg.num_edges, 0)

    def test_linear_trace(self):
        cfg = ControlFlowGraph.from_trace([0, 1, 2, 3], 5)
        self.assertGreater(cfg.num_edges, 0)
        self.assertGreater(cfg.num_nodes, 0)

    def test_loop_creates_back_edge(self):
        trace = [0, 1, 2, 0, 1, 2, 0]
        cfg = ControlFlowGraph.from_trace(trace, 3)
        back_edges = [e for e in cfg.edges if e[2] == "BACK"]
        self.assertGreater(len(back_edges), 0)

    def test_canonical_hash_deterministic(self):
        trace = [0, 1, 2, 1, 2, 3]
        cfg1 = ControlFlowGraph.from_trace(trace, 4)
        cfg2 = ControlFlowGraph.from_trace(trace, 4)
        self.assertEqual(cfg1.canonical_hash(), cfg2.canonical_hash())

    def test_different_traces_different_hash(self):
        cfg1 = ControlFlowGraph.from_trace([0, 1, 2], 3)
        cfg2 = ControlFlowGraph.from_trace([0, 2, 1], 3)
        self.assertNotEqual(cfg1.canonical_hash(), cfg2.canonical_hash())

    def test_sccs_with_loop(self):
        """A trace with a loop should produce at least one SCC."""
        trace = [0, 1, 2, 0, 1, 2, 0]
        cfg = ControlFlowGraph.from_trace(trace, 3)
        sccs = cfg.sccs()
        self.assertGreater(len(sccs), 0)

    def test_sccs_linear(self):
        """A strictly linear trace should have no multi-node SCCs."""
        cfg = ControlFlowGraph.from_trace([0, 1, 2, 3, 4], 5)
        sccs = cfg.sccs()
        # No back edges → no SCCs with >1 node
        for scc in sccs:
            self.assertEqual(len(scc), 1)  # self-loops at most


# ---------------------------------------------------------------------------
# 4. ExprNodeCompiler
# ---------------------------------------------------------------------------

class TestExprNodeCompiler(unittest.TestCase):
    def setUp(self):
        self.vocab = VocabularyLayer()
        self.compiler = ExprNodeCompiler(self.vocab)
        self.vm = VirtualMachine(max_steps=200)

    def test_compile_input_x(self):
        """input_x should compile to a program that returns the input."""
        tree = ExprNode("input_x")
        genome = self.compiler.compile(tree)
        self.assertIsInstance(genome, ProgramGenome)
        self.assertGreater(len(genome.instructions), 0)

        # Execute: input=5 → should get 5 back in R0
        st = self.vm.execute(genome, [5.0])
        self.assertAlmostEqual(st.regs[0], 5.0)

    def test_compile_const_one(self):
        tree = ExprNode("const_one")
        genome = self.compiler.compile(tree)
        st = self.vm.execute(genome, [0.0])
        self.assertAlmostEqual(st.regs[0], 1.0)

    def test_compile_const_zero(self):
        tree = ExprNode("const_zero")
        genome = self.compiler.compile(tree)
        st = self.vm.execute(genome, [0.0])
        self.assertAlmostEqual(st.regs[0], 0.0)

    def test_compile_neg(self):
        tree = ExprNode("neg", children=[ExprNode("input_x")])
        genome = self.compiler.compile(tree)
        st = self.vm.execute(genome, [7.0])
        self.assertAlmostEqual(st.regs[0], -7.0)

    def test_compile_square(self):
        tree = ExprNode("square", children=[ExprNode("input_x")])
        genome = self.compiler.compile(tree)
        st = self.vm.execute(genome, [4.0])
        self.assertAlmostEqual(st.regs[0], 16.0)

    def test_compile_add(self):
        """add(input_x, const_one) for x=3 → 4."""
        tree = ExprNode("add", children=[ExprNode("input_x"), ExprNode("const_one")])
        genome = self.compiler.compile(tree)
        st = self.vm.execute(genome, [3.0])
        self.assertAlmostEqual(st.regs[0], 4.0)

    def test_compile_sub(self):
        tree = ExprNode("sub", children=[ExprNode("input_x"), ExprNode("const_one")])
        genome = self.compiler.compile(tree)
        st = self.vm.execute(genome, [10.0])
        self.assertAlmostEqual(st.regs[0], 9.0)

    def test_compile_mul(self):
        tree = ExprNode("mul", children=[ExprNode("input_x"), ExprNode("input_x")])
        genome = self.compiler.compile(tree)
        st = self.vm.execute(genome, [5.0])
        self.assertAlmostEqual(st.regs[0], 25.0)

    def test_compile_safe_div(self):
        tree = ExprNode("safe_div", children=[ExprNode("input_x"), ExprNode("const_one")])
        genome = self.compiler.compile(tree)
        st = self.vm.execute(genome, [8.0])
        self.assertAlmostEqual(st.regs[0], 8.0)

    def test_compile_nested_expression(self):
        """add(mul(input_x, input_x), input_x) = x^2 + x for x=3 → 12."""
        tree = ExprNode("add", children=[
            ExprNode("mul", children=[ExprNode("input_x"), ExprNode("input_x")]),
            ExprNode("input_x"),
        ])
        genome = self.compiler.compile(tree)
        st = self.vm.execute(genome, [3.0])
        self.assertAlmostEqual(st.regs[0], 12.0)

    def test_compile_abs_val(self):
        tree = ExprNode("abs_val", children=[ExprNode("input_x")])
        genome = self.compiler.compile(tree)

        st_pos = self.vm.execute(genome, [5.0])
        self.assertAlmostEqual(st_pos.regs[0], 5.0)

        st_neg = self.vm.execute(genome, [-3.0])
        self.assertAlmostEqual(st_neg.regs[0], 3.0)

    def test_genome_has_halt(self):
        """Every compiled program should end with HALT."""
        tree = ExprNode("input_x")
        genome = self.compiler.compile(tree)
        self.assertEqual(genome.instructions[-1].op, "HALT")

    def test_genome_starts_with_load(self):
        """Every compiled program should start with LOAD (input from memory)."""
        tree = ExprNode("add", children=[ExprNode("input_x"), ExprNode("const_one")])
        genome = self.compiler.compile(tree)
        self.assertEqual(genome.instructions[0].op, "LOAD")


# ---------------------------------------------------------------------------
# 5. VMFitness
# ---------------------------------------------------------------------------

class TestVMFitness(unittest.TestCase):
    def setUp(self):
        self.vocab = VocabularyLayer()
        self.vmf = VMFitness()

    def test_evaluate_tree_returns_float(self):
        tree = ExprNode("input_x")
        score = self.vmf.evaluate_tree(tree, self.vocab, lambda x: x)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_identity_perfect_score(self):
        """input_x vs target f(x)=x should score high."""
        tree = ExprNode("input_x")
        score = self.vmf.evaluate_tree(tree, self.vocab, lambda x: x, (-5, 5), 20)
        self.assertGreater(score, 0.5)

    def test_wrong_tree_low_score(self):
        """const_zero vs target f(x)=x^2 should score low for non-trivial range."""
        tree = ExprNode("const_zero")
        score = self.vmf.evaluate_tree(tree, self.vocab, lambda x: x**2, (-5, 5), 20)
        self.assertLess(score, 0.5)

    def test_structural_score_returns_dict(self):
        tree = ExprNode("add", children=[ExprNode("input_x"), ExprNode("const_one")])
        result = self.vmf.structural_score(tree, self.vocab)
        self.assertIn("coverage", result)
        self.assertIn("loops", result)
        self.assertIn("branches", result)
        self.assertIn("sccs", result)
        self.assertIn("steps", result)
        self.assertIn("halted_cleanly", result)
        self.assertIn("instruction_count", result)
        self.assertIn("cfg_hash", result)

    def test_structural_score_clean_halt(self):
        tree = ExprNode("input_x")
        result = self.vmf.structural_score(tree, self.vocab)
        self.assertTrue(result["halted_cleanly"])
        self.assertIsNone(result["error"])


# ---------------------------------------------------------------------------
# 6. VM Fitness Registry
# ---------------------------------------------------------------------------

class TestVMFitnessRegistry(unittest.TestCase):
    def test_registry_has_entries(self):
        self.assertIn("vm_symbolic_regression", VM_FITNESS_REGISTRY)
        self.assertIn("vm_sine_approximation", VM_FITNESS_REGISTRY)
        self.assertIn("vm_absolute_value", VM_FITNESS_REGISTRY)

    def test_registry_functions_callable(self):
        vocab = VocabularyLayer()
        tree = ExprNode("input_x")
        for name, fn in VM_FITNESS_REGISTRY.items():
            score = fn(tree, vocab)
            self.assertIsInstance(score, float, f"{name} returned non-float")
            self.assertGreaterEqual(score, 0.0, f"{name} returned negative")
            self.assertLessEqual(score, 1.0, f"{name} returned >1")

    def test_lazy_getter_from_main(self):
        reg = _get_vm_fitness_registry()
        self.assertEqual(set(reg.keys()), set(VM_FITNESS_REGISTRY.keys()))


# ---------------------------------------------------------------------------
# 7. Integration: build_rsi_system with VM backend
# ---------------------------------------------------------------------------

class TestVMBackendIntegration(unittest.TestCase):
    def test_build_with_vm_backend(self):
        """build_rsi_system(use_vm_backend=True) should produce a working engine."""
        engine = build_rsi_system(
            use_vm_backend=True,
            vm_fitness_name="vm_symbolic_regression",
            max_depth=3,
            budget_ops=10_000,
            budget_seconds=10.0,
        )
        self.assertIsNotNone(engine)
        # Should be able to run at least 1 generation
        stats = engine.step(population_size=5)
        self.assertIsInstance(stats, dict)

    def test_build_with_vm_backend_sine(self):
        engine = build_rsi_system(
            use_vm_backend=True,
            vm_fitness_name="vm_sine_approximation",
            max_depth=3,
            budget_ops=10_000,
            budget_seconds=10.0,
        )
        stats = engine.step(population_size=5)
        self.assertIsInstance(stats, dict)

    def test_build_default_still_works(self):
        """Default (no VM backend) should still work."""
        engine = build_rsi_system(max_depth=3, budget_ops=10_000)
        stats = engine.step(population_size=5)
        self.assertIsInstance(stats, dict)


# ---------------------------------------------------------------------------
# 8. Edge Cases & Constants
# ---------------------------------------------------------------------------

class TestOmegaConstants(unittest.TestCase):
    def test_ops_list(self):
        self.assertIn("ADD", OPS)
        self.assertIn("HALT", OPS)
        self.assertIn("CALL", OPS)

    def test_control_ops(self):
        self.assertTrue(CONTROL_OPS.issubset(set(OPS)))
        self.assertIn("JMP", CONTROL_OPS)
        self.assertIn("RET", CONTROL_OPS)

    def test_memory_ops(self):
        self.assertTrue(MEMORY_OPS.issubset(set(OPS)))
        self.assertIn("LOAD", MEMORY_OPS)
        self.assertIn("STORE", MEMORY_OPS)


class TestExecutionState(unittest.TestCase):
    def test_coverage_empty(self):
        st = ExecutionState(regs=[0.0]*8, memory={})
        self.assertEqual(st.coverage(0), 0.0)

    def test_coverage_partial(self):
        st = ExecutionState(regs=[0.0]*8, memory={}, visited_pcs={0, 2})
        self.assertAlmostEqual(st.coverage(4), 0.5)


if __name__ == "__main__":
    unittest.main()
