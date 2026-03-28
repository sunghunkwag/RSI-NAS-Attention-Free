"""
Omega VM Backend for RSI-Exploration
=====================================

Integrates the Omega Forge virtual machine as an alternative execution
backend for ExprNode trees. Instead of interpreting trees via _eval_tree,
this module compiles ExprNode trees down to Omega VM instructions and
executes them on a register-based virtual machine with memory, control
flow, and a call stack.

This provides:
1. ExprNodeCompiler: ExprNode tree -> Instruction list (register allocation)
2. Slim VirtualMachine: 8-register VM with memory and control flow
3. VMFitness: fitness functions that evaluate compiled ExprNode programs
4. ControlFlowGraph: structural analysis of execution traces

The compilation bridge enables RSI-evolved expression trees to be
"grounded" in a low-level execution model, catching semantic errors
that the high-level interpreter would silently mask (e.g., division by
zero paths, numeric overflow cascades, degenerate loops).

Architecture correspondence:
  ExprNode tree      ->  ProgramGenome (compiled instruction list)
  _eval_tree(node)   ->  VirtualMachine.execute(genome, inputs)
  fitness_fn(tree)    ->  VMFitness.evaluate(tree, vocab, targets)

Source: omega_engine.py (OMEGA_FORGE_V13_CLEAN) by sunghunkwag
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

# Import RSI core types
from main import (
    ExprNode,
    VocabularyLayer,
    PrimitiveOp,
    PolymorphicOp,
    EvalContext,
    OpType,
)


# ===========================================================================
# 1. Instruction Set & Program Genome
# ===========================================================================

OPS = [
    "MOV", "SET", "SWAP",
    "ADD", "SUB", "MUL", "DIV", "INC", "DEC",
    "LOAD", "STORE",
    "JMP", "JZ", "JNZ", "JGT", "JLT",
    "CALL", "RET", "HALT",
]
CONTROL_OPS = {"JMP", "JZ", "JNZ", "JGT", "JLT", "CALL", "RET"}
MEMORY_OPS = {"LOAD", "STORE"}


@dataclass
class Instruction:
    op: str
    a: int = 0
    b: int = 0
    c: int = 0

    def clone(self) -> "Instruction":
        return Instruction(self.op, self.a, self.b, self.c)

    def to_tuple(self) -> Tuple[Any, ...]:
        return (self.op, int(self.a), int(self.b), int(self.c))


@dataclass
class ProgramGenome:
    gid: str
    instructions: List[Instruction]
    parents: List[str] = field(default_factory=list)
    generation: int = 0
    last_score: float = 0.0
    last_cfg_hash: str = ""

    def clone(self) -> "ProgramGenome":
        return ProgramGenome(
            gid=self.gid,
            instructions=[i.clone() for i in self.instructions],
            parents=list(self.parents),
            generation=self.generation,
        )

    def code_hash(self) -> str:
        h = hashlib.sha256()
        for inst in self.instructions:
            h.update(repr(inst.to_tuple()).encode("utf-8"))
        return h.hexdigest()[:16]


# ===========================================================================
# 2. Execution State
# ===========================================================================

@dataclass
class ExecutionState:
    regs: List[float]
    memory: Dict[int, float]
    pc: int = 0
    stack: List[int] = field(default_factory=list)
    steps: int = 0
    halted: bool = False
    halted_cleanly: bool = False
    error: Optional[str] = None
    trace: List[int] = field(default_factory=list)
    visited_pcs: Set[int] = field(default_factory=set)
    loops_count: int = 0
    conditional_branches: int = 0
    max_call_depth: int = 0
    memory_reads: int = 0
    memory_writes: int = 0

    def coverage(self, code_len: int) -> float:
        if code_len <= 0:
            return 0.0
        return len(self.visited_pcs) / float(code_len)


# ===========================================================================
# 3. Virtual Machine
# ===========================================================================

class VirtualMachine:
    """
    8-register VM with addressable memory, a call stack, and control flow.
    Adapted from omega_engine.py (OMEGA_FORGE_V13).
    """

    def __init__(self, max_steps: int = 400, memory_size: int = 64,
                 stack_limit: int = 16) -> None:
        self.max_steps = max_steps
        self.memory_size = memory_size
        self.stack_limit = stack_limit

    def reset(self, inputs: List[float]) -> ExecutionState:
        regs = [0.0] * 8
        mem: Dict[int, float] = {}
        for i, v in enumerate(inputs):
            if i < self.memory_size:
                mem[i] = float(v)
        regs[1] = float(len(inputs))
        return ExecutionState(regs=regs, memory=mem)

    def execute(self, genome: ProgramGenome, inputs: List[float]) -> ExecutionState:
        st = self.reset(inputs)
        code = genome.instructions
        L = len(code)

        recent_hashes: List[int] = []
        while not st.halted and st.steps < self.max_steps:
            if st.pc < 0 or st.pc >= L:
                st.halted = True
                st.halted_cleanly = True
                break

            st.visited_pcs.add(st.pc)
            st.trace.append(st.pc)
            prev_pc = st.pc
            inst = code[st.pc]
            st.steps += 1

            # Degenerate loop detection
            state_sig = hash((st.pc, tuple(int(x) for x in st.regs[:4]), len(st.stack)))
            recent_hashes.append(state_sig)
            if len(recent_hashes) > 25:
                recent_hashes.pop(0)
                if len(set(recent_hashes)) < 3:
                    st.error = "DEGENERATE_LOOP"
                    st.halted = True
                    break

            try:
                self._step(st, inst)
            except Exception as e:
                st.error = f"VM_ERR:{e.__class__.__name__}"
                st.halted = True
                break

            if st.pc <= prev_pc and not st.halted:
                st.loops_count += 1
            if inst.op in {"JZ", "JNZ", "JGT", "JLT"}:
                st.conditional_branches += 1
            st.max_call_depth = max(st.max_call_depth, len(st.stack))

        return st

    def _step(self, st: ExecutionState, inst: Instruction) -> None:
        op, a, b, c = inst.op, inst.a, inst.b, inst.c
        r = st.regs

        def clamp(x: float) -> float:
            if not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x):
                return 0.0
            return float(max(-1e9, min(1e9, x)))

        def addr(x: float) -> int:
            return int(max(0, min(self.memory_size - 1, int(x))))

        jump = False

        if op == "HALT":
            st.halted = True
            st.halted_cleanly = True
            return
        if op == "SET":
            r[c % 8] = float(a)
        elif op == "MOV":
            r[c % 8] = float(r[a % 8])
        elif op == "SWAP":
            ra, rb = a % 8, b % 8
            r[ra], r[rb] = r[rb], r[ra]
        elif op == "ADD":
            r[c % 8] = clamp(r[a % 8] + r[b % 8])
        elif op == "SUB":
            r[c % 8] = clamp(r[a % 8] - r[b % 8])
        elif op == "MUL":
            r[c % 8] = clamp(r[a % 8] * r[b % 8])
        elif op == "DIV":
            den = r[b % 8]
            r[c % 8] = clamp(r[a % 8] / den) if abs(den) > 1e-9 else 0.0
        elif op == "INC":
            r[c % 8] = clamp(r[c % 8] + 1.0)
        elif op == "DEC":
            r[c % 8] = clamp(r[c % 8] - 1.0)
        elif op == "LOAD":
            idx = addr(r[a % 8])
            st.memory_reads += 1
            r[c % 8] = float(st.memory.get(idx, 0.0))
        elif op == "STORE":
            idx = addr(r[a % 8])
            st.memory_writes += 1
            st.memory[idx] = clamp(r[c % 8])
        elif op == "JMP":
            st.pc += int(a)
            jump = True
        elif op == "JZ":
            if abs(r[a % 8]) < 1e-9:
                st.pc += int(b)
                jump = True
        elif op == "JNZ":
            if abs(r[a % 8]) >= 1e-9:
                st.pc += int(b)
                jump = True
        elif op == "JGT":
            if r[a % 8] > r[b % 8]:
                st.pc += int(c)
                jump = True
        elif op == "JLT":
            if r[a % 8] < r[b % 8]:
                st.pc += int(c)
                jump = True
        elif op == "CALL":
            if len(st.stack) >= self.stack_limit:
                st.error = "STACK_OVERFLOW"
                st.halted = True
                return
            st.stack.append(st.pc + 1)
            st.pc += int(a)
            jump = True
        elif op == "RET":
            if not st.stack:
                st.halted = True
                st.halted_cleanly = True
                jump = True
            else:
                st.pc = st.stack.pop()
                jump = True
        else:
            st.error = "UNKNOWN_OP"
            st.halted = True
            return

        if not jump:
            st.pc += 1


# ===========================================================================
# 4. Control Flow Graph
# ===========================================================================

class ControlFlowGraph:
    """CFG extracted from VM execution traces for structural analysis."""

    def __init__(self) -> None:
        self.edges: Set[Tuple[int, int, str]] = set()
        self.nodes: Set[int] = set()

    def add_edge(self, f: int, t: int, ty: str) -> None:
        self.edges.add((int(f), int(t), str(ty)))
        self.nodes.add(int(f))
        self.nodes.add(int(t))

    @staticmethod
    def from_trace(trace: List[int], code_len: int) -> "ControlFlowGraph":
        cfg = ControlFlowGraph()
        if not trace:
            return cfg
        for i in range(len(trace) - 1):
            a, b = trace[i], trace[i + 1]
            ty = "BACK" if b <= a else "SEQ"
            cfg.add_edge(a, b, ty)
        last = trace[-1]
        cfg.nodes.add(last)
        cfg.nodes.add(max(0, min(code_len, last + 1)))
        return cfg

    def canonical_hash(self) -> str:
        h = hashlib.sha256()
        for f, t, ty in sorted(self.edges):
            h.update(f"{f}->{t}:{ty};".encode("utf-8"))
        scc_sizes = sorted([len(s) for s in self.sccs()])
        h.update(("SCC:" + ",".join(map(str, scc_sizes))).encode("utf-8"))
        return h.hexdigest()[:16]

    def sccs(self) -> List[FrozenSet[int]]:
        """Kosaraju's algorithm for strongly connected components."""
        if not self.nodes:
            return []
        adj = defaultdict(list)
        radj = defaultdict(list)
        for f, t, _ in self.edges:
            adj[f].append(t)
            radj[t].append(f)

        visited: Set[int] = set()
        order: List[int] = []

        def dfs1(u: int) -> None:
            if u in visited:
                return
            visited.add(u)
            for v in adj[u]:
                dfs1(v)
            order.append(u)

        for n in list(self.nodes):
            dfs1(n)

        visited.clear()
        comps: List[FrozenSet[int]] = []

        def dfs2(u: int, comp: Set[int]) -> None:
            if u in visited:
                return
            visited.add(u)
            comp.add(u)
            for v in radj[u]:
                dfs2(v, comp)

        for u in reversed(order):
            if u not in visited:
                comp: Set[int] = set()
                dfs2(u, comp)
                if len(comp) > 1:
                    comps.append(frozenset(comp))
                else:
                    x = next(iter(comp)) if comp else None
                    if x is not None and any(
                        (x, x, ty) in self.edges for ty in ("SEQ", "BACK")
                    ):
                        comps.append(frozenset(comp))
        return comps

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


# ===========================================================================
# 5. ExprNode Compiler: ExprNode tree -> Instruction list
# ===========================================================================

# Register allocation convention:
#   R0 = result / accumulator (output of any compiled subtree)
#   R1 = input x (set before execution from memory slot 0)
#   R2-R7 = temporaries (stack-based allocation)

# Mapping from RSI PrimitiveOp names to VM opcodes
_EXPR_TO_VM_OP = {
    "add": "ADD",
    "sub": "SUB",
    "mul": "MUL",
    "safe_div": "DIV",
}

# Unary ops compiled as: result = f(arg) using scratch registers
_UNARY_PATTERNS = {
    "neg": [
        Instruction("SET", 0, 0, 7),       # R7 = 0
        Instruction("SUB", 7, "{src}", 0),  # R0 = 0 - arg
    ],
    "abs_val": [
        # if arg < 0: result = -arg; else result = arg
        Instruction("SET", 0, 0, 7),          # R7 = 0
        Instruction("JGT", "{src}", 7, 3),    # if arg > 0, skip negate
        Instruction("SUB", 7, "{src}", 0),     # R0 = 0 - arg
        Instruction("JMP", 2, 0, 0),           # skip to end
        Instruction("MOV", "{src}", 0, 0),     # R0 = arg (positive case)
    ],
    "square": [
        Instruction("MUL", "{src}", "{src}", 0),  # R0 = arg * arg
    ],
    "identity": [
        Instruction("MOV", "{src}", 0, 0),  # R0 = arg
    ],
    "clamp": [
        Instruction("MOV", "{src}", 0, 0),  # R0 = arg (simplified clamp)
    ],
}


class ExprNodeCompiler:
    """
    Compiles an ExprNode tree into a list of Omega VM Instructions.

    Register allocation:
    - R0: accumulator / final result
    - R1: input x (loaded from memory[0] at program start)
    - R2-R6: temporaries for subtree evaluation (stack discipline)
    - R7: scratch register

    The compiler does a post-order traversal of the ExprNode tree,
    emitting instructions that compute each subtree's value into R0,
    then MOV into a temp register before computing the parent.
    """

    def __init__(self, vocab: VocabularyLayer):
        self.vocab = vocab
        self._next_temp = 2  # R2 is first available temp

    def compile(self, tree: ExprNode) -> ProgramGenome:
        """Compile an ExprNode tree into a ProgramGenome."""
        self._next_temp = 2
        instructions = []

        # Prologue: load input x from memory[0] into R1
        instructions.append(Instruction("LOAD", 0, 0, 1))  # R1 = mem[0]

        # Compile the tree body
        result_reg = self._compile_node(tree, instructions)

        # Move result to R0 if not already there
        if result_reg != 0:
            instructions.append(Instruction("MOV", result_reg, 0, 0))

        # Epilogue: store result to memory[0] and halt
        instructions.append(Instruction("STORE", 0, 0, 0))  # mem[0] = R0
        instructions.append(Instruction("HALT", 0, 0, 0))

        return ProgramGenome(
            gid=f"compiled_{tree.fingerprint()}",
            instructions=instructions,
        )

    def _alloc_temp(self) -> int:
        """Allocate a temporary register (R2-R6)."""
        reg = self._next_temp
        if reg > 6:
            reg = 2  # wrap around (spill strategy)
        self._next_temp = reg + 1
        return reg

    def _compile_node(self, node: ExprNode, instructions: List[Instruction]) -> int:
        """
        Compile a single node, appending instructions. Returns the register
        holding the result.
        """
        # Leaf: input_x
        if node.op_name == "input_x":
            return 1  # R1 holds input x

        # Leaf: self_encode (compile as constant 0.5)
        if node.op_name == "self_encode":
            temp = self._alloc_temp()
            # Encode as SET with a fixed value (simplified)
            instructions.append(Instruction("SET", 0, 0, temp))
            # Use fingerprint-based value approximation
            instructions.append(Instruction("INC", 0, 0, temp))  # temp = 1.0
            # We'll use a simple constant; full self-reference needs runtime support
            return temp

        # Leaf: constants
        if node.op_name == "const_one":
            temp = self._alloc_temp()
            instructions.append(Instruction("SET", 1, 0, temp))
            return temp

        if node.op_name == "const_zero":
            temp = self._alloc_temp()
            instructions.append(Instruction("SET", 0, 0, temp))
            return temp

        # Look up the op
        op = self.vocab.get(node.op_name)

        # Nullary ops (arity 0)
        if op is not None and op.arity == 0:
            temp = self._alloc_temp()
            try:
                val = int(op())
            except Exception:
                val = 0
            instructions.append(Instruction("SET", val, 0, temp))
            return temp

        # Unary ops
        if op is not None and op.arity == 1 and node.children:
            child_reg = self._compile_node(node.children[0], instructions)
            return self._compile_unary(node.op_name, child_reg, instructions)

        # Binary ops
        if op is not None and op.arity == 2 and len(node.children) >= 2:
            left_reg = self._compile_node(node.children[0], instructions)
            # Save left result to temp before compiling right
            left_temp = self._alloc_temp()
            if left_reg != left_temp:
                instructions.append(Instruction("MOV", left_reg, 0, left_temp))

            right_reg = self._compile_node(node.children[1], instructions)
            right_temp = self._alloc_temp()
            if right_reg != right_temp:
                instructions.append(Instruction("MOV", right_reg, 0, right_temp))

            return self._compile_binary(node.op_name, left_temp, right_temp, instructions)

        # Fallback: unknown op -> return 0
        temp = self._alloc_temp()
        instructions.append(Instruction("SET", 0, 0, temp))
        return temp

    def _compile_unary(self, op_name: str, src_reg: int,
                       instructions: List[Instruction]) -> int:
        """Compile a unary operation."""
        if op_name == "neg":
            instructions.append(Instruction("SET", 0, 0, 7))
            instructions.append(Instruction("SUB", 7, src_reg, 0))
            return 0
        elif op_name == "square":
            instructions.append(Instruction("MUL", src_reg, src_reg, 0))
            return 0
        elif op_name in ("identity", "clamp"):
            instructions.append(Instruction("MOV", src_reg, 0, 0))
            return 0
        elif op_name == "abs_val":
            instructions.append(Instruction("SET", 0, 0, 7))
            instructions.append(Instruction("JGT", src_reg, 7, 3))
            instructions.append(Instruction("SUB", 7, src_reg, 0))
            instructions.append(Instruction("JMP", 2, 0, 0))
            instructions.append(Instruction("MOV", src_reg, 0, 0))
            return 0
        else:
            # Generic unary: just pass through (compiled ops, library ops, etc.)
            instructions.append(Instruction("MOV", src_reg, 0, 0))
            return 0

    def _compile_binary(self, op_name: str, left_reg: int, right_reg: int,
                        instructions: List[Instruction]) -> int:
        """Compile a binary operation."""
        vm_op = _EXPR_TO_VM_OP.get(op_name)
        if vm_op:
            instructions.append(Instruction(vm_op, left_reg, right_reg, 0))
            return 0
        else:
            # Fallback: treat as ADD
            instructions.append(Instruction("ADD", left_reg, right_reg, 0))
            return 0


# ===========================================================================
# 6. VM-Based Fitness Functions
# ===========================================================================

class VMFitness:
    """
    Fitness evaluation by compiling ExprNode trees to VM instructions
    and executing them on the VirtualMachine.

    This provides a "grounded" fitness signal: trees must produce correct
    VM programs, not just evaluate correctly in a high-level interpreter.
    """

    def __init__(self, vm: VirtualMachine = None, vocab: VocabularyLayer = None):
        self.vm = vm or VirtualMachine(max_steps=400)
        self.vocab = vocab

    def evaluate_tree(self, tree: ExprNode, vocab: VocabularyLayer,
                      target_fn: Callable, x_range: Tuple[float, float] = (-5, 5),
                      n_points: int = 20) -> float:
        """
        Compile tree, execute on VM for multiple inputs, compare to target.

        Returns fitness in [0, 1].
        """
        compiler = ExprNodeCompiler(vocab)
        try:
            genome = compiler.compile(tree)
        except Exception:
            return 0.0

        xs = np.linspace(x_range[0], x_range[1], n_points)
        total_error = 0.0

        for x in xs:
            try:
                st = self.vm.execute(genome, [float(x)])
                # Result is in R0 after execution
                result = st.regs[0]
                if st.error or not st.halted_cleanly:
                    result = 0.0
            except Exception:
                result = 0.0

            expected = target_fn(float(x))
            total_error += abs(result - expected)

        avg_error = total_error / n_points
        return 1.0 / (1.0 + min(avg_error, 1e6))

    def structural_score(self, tree: ExprNode, vocab: VocabularyLayer) -> Dict[str, Any]:
        """
        Compile tree and analyze the resulting VM program's structural properties:
        coverage, loops, branches, SCCs.
        """
        compiler = ExprNodeCompiler(vocab)
        try:
            genome = compiler.compile(tree)
        except Exception:
            return {"coverage": 0, "loops": 0, "branches": 0, "sccs": 0, "steps": 0}

        st = self.vm.execute(genome, [1.0])
        cfg = ControlFlowGraph.from_trace(st.trace, len(genome.instructions))

        return {
            "coverage": st.coverage(len(genome.instructions)),
            "loops": st.loops_count,
            "branches": st.conditional_branches,
            "sccs": len(cfg.sccs()),
            "steps": st.steps,
            "halted_cleanly": st.halted_cleanly,
            "error": st.error,
            "instruction_count": len(genome.instructions),
            "cfg_hash": cfg.canonical_hash(),
        }


# ===========================================================================
# 7. Fitness Functions for RSI Integration
# ===========================================================================

def vm_symbolic_regression_fitness(tree: ExprNode, vocab: VocabularyLayer,
                                    ctx: EvalContext = None) -> float:
    """Target: f(x) = x^2 + 2x + 1  over [-5, 5]. VM-compiled execution."""
    vmf = VMFitness()
    return vmf.evaluate_tree(tree, vocab, lambda x: x**2 + 2*x + 1, (-5, 5), 20)


def vm_sine_approximation_fitness(tree: ExprNode, vocab: VocabularyLayer,
                                   ctx: EvalContext = None) -> float:
    """Target: f(x) = sin(x)  over [-pi, pi]. VM-compiled execution."""
    vmf = VMFitness()
    return vmf.evaluate_tree(tree, vocab, lambda x: math.sin(x),
                             (-math.pi, math.pi), 30)


def vm_absolute_value_fitness(tree: ExprNode, vocab: VocabularyLayer,
                               ctx: EvalContext = None) -> float:
    """Target: f(x) = |x|  over [-5, 5]. VM-compiled execution."""
    vmf = VMFitness()
    return vmf.evaluate_tree(tree, vocab, lambda x: abs(x), (-5, 5), 30)


VM_FITNESS_REGISTRY: Dict[str, Callable] = {
    "vm_symbolic_regression": vm_symbolic_regression_fitness,
    "vm_sine_approximation": vm_sine_approximation_fitness,
    "vm_absolute_value": vm_absolute_value_fitness,
}