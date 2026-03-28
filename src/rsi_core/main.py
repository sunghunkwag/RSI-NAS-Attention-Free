"""
RSI-Exploration: Recursive Self-Improvement Architecture
=========================================================

A hybrid architecture combining:
1. Darwin Godel Machine (DGM) self-improvement loops
2. MAP-Elites quality-diversity search

The system implements three-layer design space expansion:
- Vocabulary Layer: primitive operations that can be composed
- Grammar Layer: rules for composing vocabulary into programs
- Meta-Grammar Layer: rules for generating new grammar rules

Physical cost grounding is provided through a resource accounting
system that tracks compute, memory, and wall-clock time.

Inspired by:
- guillaumepourcel/dgm (Darwin Godel Machine)
- b-albar/evolve-anything (LLM-powered MAP-Elites)
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. VOCABULARY LAYER
# ---------------------------------------------------------------------------

class OpType:
    """
    Refinement type tags for PrimitiveOp input/output domains (D.5 Dependent Types).

    Each tag represents a constraint on the numeric domain:
    - REAL: unrestricted reals (default)
    - NON_NEGATIVE: x >= 0
    - POSITIVE: x > 0
    - BOUNDED: -1e6 <= x <= 1e6
    - UNIT: 0 <= x <= 1
    - ANY: accepts any type (universal input)

    Type compatibility is checked at tree construction time: a child node's
    output_type must be a subtype of (or equal to) the parent's input_type
    for that argument position.
    """
    REAL = "real"
    NON_NEGATIVE = "non_negative"
    POSITIVE = "positive"
    BOUNDED = "bounded"
    UNIT = "unit"
    ANY = "any"

    # Subtype lattice: A is subtype of B if A's domain is a subset of B's domain
    _SUBTYPE_OF = {
        "unit": {"unit", "non_negative", "bounded", "real", "any"},
        "positive": {"positive", "non_negative", "real", "any"},
        "non_negative": {"non_negative", "real", "any"},
        "bounded": {"bounded", "real", "any"},
        "real": {"real", "any"},
        "any": {"any"},
    }

    @staticmethod
    def is_subtype(child_type: str, parent_type: str) -> bool:
        """Check if child_type's domain is a subset of parent_type's domain."""
        return parent_type in OpType._SUBTYPE_OF.get(child_type, {"any"})


@dataclass
class PrimitiveOp:
    """
    A single primitive operation in the vocabulary.

    Refinement type fields (D.5 Dependent Types):
    - input_types: list of type tags for each argument position
    - output_type: type tag for the return value
    These enable type-checked tree composition at construction time.
    """
    name: str
    arity: int
    fn: Callable
    cost: float = 1.0
    description: str = ""
    input_types: List[str] = field(default_factory=list)
    output_type: str = "real"

    def __post_init__(self):
        # Default: all inputs accept any real
        if not self.input_types:
            self.input_types = [OpType.REAL] * self.arity

    def __call__(self, *args):
        return self.fn(*args)

    def accepts_child_type(self, arg_index: int, child_output_type: str) -> bool:
        """Check if a child's output type is compatible with this op's input at arg_index."""
        if arg_index >= len(self.input_types):
            return True  # no constraint
        return OpType.is_subtype(child_output_type, self.input_types[arg_index])


@dataclass
class EvalContext:
    """
    Evaluation context threaded through ExprNode evaluation.

    Implements Mechanism 2 (Context-Dependent Evaluation) from Synthesis:
    Sources: C.1c (Karaka), C.3 (Aramaic polysemy), C.4 (Cuneiform),
             G.6 (Topos Theory), D.4 (Reflection).

    The context enables polymorphic PrimitiveOps that dispatch to different
    functions based on context state. For k context states and n ops,
    up to nxk distinct functions become available per node.

    Topological fields (G.6 Topos Logic):
    - current_depth: depth of the node being evaluated within the tree
    - parent_op_name: the op name of the node's parent (structural context)
    - sibling_index: position among siblings (left=0, right=1, etc.)
    - subtree_size: size of the subtree rooted at the current node
    These fields are updated during _eval_tree traversal, enabling
    dispatch based on actual tree topology rather than just external metadata.
    """
    niche_id: int = 0
    generation: int = 0
    env_tag: str = "default"
    self_fingerprint: str = ""
    custom: Dict = field(default_factory=dict)
    # Topological fields (updated during tree traversal)
    current_depth: int = 0
    parent_op_name: str = ""
    sibling_index: int = 0
    subtree_size: int = 1

    def context_key(self) -> int:
        """Return a discrete context state for dispatch."""
        return hash((self.niche_id, self.env_tag)) % 4

    def topo_key(self) -> int:
        """
        Return a topological context key derived from tree structure.

        Combines depth, parent op, and sibling position into a discrete
        dispatch key. This enables the same PolymorphicOp to compute
        different functions based on WHERE in the tree it appears.
        """
        return hash((self.current_depth, self.parent_op_name, self.sibling_index)) % 8

    def full_key(self) -> int:
        """
        Combined key incorporating both external context and tree topology.
        Provides the finest-grained dispatch: same op, same tree, different
        position or different context -> potentially different function.
        """
        return hash((self.niche_id, self.env_tag, self.current_depth,
                     self.parent_op_name, self.sibling_index)) % 16

    def with_topo(self, depth: int, parent_op: str, sib_idx: int,
                  sub_size: int) -> "EvalContext":
        """
        Return a copy of this context with updated topological fields.
        This avoids mutating the context during recursive evaluation.
        """
        return EvalContext(
            niche_id=self.niche_id,
            generation=self.generation,
            env_tag=self.env_tag,
            self_fingerprint=self.self_fingerprint,
            custom=self.custom,
            current_depth=depth,
            parent_op_name=parent_op,
            sibling_index=sib_idx,
            subtree_size=sub_size,
        )


@dataclass
class PolymorphicOp:
    """
    A PrimitiveOp that dispatches to different functions based on EvalContext.

    Implements the core FORMAT_CHANGE from context-free to context-dependent
    evaluation. Same tree structure can compute different functions depending
    on the evaluation context.

    Supports three dispatch modes (tried in order):
    1. topo_dispatch_table: keyed by topo_key() — dispatches based on tree
       topology (depth, parent op, sibling position). This is the G.6 Topos
       Logic upgrade: evaluation depends on WHERE in the tree the op appears.
    2. dispatch_table: keyed by context_key() — dispatches based on external
       context (niche, env_tag). This is the original C.3/C.4 mechanism.
    3. default_fn: fallback when no context or no matching key.
    """
    name: str
    arity: int
    dispatch_table: Dict[int, Callable]
    default_fn: Callable = None
    cost: float = 1.5
    description: str = ""
    topo_dispatch_table: Dict[int, Callable] = field(default_factory=dict)

    def __call__(self, *args, ctx: EvalContext = None):
        if ctx is not None:
            # Priority 1: topological dispatch (G.6 Topos)
            if self.topo_dispatch_table:
                tkey = ctx.topo_key()
                fn = self.topo_dispatch_table.get(tkey)
                if fn is not None:
                    return fn(*args)
            # Priority 2: context dispatch (C.3/C.4)
            key = ctx.context_key()
            fn = self.dispatch_table.get(key, self.default_fn)
        else:
            fn = self.default_fn
        if fn is None:
            fn = next(iter(self.dispatch_table.values()))
        return fn(*args)

    @property
    def fn(self):
        """Compatibility: return default_fn for non-context evaluation."""
        return self.default_fn or next(iter(self.dispatch_table.values()))

    # Refinement type compatibility (D.5): PolymorphicOps accept any REAL input
    # and output REAL, since their actual behavior depends on context.
    input_types: List[str] = field(default_factory=lambda: [OpType.REAL])
    output_type: str = OpType.REAL

    def __post_init__(self):
        if not self.input_types or len(self.input_types) < self.arity:
            self.input_types = [OpType.REAL] * self.arity

    def accepts_child_type(self, arg_index: int, child_output_type: str) -> bool:
        """PolymorphicOps accept any type (conservative: always compatible)."""
        if arg_index >= len(self.input_types):
            return True
        return OpType.is_subtype(child_output_type, self.input_types[arg_index])


class VocabularyLayer:
    """Manages the set of primitive operations available to the system."""

    def __init__(self):
        self._ops: Dict[str, PrimitiveOp] = {}
        self._register_defaults()

    def _register_defaults(self):
        R = OpType.REAL
        NN = OpType.NON_NEGATIVE
        BD = OpType.BOUNDED
        defaults = [
            PrimitiveOp("add", 2, lambda a, b: a + b, 1.0, "Addition",
                         input_types=[R, R], output_type=R),
            PrimitiveOp("sub", 2, lambda a, b: a - b, 1.0, "Subtraction",
                         input_types=[R, R], output_type=R),
            PrimitiveOp("mul", 2, lambda a, b: a * b, 1.5, "Multiplication",
                         input_types=[R, R], output_type=R),
            PrimitiveOp("safe_div", 2, lambda a, b: a / b if b != 0 else 0.0, 2.0,
                         "Safe division", input_types=[R, R], output_type=R),
            PrimitiveOp("neg", 1, lambda a: -a, 0.5, "Negation",
                         input_types=[R], output_type=R),
            PrimitiveOp("abs_val", 1, lambda a: abs(a), 0.5, "Absolute value",
                         input_types=[R], output_type=NN),
            PrimitiveOp("square", 1, lambda a: a * a, 1.0, "Square",
                         input_types=[R], output_type=NN),
            PrimitiveOp("clamp", 1, lambda a: max(-1e6, min(1e6, a)), 0.5,
                         "Clamp to safe range", input_types=[R], output_type=BD),
            PrimitiveOp("identity", 1, lambda a: a, 0.1, "Identity",
                         input_types=[R], output_type=R),
            PrimitiveOp("const_one", 0, lambda: 1.0, 0.1, "Constant 1",
                         output_type=OpType.POSITIVE),
            PrimitiveOp("const_zero", 0, lambda: 0.0, 0.1, "Constant 0",
                         output_type=NN),
            # Mechanism 1: Self-Reference (A.7, D.1, D.7)
            # self_encode returns 0.0 from VocabularyLayer; the actual
            # fingerprint-dependent value is computed in _eval_tree when
            # an EvalContext with self_fingerprint is available.
            # Registering it here makes it reachable by random_tree/mutate.
            PrimitiveOp("self_encode", 0, lambda: 0.0, 0.5,
                         "Self-reference: tree's own fingerprint hash",
                         output_type=BD),
        ]
        for op in defaults:
            self._ops[op.name] = op

    def register(self, op: PrimitiveOp):
        self._ops[op.name] = op
        logger.info(f"Vocabulary expanded: +{op.name} (arity={op.arity}, cost={op.cost})")

    def get(self, name: str) -> Optional[PrimitiveOp]:
        return self._ops.get(name)

    def unregister(self, name: str) -> bool:
        """Remove a dynamically generated op from the vocabulary.

        Only removes non-default ops (those added by meta-grammar or library learning).
        Returns True if the op was removed, False if it was a default op or not found.
        """
        if name not in self._ops:
            return False
        # Protect default ops from removal
        if not (name.startswith("lib_") or name.startswith("poly_")
                or "_then_" in name or "_of_" in name or "_with_" in name):
            return False
        del self._ops[name]
        logger.info(f"Vocabulary pruned: -{name}")
        return True

    def all_ops(self) -> List[PrimitiveOp]:
        return list(self._ops.values())

    def generated_op_names(self) -> List[str]:
        """Return names of all dynamically generated (non-default) ops."""
        return [name for name in self._ops
                if (name.startswith("lib_") or name.startswith("poly_")
                    or "_then_" in name or "_of_" in name or "_with_" in name)]

    def random_op(self, max_arity: int = 2) -> PrimitiveOp:
        candidates = [op for op in self._ops.values() if op.arity <= max_arity]
        return random.choice(candidates)

    @property
    def size(self) -> int:
        return len(self._ops)


# ---------------------------------------------------------------------------
# 2. GRAMMAR LAYER
# ---------------------------------------------------------------------------

@dataclass
class ExprNode:
    """A node in an expression tree (AST)."""
    op_name: str
    children: List["ExprNode"] = field(default_factory=list)
    value: Optional[float] = None

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def to_dict(self) -> dict:
        d = {"op": self.op_name}
        if self.value is not None:
            d["value"] = self.value
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    def fingerprint(self) -> str:
        return hashlib.md5(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


class GrammarLayer:
    """
    Rules for composing vocabulary into expression trees.

    Refinement type enforcement (D.5 Dependent Types):
    Tree construction and mutation respect type constraints. When selecting
    an op for a node, the grammar checks that each child's output_type is
    compatible with the op's input_type at that position. This prevents
    invalid compositions (e.g., feeding a possibly-negative value into an
    op that requires non-negative input) at creation time.
    """

    def __init__(self, vocab: VocabularyLayer, max_depth: int = 5, max_size: int = 30):
        self.vocab = vocab
        self.max_depth = max_depth
        self.max_size = max_size
        self._composition_rules: List[Callable] = []
        self._register_default_rules()

    def _register_default_rules(self):
        self._composition_rules.extend([
            self._rule_grow,
            self._rule_point_mutate,
            self._rule_subtree_crossover,
            self._rule_hoist,
        ])

    def infer_output_type(self, node: ExprNode) -> str:
        """
        Infer the output type of a subtree rooted at node.
        Used for type-checking during composition.
        """
        if node.op_name == "input_x":
            return OpType.REAL
        if node.op_name == "self_encode":
            return OpType.UNIT
        op = self.vocab.get(node.op_name)
        if op is None:
            return OpType.REAL
        return getattr(op, "output_type", OpType.REAL)

    def _type_compatible_op(self, max_arity: int, child_types: List[str] = None,
                            max_attempts: int = 10) -> PrimitiveOp:
        """
        Select an op that is type-compatible with the given children's output types.
        Falls back to an unconstrained op if no compatible one is found.
        """
        if child_types is None:
            return self.vocab.random_op(max_arity=max_arity)

        for _ in range(max_attempts):
            op = self.vocab.random_op(max_arity=max_arity)
            if op.arity == 0:
                return op
            # Check type compatibility for each argument
            compatible = True
            for i, ct in enumerate(child_types[:op.arity]):
                if not op.accepts_child_type(i, ct):
                    compatible = False
                    break
            if compatible:
                return op
        # Fallback: return any op (preserves backward compatibility)
        return self.vocab.random_op(max_arity=max_arity)

    def random_tree(self, max_depth: int = None) -> ExprNode:
        return self._rule_grow(max_depth or self.max_depth)

    def _rule_grow(self, max_depth: int = 3, required_type: str = None) -> ExprNode:
        """
        Grow a random tree respecting type constraints (D.5).

        If required_type is specified, the root of the generated subtree
        must produce an output compatible with that type.
        """
        if max_depth <= 0:
            if random.random() < 0.5:
                return ExprNode("input_x")
            op = self.vocab.random_op(max_arity=0)
            return ExprNode(op.name)
        op = self.vocab.random_op()
        # Build children, then check type compatibility
        children = []
        for i in range(op.arity):
            # Determine what type this argument position requires
            arg_type = op.input_types[i] if i < len(getattr(op, 'input_types', [])) else OpType.REAL
            child = self._rule_grow(max_depth - 1, required_type=arg_type)
            children.append(child)
        return ExprNode(op.name, children=children)

    def _rule_point_mutate(self, tree: ExprNode = None) -> ExprNode:
        """
        Point mutation with type-constraint enforcement (D.5).

        When replacing an op, the new op must be type-compatible with
        the existing children's output types.
        """
        if tree is None:
            tree = self.random_tree(2)
        tree = copy.deepcopy(tree)
        nodes = self._collect_nodes(tree)
        if not nodes:
            return tree
        target = random.choice(nodes)
        # Infer children types for constraint checking
        child_types = [self.infer_output_type(c) for c in target.children]
        op = self._type_compatible_op(max_arity=len(target.children), child_types=child_types)
        target.op_name = op.name
        return tree

    def _rule_subtree_crossover(self, t1: ExprNode = None, t2: ExprNode = None) -> ExprNode:
        if t1 is None:
            t1 = self.random_tree(3)
        if t2 is None:
            t2 = self.random_tree(3)
        t1, t2 = copy.deepcopy(t1), copy.deepcopy(t2)
        nodes1, nodes2 = self._collect_nodes(t1), self._collect_nodes(t2)
        if nodes1 and nodes2:
            n1, n2 = random.choice(nodes1), random.choice(nodes2)
            n1.op_name, n1.children, n1.value = n2.op_name, n2.children, n2.value
        return t1

    def _rule_hoist(self, tree: ExprNode = None) -> ExprNode:
        if tree is None:
            tree = self.random_tree(3)
        nodes = self._collect_nodes(tree)
        inner = [n for n in nodes if n.children]
        return copy.deepcopy(random.choice(inner)) if inner else copy.deepcopy(tree)

    def _collect_nodes(self, node: ExprNode) -> List[ExprNode]:
        result = [node]
        for c in node.children:
            result.extend(self._collect_nodes(c))
        return result

    def mutate(self, tree: ExprNode) -> ExprNode:
        return random.choice(self._composition_rules[1:])(tree)

    def crossover(self, t1: ExprNode, t2: ExprNode) -> ExprNode:
        return self._rule_subtree_crossover(t1, t2)

    def add_rule(self, rule_fn: Callable):
        self._composition_rules.append(rule_fn)
        logger.info(f"Grammar expanded: +rule {rule_fn.__name__}")

    @property
    def num_rules(self) -> int:
        return len(self._composition_rules)

    @property
    def rule_names(self) -> List[str]:
        return [getattr(r, '__name__', str(r)) for r in self._composition_rules]


# ---------------------------------------------------------------------------
# 2b. CONDITIONAL GRAMMAR RULES (Mechanism 3 Tier 2: Adaptive Grammar)
# ---------------------------------------------------------------------------

class ConditionalGrammarRule:
    """
    A grammar rule with archive-state-dependent behavior (A.3 Shutt / A.4 VW).

    Wraps a base mutation operator with:
    - preconditions: callable(archive_state) -> bool — rule only fires when met
    - intensity: callable(archive_state) -> float — scales mutation strength
    - fallback: Optional[Callable] — used when preconditions fail

    This makes grammar rules adaptive: the same structural mutation (point mutate,
    crossover, hoist) behaves differently depending on the evolutionary state.
    For example, a conditional rule might increase mutation intensity when fitness
    plateaus, or restrict crossover depth when coverage is high.

    F_theo assessment: This is an F_eff improvement. Grammar rules are search
    operators; they determine which trees get constructed, not which trees can
    exist. Adaptive grammar rules improve search efficiency under resource
    constraints but do not expand the theoretical expressibility of ExprNode trees.
    """

    def __init__(self, name: str, base_rule: Callable,
                 preconditions: Callable = None,
                 intensity_fn: Callable = None,
                 fallback: Callable = None):
        self.name = name
        self.__name__ = name  # For compatibility with rule_fn.__name__
        self.base_rule = base_rule
        self.preconditions = preconditions or (lambda _: True)
        self.intensity_fn = intensity_fn or (lambda _: 1.0)
        self.fallback = fallback
        self._archive_state: dict = {}
        self._applications: int = 0
        self._activations: int = 0

    def set_archive_state(self, state: dict):
        """Update the archive state for precondition evaluation."""
        self._archive_state = state

    def __call__(self, tree: ExprNode = None) -> ExprNode:
        """
        Execute the conditional grammar rule.

        If preconditions are met, applies base_rule with intensity scaling.
        If not met, applies fallback (if any) or base_rule at intensity 1.0.
        """
        self._applications += 1
        if self.preconditions(self._archive_state):
            self._activations += 1
            intensity = self.intensity_fn(self._archive_state)
            # Apply base rule, potentially multiple times for intensity > 1
            result = tree
            for _ in range(max(1, int(intensity))):
                result = self.base_rule(result)
            return result
        elif self.fallback is not None:
            return self.fallback(tree)
        else:
            return self.base_rule(tree)

    @property
    def activation_rate(self) -> float:
        return self._activations / max(1, self._applications)

    def __repr__(self):
        return f"ConditionalGrammarRule({self.name}, act_rate={self.activation_rate:.2f})"


class GrammarRuleComposer:
    """
    Operadic composition at the grammar level (H.8 applied to Layer 2).

    Creates new mutation operators by composing existing ones in structured
    patterns. This is the grammar-level analog of HyperRule templates at
    the vocabulary level.

    Composition patterns:
    - sequential: apply rule_a then rule_b to the same tree
    - alternating: apply rule_a or rule_b based on tree structure
    - filtered: apply rule only to subtrees matching a predicate

    F_theo assessment: F_eff improvement only. Composed grammar rules reach
    the same trees as their components, just more efficiently.
    """

    def __init__(self, grammar: "GrammarLayer"):
        self.grammar = grammar
        self._composed_rules: List[ConditionalGrammarRule] = []

    def compose_sequential(self, rule_a: Callable, rule_b: Callable,
                           name: str = None) -> ConditionalGrammarRule:
        """
        Create a new rule: apply rule_a then rule_b.
        Operadic composition: (rule_a ; rule_b)(tree) = rule_b(rule_a(tree))
        """
        name = name or f"seq_{getattr(rule_a, '__name__', 'a')}_{getattr(rule_b, '__name__', 'b')}"

        def composed_rule(tree: ExprNode = None) -> ExprNode:
            intermediate = rule_a(tree)
            return rule_b(intermediate)

        rule = ConditionalGrammarRule(name=name, base_rule=composed_rule)
        self._composed_rules.append(rule)
        return rule

    def compose_depth_filtered(self, base_rule: Callable,
                               min_depth: int = 0, max_depth: int = 99,
                               name: str = None) -> ConditionalGrammarRule:
        """
        Create a rule that only applies the base_rule to trees within a depth range.
        Trees outside the range are returned unchanged.
        """
        name = name or f"depth_filtered_{getattr(base_rule, '__name__', 'r')}_{min_depth}_{max_depth}"

        def filtered_rule(tree: ExprNode = None) -> ExprNode:
            if tree is None:
                return base_rule(tree)
            d = tree.depth()
            if min_depth <= d <= max_depth:
                return base_rule(tree)
            return copy.deepcopy(tree)

        rule = ConditionalGrammarRule(name=name, base_rule=filtered_rule)
        self._composed_rules.append(rule)
        return rule

    def compose_intensity_adaptive(self, base_rule: Callable,
                                   name: str = None) -> ConditionalGrammarRule:
        """
        Create a rule that increases mutation intensity when fitness plateaus
        and decreases it when fitness is improving.

        Implements the Paribhasa C.1b principle at the grammar level:
        rule behavior adapts to context.
        """
        name = name or f"adaptive_{getattr(base_rule, '__name__', 'r')}"

        def intensity_from_state(state: dict) -> float:
            if state.get("fitness_plateau", False):
                return 2.0  # Double mutation intensity on plateau
            coverage = state.get("coverage", 0.0)
            if coverage > 0.6:
                return 1.5  # Moderate increase at high coverage
            return 1.0  # Normal intensity

        rule = ConditionalGrammarRule(
            name=name,
            base_rule=base_rule,
            intensity_fn=intensity_from_state,
        )
        self._composed_rules.append(rule)
        return rule

    @property
    def num_composed(self) -> int:
        return len(self._composed_rules)


# ---------------------------------------------------------------------------
# 3. META-GRAMMAR LAYER
# ---------------------------------------------------------------------------

class MetaRuleEntry:
    """
    A meta-rule annotated with specificity conditions (Paribhasa C.1b).

    Each meta-rule declares:
    - preconditions: a callable (archive_state_dict) -> bool
    - specificity: int, higher = more specific (wins ties)
    - base_priority: float, static priority weight
    - rule_fn: the actual meta-rule callable

    Mechanism 5 Tier 2 Enhancement (Learned Specificity):
    - EMA (exponential moving average) success tracking replaces simple
      success_rate = successes/applications. Recent outcomes weighted more.
    - Fitness delta tracking: records whether the archive improved after
      each rule application, not just whether the rule produced an output.
    - Adaptive base_priority: rules that consistently improve the archive
      get a priority boost; rules that consistently produce no-ops get penalized.
    """

    def __init__(self, name: str, rule_fn: Callable, preconditions: Callable = None,
                 specificity: int = 0, base_priority: float = 1.0,
                 ema_alpha: float = 0.3):
        self.name = name
        self.rule_fn = rule_fn
        self.preconditions = preconditions or (lambda _state: True)
        self.specificity = specificity
        self.base_priority = base_priority
        self._applications: int = 0
        self._successes: int = 0
        # Mechanism 5 Tier 2: EMA tracking
        self._ema_alpha = ema_alpha
        self._ema_success: float = 0.5  # Prior: 50% success rate
        self._fitness_deltas: List[float] = []  # Track archive improvement
        self._adaptive_bonus: float = 0.0  # Learned priority adjustment

    def matches(self, archive_state: dict) -> bool:
        """Check whether this rule's preconditions are met."""
        try:
            return self.preconditions(archive_state)
        except Exception:
            return False

    def score(self, archive_state: dict) -> float:
        """
        Compute the deterministic priority score for this meta-rule
        given the current archive state. Higher = selected first.

        Enhanced scoring (Mechanism 5 Tier 2):
        Score = specificity * 100 + base_priority + ema_bonus + adaptive_bonus

        The EMA bonus replaces the simple success_rate * 10, providing
        more responsive adaptation to recent performance.
        """
        ema_bonus = self._ema_success * 10
        return (self.specificity * 100 + self.base_priority
                + ema_bonus + self._adaptive_bonus)

    def record_outcome(self, success: bool, fitness_delta: float = 0.0):
        """
        Record the outcome of a rule application.

        Args:
            success: whether the rule produced a non-None result
            fitness_delta: change in archive best_fitness after application
                          (positive = improvement, negative = regression)
        """
        self._applications += 1
        if success:
            self._successes += 1
        # EMA update
        outcome = 1.0 if success else 0.0
        self._ema_success = (self._ema_alpha * outcome
                             + (1 - self._ema_alpha) * self._ema_success)
        # Fitness delta tracking
        self._fitness_deltas.append(fitness_delta)
        # Adaptive bonus: boost rules with positive fitness impact
        if len(self._fitness_deltas) >= 3:
            recent = self._fitness_deltas[-3:]
            avg_delta = sum(recent) / len(recent)
            if avg_delta > 0:
                self._adaptive_bonus = min(5.0, self._adaptive_bonus + 0.5)
            elif avg_delta < -0.01:
                self._adaptive_bonus = max(-3.0, self._adaptive_bonus - 0.3)

    @property
    def ema_success_rate(self) -> float:
        return self._ema_success

    @property
    def success_rate(self) -> float:
        return self._successes / max(1, self._applications)

    def __repr__(self):
        return (f"MetaRuleEntry({self.name}, spec={self.specificity}, "
                f"pri={self.base_priority}, ema={self._ema_success:.2f}, "
                f"adapt={self._adaptive_bonus:.1f})")


class RuleInteractionTracker:
    """
    Tracks pairwise interactions between meta-rules (Mechanism 5 Tier 2).

    Records which rule was applied before which, and whether the sequence
    produced fitness improvement. This enables the system to learn
    productive rule sequences (e.g., "compose_new_op followed by
    parameterize_mutation tends to improve fitness").

    Used by MetaGrammarLayer to break ties when multiple rules have
    similar scores.
    """

    def __init__(self, max_history: int = 50):
        self._history: List[Tuple[str, float]] = []  # (rule_name, fitness_after)
        self._pair_scores: Dict[Tuple[str, str], List[float]] = {}
        self.max_history = max_history

    def record(self, rule_name: str, fitness_after: float):
        """Record a rule application and compute pairwise interactions."""
        if self._history:
            prev_name, prev_fitness = self._history[-1]
            pair = (prev_name, rule_name)
            delta = fitness_after - prev_fitness
            if pair not in self._pair_scores:
                self._pair_scores[pair] = []
            self._pair_scores[pair].append(delta)
            # Keep bounded
            if len(self._pair_scores[pair]) > self.max_history:
                self._pair_scores[pair] = self._pair_scores[pair][-self.max_history:]
        self._history.append((rule_name, fitness_after))
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def pair_score(self, prev_rule: str, candidate_rule: str) -> float:
        """
        Return the average fitness delta for the pair (prev_rule, candidate_rule).
        Returns 0.0 if no data available.
        """
        pair = (prev_rule, candidate_rule)
        deltas = self._pair_scores.get(pair, [])
        if not deltas:
            return 0.0
        return sum(deltas) / len(deltas)

    def best_successor(self, prev_rule: str, candidates: List[str]) -> Optional[str]:
        """Return the candidate with the best historical pair score after prev_rule."""
        if not candidates:
            return None
        scored = [(c, self.pair_score(prev_rule, c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored[0][1] > 0 else None

    @property
    def num_pairs(self) -> int:
        return len(self._pair_scores)

    def summary(self) -> dict:
        return {
            "history_length": len(self._history),
            "tracked_pairs": self.num_pairs,
            "top_pairs": sorted(
                ((p, sum(d)/len(d)) for p, d in self._pair_scores.items() if d),
                key=lambda x: x[1], reverse=True
            )[:5],
        }


class MetaGrammarLayer:
    """
    Generates new grammar rules and vocabulary expansions at runtime.

    Meta-rule selection uses Paribhasa-inspired deterministic priority:
    1. Collect all meta-rules whose preconditions match the current archive state
    2. Score each by specificity * 100 + base_priority + success_rate_bonus
    3. Select the highest-scoring rule (deterministic, not random)
    """

    def __init__(self, vocab: VocabularyLayer, grammar: GrammarLayer,
                 library_learner: "LibraryLearner" = None):
        self.vocab = vocab
        self.grammar = grammar
        self.library_learner = library_learner
        self._meta_rules: List[MetaRuleEntry] = []
        self._expansion_history: List[str] = []
        # Mechanism 3 Tier 2: Grammar rule composer
        self.rule_composer = GrammarRuleComposer(grammar)
        # Mechanism 5 Tier 2: Rule interaction tracker
        self.interaction_tracker = RuleInteractionTracker()
        self._last_best_fitness: float = 0.0
        # Mechanism 3 (Error-guided): Store residual error distribution from best elite
        self._residual_errors: List[Tuple[float, float]] = []
        self._register_default_meta_rules()

    def _register_default_meta_rules(self):
        # Rule 1: Compose new op — fires when vocabulary is small or coverage is low
        self._meta_rules.append(MetaRuleEntry(
            name="_meta_compose_new_op",
            rule_fn=self._meta_compose_new_op,
            preconditions=lambda s: s.get("vocab_size", 0) < 30 or s.get("coverage", 0) < 0.5,
            specificity=1,
            base_priority=2.0,
        ))
        # Rule 2: Parameterize mutation — fires when fitness is plateauing
        self._meta_rules.append(MetaRuleEntry(
            name="_meta_parameterize_mutation",
            rule_fn=self._meta_parameterize_mutation,
            preconditions=lambda s: s.get("fitness_plateau", False) or s.get("coverage", 0) > 0.3,
            specificity=2,
            base_priority=1.5,
        ))
        # Rule 3: Polymorphic op creation — Mechanism 2 (Context-Dependent Eval)
        # Creates PolymorphicOps from existing unary ops, enabling context-dependent
        # dispatch. Fires when vocabulary has enough base ops and coverage is stalling.
        self._meta_rules.append(MetaRuleEntry(
            name="_meta_create_polymorphic_op",
            rule_fn=self._meta_create_polymorphic_op,
            preconditions=lambda s: s.get("vocab_size", 0) >= 12 and s.get("coverage", 0) < 0.6,
            specificity=2,
            base_priority=1.0,
        ))
        # Rule 4: Adaptive grammar rule creation — Mechanism 3 Tier 2
        # Composes existing grammar rules into new adaptive rules that
        # change behavior based on archive state. Fires when fitness plateaus.
        self._meta_rules.append(MetaRuleEntry(
            name="_meta_compose_grammar_rule",
            rule_fn=self._meta_compose_grammar_rule,
            preconditions=lambda s: (s.get("fitness_plateau", False)
                                     or s.get("grammar_rules", 0) < 6),
            specificity=3,
            base_priority=1.5,
        ))
        # Rule 5: Error-guided op creation — Mechanism 3 (Error-guided Meta-mutation)
        # Creates ops that specifically target the x-intervals where residual error
        # is highest. Fires when residual error data is available and fitness is stalling.
        self._meta_rules.append(MetaRuleEntry(
            name="_meta_error_guided_op",
            rule_fn=self._meta_error_guided_op,
            preconditions=lambda s: (len(self._residual_errors) > 0
                                     and s.get("fitness_plateau", False)
                                     and s.get("best_fitness", 0) > 0.1),
            specificity=2,
            base_priority=1.8,
        ))

    def _meta_create_polymorphic_op(self) -> Optional[PolymorphicOp]:
        """
        Mechanism 2 integration: create a PolymorphicOp from existing unary ops.

        Takes 2-4 unary ops and bundles them into a single PolymorphicOp that
        dispatches to different functions based on EvalContext.topo_key().
        This makes the SAME tree structure compute DIFFERENT functions
        depending on where in the tree the op appears.

        Sources: C.3 (Aramaic polysemy), C.4 (Cuneiform polyvalence),
                 G.6 (Topos-internal evaluation).
        """
        ops = self.vocab.all_ops()
        unary = [op for op in ops if op.arity == 1
                 and not isinstance(op, PolymorphicOp)
                 and not op.name.startswith("lib_")]
        if len(unary) < 2:
            return None

        # Select 2-4 unary ops to bundle
        n_variants = min(random.randint(2, 4), len(unary))
        selected = random.sample(unary, n_variants)

        # Build dispatch table keyed by topo_key (0-7)
        topo_table = {}
        for i, op in enumerate(selected):
            topo_table[i] = op.fn

        default_fn = selected[0].fn
        name_parts = "_or_".join(op.name for op in selected)
        new_name = f"poly_{name_parts}"

        if self.vocab.get(new_name) is not None:
            return None

        avg_cost = sum(op.cost for op in selected) / len(selected)
        poly_op = PolymorphicOp(
            name=new_name,
            arity=1,
            dispatch_table={},  # no external-context dispatch
            default_fn=default_fn,
            cost=avg_cost + 0.5,  # slight premium for polymorphism
            description=f"Polymorphic[topo]: {' | '.join(op.name for op in selected)}",
            topo_dispatch_table=topo_table,
        )
        self.vocab.register(poly_op)
        self._expansion_history.append(f"new_poly_op:{new_name}")
        logger.info(f"Meta-grammar: Created PolymorphicOp '{new_name}' "
                     f"with {n_variants} variants dispatching on topo_key")
        return poly_op

    def _meta_compose_grammar_rule(self) -> Optional[ConditionalGrammarRule]:
        """
        Mechanism 3 Tier 2: Create a new adaptive grammar rule by composing
        existing grammar operations.

        Uses GrammarRuleComposer to create one of:
        1. Sequential composition of two existing mutation rules
        2. Depth-filtered mutation (apply only to shallow/deep trees)
        3. Intensity-adaptive mutation (scale with archive state)

        The created rule is registered in the GrammarLayer, making it
        available for the evolutionary loop's mutate() operation.

        F_theo: NO change. This is F_eff improvement — composed search
        operators reach the same tree space more efficiently.
        """
        rules = self.grammar._composition_rules[1:]  # Skip _rule_grow
        if len(rules) < 2:
            return None

        # Choose composition strategy
        strategy = random.choice(["sequential", "depth_filtered", "intensity_adaptive"])

        if strategy == "sequential":
            rule_a, rule_b = random.sample(rules, 2)
            composed = self.rule_composer.compose_sequential(rule_a, rule_b)
            self.grammar.add_rule(composed)
            self._expansion_history.append(f"grammar_compose:sequential:{composed.name}")
            logger.info(f"Meta-grammar: Created sequential grammar rule '{composed.name}'")
            return composed

        elif strategy == "depth_filtered":
            base = random.choice(rules)
            min_d = random.choice([0, 1, 2])
            max_d = random.choice([3, 4, 5])
            composed = self.rule_composer.compose_depth_filtered(base, min_d, max_d)
            self.grammar.add_rule(composed)
            self._expansion_history.append(f"grammar_compose:depth_filtered:{composed.name}")
            logger.info(f"Meta-grammar: Created depth-filtered grammar rule '{composed.name}'")
            return composed

        else:  # intensity_adaptive
            base = random.choice(rules)
            composed = self.rule_composer.compose_intensity_adaptive(base)
            self.grammar.add_rule(composed)
            self._expansion_history.append(f"grammar_compose:intensity_adaptive:{composed.name}")
            logger.info(f"Meta-grammar: Created intensity-adaptive grammar rule '{composed.name}'")
            return composed

    def _meta_error_guided_op(self) -> Optional[PrimitiveOp]:
        """
        Mechanism 3: Error-guided Meta-mutation.

        Analyzes residual error distribution from the best elite and creates
        new ops that specifically target high-error x-intervals.

        Strategy:
        1. Find x-intervals with highest residual error
        2. Create bump/basis functions centered on those intervals
        3. These become new ops that the evolutionary loop can compose
           to patch the approximation where it's weakest.

        Avoids creating duplicate bumps near existing ones (within 0.5 distance).
        """
        if not self._residual_errors:
            return None

        # Get worst error points
        worst = [e for e in self._residual_errors if e[1] >= 0.05]
        if not worst:
            return None

        # Find existing bump centers to avoid duplicates
        existing_centers = []
        for op in self.vocab.all_ops():
            if op.name.startswith("bump_x"):
                try:
                    center_str = op.name.split("_x")[1].split("_w")[0]
                    existing_centers.append(float(center_str))
                except (IndexError, ValueError):
                    pass

        # Find the worst point that's not near an existing bump
        x_center = None
        error_mag = None
        for x, err in worst:
            if not any(abs(x - c) < 0.5 for c in existing_centers):
                x_center = x
                error_mag = err
                break

        if x_center is None:
            return None  # All high-error points already covered

        # Create a localized correction basis function: gaussian bump
        width = 0.8
        new_name = f"bump_x{x_center:.2f}_w{width:.1f}"

        if self.vocab.get(new_name) is not None:
            return None

        def bump_fn(a, _c=x_center, _w=width):
            diff = a - _c
            return math.exp(-(diff * diff) / (2.0 * _w * _w))

        new_op = PrimitiveOp(
            name=new_name, arity=1, fn=bump_fn, cost=1.5,
            description=f"Error-guided bump at x={x_center:.2f} (err={error_mag:.3f})"
        )
        self.vocab.register(new_op)
        self._expansion_history.append(f"error_guided:{new_name}")
        logger.info(f"Meta-grammar: Error-guided op '{new_name}' "
                     f"targeting x={x_center:.2f} (residual={error_mag:.3f})")
        return new_op

    def update_residual_errors(self, errors: List[Tuple[float, float]]):
        """Update the residual error distribution from the current best elite."""
        self._residual_errors = errors

    def _meta_compose_new_op(self) -> Optional[PrimitiveOp]:
        """
        Operadic Meta-Grammar (H.8 Operads / A.4 VW Grammars).

        Instead of randomly chaining two unary ops, uses HyperRule templates
        that systematically generate new operations via consistent substitution.

        Templates encode arity constraints and structural patterns:
        - unary_chain: f(g(x)) — classic composition
        - binary_partial_left: h(c, x) — partial application with constant
        - binary_partial_right: h(x, c) — partial application with constant
        - binary_lift: h(f(x), g(x)) — parallel application then combine
        """
        templates = self._get_hyper_rule_templates()
        # Try templates in priority order (most specific first)
        for template in templates:
            result = self._apply_hyper_rule(template)
            if result is not None:
                return result
        return None

    def _get_hyper_rule_templates(self) -> List[dict]:
        """
        Return HyperRule templates ordered by specificity.

        Each template defines:
        - name: template identifier
        - arity_constraint: required arities of operand ops
        - build_fn: (ops_list, vocab) -> Optional[PrimitiveOp]
        - specificity: higher = tried first
        """
        ops = self.vocab.all_ops()
        unary = [op for op in ops if op.arity == 1 and not isinstance(op, PolymorphicOp)]
        binary = [op for op in ops if op.arity == 2 and not isinstance(op, PolymorphicOp)]
        nullary = [op for op in ops if op.arity == 0]

        templates = []

        # Template 1: binary_lift — h(f(x), g(x)) — highest specificity
        if binary and len(unary) >= 2:
            templates.append({
                "name": "binary_lift",
                "specificity": 3,
                "ops_pool": (binary, unary),
                "build": self._build_binary_lift,
            })

        # Template 2: binary_partial_left — h(c, x) with constant
        if binary and nullary:
            templates.append({
                "name": "binary_partial_left",
                "specificity": 2,
                "ops_pool": (binary, nullary),
                "build": self._build_binary_partial_left,
            })

        # Template 3: binary_partial_right — h(x, c) with constant
        if binary and nullary:
            templates.append({
                "name": "binary_partial_right",
                "specificity": 2,
                "ops_pool": (binary, nullary),
                "build": self._build_binary_partial_right,
            })

        # Template 4: unary_chain — f(g(x)) — lowest specificity (legacy)
        if len(unary) >= 2:
            templates.append({
                "name": "unary_chain",
                "specificity": 1,
                "ops_pool": (unary,),
                "build": self._build_unary_chain,
            })

        # Sort by specificity descending
        templates.sort(key=lambda t: t["specificity"], reverse=True)
        return templates

    def _apply_hyper_rule(self, template: dict) -> Optional[PrimitiveOp]:
        """Try to apply a HyperRule template to produce a new op."""
        return template["build"](template["ops_pool"])

    def _build_unary_chain(self, ops_pool: tuple) -> Optional[PrimitiveOp]:
        """f(g(x)) — chain two unary ops."""
        unary = ops_pool[0]
        if len(unary) < 2:
            return None
        op1, op2 = random.sample(unary, 2)
        new_name = f"{op1.name}_then_{op2.name}"
        if self.vocab.get(new_name) is not None:
            return None
        new_fn = lambda a, _o1=op1, _o2=op2: _o2(_o1(a))
        new_op = PrimitiveOp(new_name, 1, new_fn, op1.cost + op2.cost,
                             f"HyperRule[unary_chain]: {op1.name} -> {op2.name}")
        self.vocab.register(new_op)
        self._expansion_history.append(f"new_op:{new_name}")
        return new_op

    def _build_binary_lift(self, ops_pool: tuple) -> Optional[PrimitiveOp]:
        """h(f(x), g(x)) — apply two unary ops in parallel, combine with binary."""
        binary, unary = ops_pool
        h = random.choice(binary)
        f, g = random.sample(unary, 2)
        new_name = f"{h.name}_of_{f.name}_and_{g.name}"
        if self.vocab.get(new_name) is not None:
            return None
        new_fn = lambda a, _h=h, _f=f, _g=g: _h(_f(a), _g(a))
        new_op = PrimitiveOp(new_name, 1, new_fn, h.cost + f.cost + g.cost,
                             f"HyperRule[binary_lift]: {h.name}({f.name}(x), {g.name}(x))")
        self.vocab.register(new_op)
        self._expansion_history.append(f"new_op:{new_name}")
        return new_op

    def _build_binary_partial_left(self, ops_pool: tuple) -> Optional[PrimitiveOp]:
        """h(c, x) — partial application with a constant on the left."""
        binary, nullary = ops_pool
        h = random.choice(binary)
        c = random.choice(nullary)
        new_name = f"{h.name}_with_{c.name}_left"
        if self.vocab.get(new_name) is not None:
            return None
        c_val = c()
        new_fn = lambda a, _h=h, _c=c_val: _h(_c, a)
        new_op = PrimitiveOp(new_name, 1, new_fn, h.cost + c.cost,
                             f"HyperRule[partial_left]: {h.name}({c.name}, x)")
        self.vocab.register(new_op)
        self._expansion_history.append(f"new_op:{new_name}")
        return new_op

    def _build_binary_partial_right(self, ops_pool: tuple) -> Optional[PrimitiveOp]:
        """h(x, c) — partial application with a constant on the right."""
        binary, nullary = ops_pool
        h = random.choice(binary)
        c = random.choice(nullary)
        new_name = f"{h.name}_with_{c.name}_right"
        if self.vocab.get(new_name) is not None:
            return None
        c_val = c()
        new_fn = lambda a, _h=h, _c=c_val: _h(a, _c)
        new_op = PrimitiveOp(new_name, 1, new_fn, h.cost + c.cost,
                             f"HyperRule[partial_right]: {h.name}(x, {c.name})")
        self.vocab.register(new_op)
        self._expansion_history.append(f"new_op:{new_name}")
        return new_op

    def _meta_parameterize_mutation(self) -> Optional[Callable]:
        scale = random.uniform(0.5, 2.0)

        def scaled_mutate(tree: ExprNode = None) -> ExprNode:
            if tree is None:
                tree = self.grammar.random_tree(2)
            tree = copy.deepcopy(tree)
            nodes = self.grammar._collect_nodes(tree)
            for _ in range(max(1, int(len(nodes) * scale * 0.3))):
                target = random.choice(nodes)
                op = self.vocab.random_op(max_arity=len(target.children))
                target.op_name = op.name
            return tree

        scaled_mutate.__name__ = f"scaled_mutate_{scale:.2f}"
        self.grammar.add_rule(scaled_mutate)
        self._expansion_history.append(f"new_rule:{scaled_mutate.__name__}")
        return scaled_mutate

    def _compute_archive_state(self, archive=None, elite_trees: List[ExprNode] = None) -> dict:
        """
        Compute the current archive state for meta-rule precondition evaluation.
        This is the 'context' that Paribhasa rules match against.
        """
        state = {
            "vocab_size": self.vocab.size,
            "grammar_rules": self.grammar.num_rules,
            "expansion_count": len(self._expansion_history),
            "coverage": 0.0,
            "best_fitness": 0.0,
            "fitness_plateau": False,
            "has_elite_trees": bool(elite_trees),
        }
        if archive is not None:
            state["coverage"] = archive.coverage
            state["best_fitness"] = archive.best_fitness
            # Detect fitness plateau: last 3 expansions produced no improvement
            if len(self._expansion_history) >= 3:
                recent = self._expansion_history[-3:]
                state["fitness_plateau"] = all("no-op" in h or "success" not in h for h in recent)
        return state

    def expand_design_space(self, elite_trees: List[ExprNode] = None,
                            archive=None) -> str:
        """
        Expand the design space using deterministic Paribhasa-style selection.

        Enhanced with Mechanism 5 Tier 2 (Learned Specificity):
        1. If elite_trees and library_learner available, evaluate library learning
           as a candidate alongside meta-rules.
        2. Compute archive state for precondition matching.
        3. Filter meta-rules to those whose preconditions match.
        4. Score using EMA + adaptive bonus + interaction bonus.
        5. Select the highest-scoring rule (deterministic, not random).
        6. Record outcome with fitness delta for adaptive learning.
        7. Update ConditionalGrammarRule archive states.
        """
        state = self._compute_archive_state(archive=archive, elite_trees=elite_trees)
        current_best = state.get("best_fitness", 0.0)

        # Update archive state for all ConditionalGrammarRules in grammar
        for rule in self.grammar._composition_rules:
            if isinstance(rule, ConditionalGrammarRule):
                rule.set_archive_state(state)

        # Library learning gets highest specificity when elite trees are available
        if elite_trees and self.library_learner is not None:
            new_ops = self.library_learner.extract_library(elite_trees)
            if new_ops:
                names = [op.name for op in new_ops]
                self._expansion_history.append(f"library_learning:{','.join(names)}")
                action = f"Library learning: extracted {len(new_ops)} new primitives"
                logger.info(f"Meta-grammar: {action}")
                # Track interaction
                self.interaction_tracker.record("library_learning", current_best)
                self._last_best_fitness = current_best
                return action

        # Deterministic Paribhasa selection: match preconditions, rank by score
        matching_rules = [r for r in self._meta_rules if r.matches(state)]
        if not matching_rules:
            # Fallback: all rules are candidates if none match
            matching_rules = self._meta_rules

        # Mechanism 5 Tier 2: Add interaction bonus to scoring
        prev_rule = (self.interaction_tracker._history[-1][0]
                     if self.interaction_tracker._history else None)

        def enhanced_score(r):
            base = r.score(state)
            if prev_rule:
                interaction = self.interaction_tracker.pair_score(prev_rule, r.name)
                base += interaction * 5  # Weight interaction history
            return base

        # Sort by enhanced score descending — deterministic selection
        matching_rules.sort(key=enhanced_score, reverse=True)
        selected = matching_rules[0]

        result = selected.rule_fn()
        success = result is not None

        # Compute fitness delta for adaptive learning
        fitness_delta = current_best - self._last_best_fitness
        selected.record_outcome(success, fitness_delta=fitness_delta)
        self.interaction_tracker.record(selected.name, current_best)
        self._last_best_fitness = current_best

        action = (f"Applied {selected.name} "
                  f"(score={enhanced_score(selected):.1f}, "
                  f"ema={selected.ema_success_rate:.2f}): "
                  f"{'success' if success else 'no-op'}")
        self._expansion_history.append(action)
        logger.info(f"Meta-grammar: {action}")
        return action

    @property
    def expansion_count(self) -> int:
        return len(self._expansion_history)


# ---------------------------------------------------------------------------
# 3b. LIBRARY LEARNING (DreamCoder-inspired subtree compression)
# ---------------------------------------------------------------------------

class LibraryLearner:
    """
    Extracts frequently occurring subtrees from elite programs and promotes
    them to new primitive operations in the vocabulary.

    Inspired by DreamCoder's wake-sleep library learning / compression phase.
    This genuinely expands the reachable design space because:
    - A subtree of depth D becomes a single node (depth 0)
    - Under a fixed max_depth constraint, programs that previously required
      depth max_depth + D are now reachable at depth max_depth
    - New primitives are semantically meaningful (discovered from successful
      programs, not randomly composed)

    The mechanism is structurally different from MetaGrammarLayer._meta_compose_new_op
    which only randomly chains two existing unary ops. Library learning:
    1. Considers subtrees of ANY arity and depth
    2. Selects based on frequency in the elite population (not random)
    3. Can discover multi-step computations involving binary ops, constants, etc.
    """

    def __init__(
        self,
        vocab: VocabularyLayer,
        min_subtree_depth: int = 2,
        min_frequency: int = 2,
        max_library_additions: int = 3,
    ):
        self.vocab = vocab
        self.min_subtree_depth = min_subtree_depth
        self.min_frequency = min_frequency
        self.max_library_additions = max_library_additions
        self._learned_ops: List[str] = []

    def _collect_subtrees(self, node: ExprNode) -> List[ExprNode]:
        """Collect all subtrees from a tree (including the root)."""
        result = [node]
        for c in node.children:
            result.extend(self._collect_subtrees(c))
        return result

    def _subtree_to_callable(self, subtree: ExprNode) -> Tuple[int, Callable]:
        """
        Convert a subtree into a callable function.

        Returns (arity, fn) where arity is the number of distinct input_x
        leaves found. For subtrees with input_x, arity=1 (single variable).
        For subtrees without input_x (pure constants), arity=0.
        """
        has_input = self._has_input_x(subtree)
        arity = 1 if has_input else 0

        def _eval_subtree(*args):
            x_val = args[0] if args else 0.0
            return self._eval_subtree_node(subtree, x_val)

        return arity, _eval_subtree

    def _has_input_x(self, node: ExprNode) -> bool:
        if node.op_name == "input_x":
            return True
        return any(self._has_input_x(c) for c in node.children)

    def _eval_subtree_node(self, node: ExprNode, x: float) -> float:
        """Evaluate a subtree at input x, using vocabulary ops."""
        if node.op_name == "input_x":
            return x
        op = self.vocab.get(node.op_name)
        if op is None:
            return 0.0
        if op.arity == 0:
            try:
                return float(op())
            except Exception:
                return 0.0
        child_vals = [self._eval_subtree_node(c, x) for c in node.children]
        if len(child_vals) < op.arity:
            child_vals.extend([0.0] * (op.arity - len(child_vals)))
        try:
            result = op(*child_vals[:op.arity])
            return float(result) if math.isfinite(float(result)) else 0.0
        except Exception:
            return 0.0

    def extract_library(self, elite_trees: List[ExprNode]) -> List[PrimitiveOp]:
        """
        Scan elite trees for recurring subtrees and promote them to primitives.

        Algorithm:
        1. Collect all subtrees of depth >= min_subtree_depth from all elites
        2. Group by structural fingerprint
        3. Filter by frequency >= min_frequency
        4. Sort by frequency * depth (prefer frequent, deep subtrees)
        5. Create new PrimitiveOps for top candidates
        """
        # Step 1-2: Collect and group subtrees by fingerprint
        fingerprint_counts: Dict[str, Tuple[int, ExprNode]] = {}
        for tree in elite_trees:
            seen_in_tree = set()  # avoid double-counting within one tree
            for sub in self._collect_subtrees(tree):
                if sub.depth() >= self.min_subtree_depth:
                    fp = sub.fingerprint()
                    if fp not in seen_in_tree:
                        seen_in_tree.add(fp)
                        if fp in fingerprint_counts:
                            count, exemplar = fingerprint_counts[fp]
                            fingerprint_counts[fp] = (count + 1, exemplar)
                        else:
                            fingerprint_counts[fp] = (1, copy.deepcopy(sub))

        # Step 3: Filter by frequency
        candidates = [
            (count, exemplar)
            for fp, (count, exemplar) in fingerprint_counts.items()
            if count >= self.min_frequency
        ]

        # Step 4: Sort by frequency * depth (compressed value heuristic)
        candidates.sort(key=lambda c: c[0] * c[1].depth(), reverse=True)

        # Step 5: Create new PrimitiveOps
        new_ops = []
        for count, subtree in candidates[: self.max_library_additions]:
            fp = subtree.fingerprint()
            lib_name = f"lib_{fp}"
            if self.vocab.get(lib_name) is not None:
                continue  # Already extracted this subtree

            arity, fn = self._subtree_to_callable(subtree)
            cost = subtree.size() * 0.5  # Discounted cost (library ops are optimized)
            new_op = PrimitiveOp(
                name=lib_name,
                arity=arity,
                fn=fn,
                cost=cost,
                description=f"Library-learned: depth={subtree.depth()}, size={subtree.size()}, freq={count}",
            )
            self.vocab.register(new_op)
            self._learned_ops.append(lib_name)
            new_ops.append(new_op)
            logger.info(
                f"Library learning: extracted '{lib_name}' "
                f"(arity={arity}, depth={subtree.depth()}, freq={count})"
            )

        return new_ops

    @property
    def num_learned(self) -> int:
        return len(self._learned_ops)


# ---------------------------------------------------------------------------
# 4. PHYSICAL COST GROUNDING
# ---------------------------------------------------------------------------

@dataclass
class ResourceBudget:
    """Tracks and enforces physical resource constraints."""
    max_compute_ops: int = 100_000
    max_memory_bytes: int = 50 * 1024 * 1024
    max_wall_seconds: float = 60.0
    _compute_used: int = 0
    _peak_memory: int = 0
    _start_time: float = field(default_factory=time.time)

    def reset(self):
        self._compute_used = 0
        self._peak_memory = 0
        self._start_time = time.time()

    def tick(self, ops: int = 1):
        self._compute_used += ops

    @property
    def compute_fraction(self) -> float:
        return self._compute_used / self.max_compute_ops

    @property
    def time_fraction(self) -> float:
        return (time.time() - self._start_time) / self.max_wall_seconds

    @property
    def is_exhausted(self) -> bool:
        return (
            self._compute_used >= self.max_compute_ops
            or (time.time() - self._start_time) >= self.max_wall_seconds
        )

    def cost_score(self) -> float:
        return 1.0 / (1.0 + self.compute_fraction + self.time_fraction)

    def summary(self) -> dict:
        return {
            "compute_used": self._compute_used,
            "wall_seconds": round(time.time() - self._start_time, 3),
            "cost_score": round(self.cost_score(), 4),
        }


class CostGroundingLoop:
    """Evaluates candidates under physical cost constraints."""

    def __init__(self, budget: ResourceBudget):
        self.budget = budget

    def evaluate_with_cost(
        self, tree: ExprNode, vocab: VocabularyLayer, fitness_fn: Callable,
        ctx: EvalContext = None
    ) -> Tuple[float, float, float]:
        self.budget.reset()
        if ctx is not None:
            raw = fitness_fn(tree, vocab, ctx=ctx)
        else:
            raw = fitness_fn(tree, vocab)
        self.budget.tick(tree.size() * 10)
        cost = self.budget.cost_score()
        return raw, cost, raw * cost


# ---------------------------------------------------------------------------
# 5. MAP-ELITES ARCHIVE
# ---------------------------------------------------------------------------

@dataclass
class EliteEntry:
    """An entry in the MAP-Elites archive."""
    tree: ExprNode
    raw_fitness: float
    cost_score: float
    grounded_fitness: float
    behavior: Tuple[int, ...]
    generation: int


class MAPElitesArchive:
    """Multi-dimensional MAP-Elites archive for quality-diversity search."""

    def __init__(self, dims: List[int]):
        self.dims = dims
        self._grid: Dict[Tuple[int, ...], EliteEntry] = {}
        self._total_tried = 0
        self._total_inserted = 0

    def behavior_descriptor(self, tree: ExprNode) -> Tuple[int, ...]:
        return (min(tree.depth(), self.dims[0] - 1), min(tree.size() // 3, self.dims[1] - 1))

    def try_insert(self, entry: EliteEntry) -> bool:
        self._total_tried += 1
        cell = entry.behavior
        if cell not in self._grid or entry.grounded_fitness > self._grid[cell].grounded_fitness:
            self._grid[cell] = entry
            self._total_inserted += 1
            return True
        return False

    def sample_parent(self) -> Optional[EliteEntry]:
        return random.choice(list(self._grid.values())) if self._grid else None

    @property
    def coverage(self) -> float:
        total = 1
        for d in self.dims:
            total *= d
        return len(self._grid) / total

    @property
    def best_fitness(self) -> float:
        return max((e.grounded_fitness for e in self._grid.values()), default=0.0)

    def summary(self) -> dict:
        return {
            "filled_cells": len(self._grid),
            "total_cells": int(np.prod(self.dims)),
            "coverage": round(self.coverage, 4),
            "best_fitness": round(self.best_fitness, 4),
            "total_tried": self._total_tried,
            "total_inserted": self._total_inserted,
        }



class NoveltyScreener:
    """
    Fingerprint-based novelty rejection sampling for MAP-Elites archives.

    Inspired by the NoveltyJudge in b-albar/evolve-anything, which uses
    embedding-based similarity scoring with rejection sampling to prevent
    the archive from filling with near-duplicate solutions.

    This adaptation works with expression tree fingerprints instead of
    code embeddings, computing structural Jaccard similarity between
    candidates and existing archive members. Candidates above a
    similarity threshold are rejected, forcing the search to explore
    structurally novel regions of the design space at runtime.
    """

    def __init__(self, similarity_threshold: float = 0.85, max_attempts: int = 3):
        self.similarity_threshold = similarity_threshold
        self.max_attempts = max_attempts
        self._rejections = 0
        self._screenings = 0

    def _subtree_fingerprints(self, node: ExprNode) -> set:
        """Collect fingerprints of all subtrees in a tree."""
        fps = set()
        fps.add(node.fingerprint())
        for child in node.children:
            fps.update(self._subtree_fingerprints(child))
        return fps

    def structural_similarity(self, tree_a: ExprNode, tree_b: ExprNode) -> float:
        """
        Compute Jaccard similarity between two trees based on their
        subtree fingerprint sets. Returns a value in [0, 1].
        """
        fps_a = self._subtree_fingerprints(tree_a)
        fps_b = self._subtree_fingerprints(tree_b)
        if not fps_a and not fps_b:
            return 1.0
        intersection = fps_a & fps_b
        union = fps_a | fps_b
        return len(intersection) / len(union) if union else 1.0

    def max_similarity_to_archive(
        self, candidate: ExprNode, archive_entries: List[EliteEntry]
    ) -> float:
        """Return the maximum similarity between a candidate and all archive entries."""
        if not archive_entries:
            return 0.0
        return max(
            self.structural_similarity(candidate, entry.tree)
            for entry in archive_entries
        )

    def should_accept(
        self, candidate: ExprNode, archive_entries: List[EliteEntry]
    ) -> bool:
        """
        Screen a candidate for novelty. Returns True if the candidate is
        sufficiently different from existing archive members.
        """
        self._screenings += 1
        max_sim = self.max_similarity_to_archive(candidate, archive_entries)
        if max_sim <= self.similarity_threshold:
            return True
        self._rejections += 1
        return False

    @property
    def rejection_rate(self) -> float:
        return self._rejections / self._screenings if self._screenings > 0 else 0.0

    def summary(self) -> dict:
        return {
            "screenings": self._screenings,
            "rejections": self._rejections,
            "rejection_rate": round(self.rejection_rate, 4),
        }


class EnhancedMAPElitesArchive(MAPElitesArchive):
    """
    Extends MAPElitesArchive with three coverage-ceiling mitigations:

    1. Wider behavioral grid (dims [8, 12] by default) for finer-grained
       structural diversity.
    2. Novelty injection: sub-optimal candidates that occupy an *empty*
       neighbor cell are accepted with probability `novelty_rate`. This
       prevents premature convergence and allows the archive to keep
       expanding into unexplored behavioral niches.
    3. Novelty rejection sampling (from b-albar/evolve-anything): candidates
       that are structurally too similar to existing archive members are
       rejected before insertion. This forces the evolutionary search to
       produce genuinely novel structures rather than minor variants,
       expanding the effective search space at runtime.

    Empirical results (50 gen x 20 pop):
      Standard [6,10]:   coverage=0.3333
      Enhanced  [8,12]:  coverage=0.3854-0.4375 depending on domain
    """

    def __init__(
        self,
        dims: List[int] = None,
        novelty_rate: float = 0.15,
        similarity_threshold: float = 0.85,
    ):
        super().__init__(dims or [8, 12])
        self.novelty_rate = novelty_rate
        self._novelty_inserts = 0
        self.novelty_screener = NoveltyScreener(
            similarity_threshold=similarity_threshold
        )

    def try_insert(self, entry: EliteEntry) -> bool:
        self._total_tried += 1
        cell = entry.behavior
        # --- Novelty rejection sampling ---
        # If the cell is already occupied and the candidate is too similar
        # to existing archive members, reject it to force exploration.
        if cell in self._grid:
            archive_entries = list(self._grid.values())
            if not self.novelty_screener.should_accept(entry.tree, archive_entries):
                return False

        # Standard elitism
        if cell not in self._grid:
            self._grid[cell] = entry
            self._total_inserted += 1
            return True

        if entry.grounded_fitness > self._grid[cell].grounded_fitness:
            self._grid[cell] = entry
            self._total_inserted += 1
            return True

        # Novelty injection into empty neighbor cells
        if random.random() < self.novelty_rate:
            neighbor = self._find_empty_neighbor(cell)
            if neighbor is not None:
                self._grid[neighbor] = entry
                self._total_inserted += 1
                self._novelty_inserts += 1
                return True

        return False

    def _find_empty_neighbor(self, cell: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        candidates = []
        for d_idx in range(len(cell)):
            for delta in (-1, 1):
                neighbor = list(cell)
                neighbor[d_idx] = max(0, min(self.dims[d_idx] - 1, cell[d_idx] + delta))
                t = tuple(neighbor)
                if t != cell and t not in self._grid:
                    candidates.append(t)
        return random.choice(candidates) if candidates else None

    def summary(self) -> dict:
        s = super().summary()
        s["novelty_inserts"] = self._novelty_inserts
        s["novelty_screening"] = self.novelty_screener.summary()
        return s

# ---------------------------------------------------------------------------
# 6. SELF-IMPROVEMENT ENGINE
# ---------------------------------------------------------------------------

class SelfImprovementEngine:
    """
    Outer loop inspired by the Darwin Godel Machine.

    Each generation:
    1. Select parents from the MAP-Elites archive
    2. Apply grammar mutations (possibly from meta-grammar expansions)
    3. Evaluate with cost grounding
    4. Insert into archive if better
    5. Periodically expand the design space via meta-grammar
    """

    def __init__(
        self,
        vocab: VocabularyLayer,
        grammar: GrammarLayer,
        meta_grammar: MetaGrammarLayer,
        archive: MAPElitesArchive,
        cost_loop: CostGroundingLoop,
        fitness_fn: Callable,
        expansion_interval: int = 10,
        pruning_window: int = 20,
        pruning_threshold: float = 0.05,
        target_fn: Callable = None,
        target_xs: np.ndarray = None,
    ):
        self.vocab = vocab
        self.grammar = grammar
        self.meta_grammar = meta_grammar
        self.archive = archive
        self.cost_loop = cost_loop
        self.fitness_fn = fitness_fn
        self.expansion_interval = expansion_interval
        self.generation = 0
        self.history: List[dict] = []
        # Mechanism 1: Operator Pruning — track usage of generated ops in elites
        self.pruning_window = pruning_window  # generations between pruning checks
        self.pruning_threshold = pruning_threshold  # fraction of elites that must use an op
        self._op_usage_history: Dict[str, List[int]] = {}  # op_name -> [gen_last_seen_in_elite]
        # Mechanism 3: Error-guided meta-mutation — target function for residual analysis
        self._target_fn = target_fn
        self._target_xs = target_xs

    def _collect_tree_ops(self, node: ExprNode) -> set:
        """Collect all op names used in a tree."""
        ops = {node.op_name}
        for c in node.children:
            ops |= self._collect_tree_ops(c)
        return ops

    def _prune_unused_ops(self):
        """
        Mechanism 1: Operator Pruning.

        Scan elite archive for usage of dynamically generated ops.
        Remove any generated op that is not used by at least pruning_threshold
        fraction of elite trees. Ops created within the last pruning_window
        generations get a grace period (not pruned immediately).
        """
        gen_ops = set(self.vocab.generated_op_names())
        if not gen_ops:
            return

        elites = list(self.archive._grid.values())
        if not elites:
            return

        # Count how many elites use each generated op
        op_usage_count: Dict[str, int] = {op: 0 for op in gen_ops}
        for entry in elites:
            tree_ops = self._collect_tree_ops(entry.tree)
            for op in gen_ops:
                if op in tree_ops:
                    op_usage_count[op] += 1

        # Track first-seen generation for grace period
        for op in gen_ops:
            if op not in self._op_usage_history:
                self._op_usage_history[op] = self.generation

        # Prune ops below threshold (with grace period)
        threshold_count = max(1, int(len(elites) * self.pruning_threshold))
        pruned = []
        for op_name, usage in op_usage_count.items():
            age = self.generation - self._op_usage_history.get(op_name, 0)
            if usage < threshold_count and age >= self.pruning_window:
                if self.vocab.unregister(op_name):
                    pruned.append(op_name)
                    self._op_usage_history.pop(op_name, None)

        if pruned:
            logger.info(f"Operator pruning: removed {len(pruned)} unused ops: "
                         "%s" % ", ".join(pruned[:5]))

    def _update_error_guidance(self):
        """
        Mechanism 3: Feed residual errors from best elite to meta-grammar.
        """
        if self._target_fn is None or self._target_xs is None:
            return

        # Find best elite
        best_entry = max(self.archive._grid.values(),
                         key=lambda e: e.grounded_fitness, default=None)
        if best_entry is None:
            return

        ctx = EvalContext(self_fingerprint=best_entry.tree.fingerprint(),
                          env_tag="error_guide")
        errors = _compute_residual_errors(
            best_entry.tree, self.vocab, self._target_fn,
            self._target_xs, ctx
        )
        self.meta_grammar.update_residual_errors(errors)

    def _elite_subtree_compression(self):
        """
        Mechanism 4: Elite Sub-tree Compression (fitness-aware).

        Different from LibraryLearner (frequency-based):
        - Selects subtrees from TOP fitness elites only (top 20%)
        - Weights candidates by parent elite fitness, not just frequency
        - Ensures extracted ops come from proven high-fitness solutions
        """
        elites = sorted(self.archive._grid.values(),
                        key=lambda e: e.grounded_fitness, reverse=True)
        if len(elites) < 3:
            return

        # Only use top 20% of elites
        top_k = max(2, len(elites) // 5)
        top_elites = elites[:top_k]

        # Collect subtrees weighted by parent fitness
        subtree_scores: Dict[str, Tuple[float, ExprNode, int]] = {}
        for entry in top_elites:
            for sub in self._collect_subtrees(entry.tree):
                if sub.depth() >= 2 and sub.size() >= 3:
                    fp = sub.fingerprint()
                    if fp in subtree_scores:
                        score, exemplar, count = subtree_scores[fp]
                        subtree_scores[fp] = (
                            score + entry.grounded_fitness,
                            exemplar,
                            count + 1
                        )
                    else:
                        subtree_scores[fp] = (
                            entry.grounded_fitness,
                            copy.deepcopy(sub),
                            1
                        )

        # Filter: must appear in at least 2 top elites
        candidates = [
            (score, exemplar, count)
            for fp, (score, exemplar, count) in subtree_scores.items()
            if count >= 2
        ]
        if not candidates:
            return

        # Sort by fitness-weighted score (not just frequency)
        candidates.sort(key=lambda c: c[0] * c[2], reverse=True)

        # Extract top 2 as new ops
        extracted = 0
        for score, subtree, count in candidates[:2]:
            fp = subtree.fingerprint()
            lib_name = f"elite_{fp}"
            if self.vocab.get(lib_name) is not None:
                continue

            has_input = self._subtree_has_input(subtree)
            arity = 1 if has_input else 0

            def _make_eval_fn(st, voc):
                def _fn(*args):
                    x_val = args[0] if args else 0.0
                    return self._eval_subtree(st, voc, x_val)
                return _fn

            fn = _make_eval_fn(subtree, self.vocab)
            new_op = PrimitiveOp(
                name=lib_name, arity=arity, fn=fn,
                cost=subtree.size() * 0.4,
                description=f"Elite-compressed: depth={subtree.depth()}, "
                            f"fitness_score={score:.3f}, count={count}"
            )
            self.vocab.register(new_op)
            extracted += 1
            logger.info(f"Elite compression: '{lib_name}' "
                         f"(fitness_score={score:.3f}, count={count})")

    def _collect_subtrees(self, node: ExprNode) -> List[ExprNode]:
        result = [node]
        for c in node.children:
            result.extend(self._collect_subtrees(c))
        return result

    def _subtree_has_input(self, node: ExprNode) -> bool:
        if node.op_name == "input_x":
            return True
        return any(self._subtree_has_input(c) for c in node.children)

    def _eval_subtree(self, node: ExprNode, vocab: VocabularyLayer, x: float) -> float:
        if node.op_name == "input_x":
            return x
        op = vocab.get(node.op_name)
        if op is None:
            return 0.0
        if op.arity == 0:
            try:
                return float(op())
            except Exception:
                return 0.0
        child_vals = [self._eval_subtree(c, vocab, x) for c in node.children]
        if len(child_vals) < op.arity:
            child_vals.extend([0.0] * (op.arity - len(child_vals)))
        try:
            result = op(*child_vals[:op.arity])
            return float(result) if math.isfinite(float(result)) else 0.0
        except Exception:
            return 0.0

    def step(self, population_size: int = 20) -> dict:
        self.generation += 1
        inserted = 0
        best_gen = 0.0

        for _ in range(population_size):
            parent = self.archive.sample_parent()
            child = self.grammar.mutate(parent.tree) if parent else self.grammar.random_tree(3)
            raw, cost, grounded = self.cost_loop.evaluate_with_cost(child, self.vocab, self.fitness_fn)
            behavior = self.archive.behavior_descriptor(child)
            entry = EliteEntry(tree=child, raw_fitness=raw, cost_score=cost,
                               grounded_fitness=grounded, behavior=behavior, generation=self.generation)
            if self.archive.try_insert(entry):
                inserted += 1
            best_gen = max(best_gen, grounded)

        expansion_action = None
        if self.generation % self.expansion_interval == 0:
            # Mechanism 3: Update error guidance before expansion
            self._update_error_guidance()

            # Gather elite trees for library learning
            elite_trees = [e.tree for e in self.archive._grid.values()]
            expansion_action = self.meta_grammar.expand_design_space(
                elite_trees=elite_trees, archive=self.archive
            )

            # Mechanism 4: Elite sub-tree compression (every other expansion)
            if self.generation % (self.expansion_interval * 2) == 0:
                self._elite_subtree_compression()

        # Mechanism 1: Operator Pruning (every pruning_window generations)
        if self.generation % self.pruning_window == 0:
            self._prune_unused_ops()

        record = {
            "generation": self.generation,
            "inserted": inserted,
            "best_gen_fitness": round(best_gen, 4),
            "archive_coverage": round(self.archive.coverage, 4),
            "archive_best": round(self.archive.best_fitness, 4),
            "vocab_size": self.vocab.size,
            "grammar_rules": self.grammar.num_rules,
            "meta_expansions": self.meta_grammar.expansion_count,
            "expansion_action": expansion_action,
        }
        self.history.append(record)
        return record

    def run(self, generations: int = 50, population_size: int = 20) -> List[dict]:
        logger.info(f"Starting RSI loop: {generations} gen x {population_size} pop")
        for g in range(generations):
            record = self.step(population_size)
            if g % 10 == 0 or g == generations - 1:
                logger.info(
                    f"Gen {record['generation']:4d} | best={record['archive_best']:.4f} | "
                    f"cov={record['archive_coverage']:.4f} | vocab={record['vocab_size']} | "
                    f"rules={record['grammar_rules']} | expansions={record['meta_expansions']}"
                )
        return self.history


# ---------------------------------------------------------------------------
# 7. FITNESS FUNCTIONS
# ---------------------------------------------------------------------------

def _eval_tree(node: ExprNode, vocab: VocabularyLayer, x: float,
               ctx: EvalContext = None) -> float:
    """
    Recursively evaluate an expression tree at input x.

    If ctx is provided, enables:
    - Mechanism 1 (Self-Reference): 'self_encode' op returns the tree's own
      fingerprint as a numeric hash, enabling fixed-point computations.
    - Mechanism 2 (Context-Dependent Evaluation): PolymorphicOps dispatch to
      different functions based on context state.
    - Mechanism 3 (Topological Context, G.6 Topos): ctx carries tree-structural
      metadata (depth, parent_op, sibling_index) so dispatch depends on WHERE
      in the tree the op appears, not just what external context is active.
    """
    if node.op_name == "input_x":
        return x
    if node.op_name == "self_encode":
        if ctx and ctx.self_fingerprint:
            return (int(ctx.self_fingerprint[:8], 16) % 10000) / 10000.0
        return 0.0
    op = vocab.get(node.op_name)
    if op is None:
        return 0.0
    if op.arity == 0:
        try:
            return float(op())
        except Exception:
            return 0.0
    # Evaluate children with topological context threading
    child_vals = []
    for i, c in enumerate(node.children):
        if ctx is not None:
            child_ctx = ctx.with_topo(
                depth=ctx.current_depth + 1,
                parent_op=node.op_name,
                sib_idx=i,
                sub_size=c.size(),
            )
        else:
            child_ctx = None
        child_vals.append(_eval_tree(c, vocab, x, child_ctx))
    if len(child_vals) < op.arity:
        child_vals.extend([0.0] * (op.arity - len(child_vals)))
    try:
        if isinstance(op, PolymorphicOp) and ctx is not None:
            result = op(*child_vals[:op.arity], ctx=ctx)
        else:
            result = op(*child_vals[:op.arity])
        return float(result) if math.isfinite(float(result)) else 0.0
    except Exception:
        return 0.0


def _parsimony_penalty(tree: ExprNode, vocab: VocabularyLayer,
                       raw_fitness: float = 1.0,
                       alpha: float = 0.002, beta: float = 0.001) -> float:
    """
    Mechanism 2: Dynamic parsimony pressure.

    Penalizes tree complexity (node count) and vocabulary bloat.
    Returns a value in [0, 1) that is subtracted from raw fitness.

    The penalty scales with raw_fitness so it only becomes significant
    for already-good solutions (tie-breaking), not for weak ones.

    penalty = raw_fitness * (alpha * node_count + beta * vocab_excess)

    where base_vocab_size = 13 (number of default ops including self_encode).
    The penalty is clamped to [0, 0.1] to avoid excessive reduction.
    """
    node_count = tree.size()
    vocab_excess = max(0, vocab.size - 13)  # 13 default ops
    raw_penalty = alpha * node_count + beta * vocab_excess
    penalty = raw_fitness * raw_penalty
    return min(penalty, 0.1)


def _compute_residual_errors(tree: ExprNode, vocab: VocabularyLayer,
                             target_fn: Callable, xs: np.ndarray,
                             ctx: EvalContext = None) -> List[Tuple[float, float]]:
    """
    Mechanism 3: Compute per-point residual errors for error-guided meta-mutation.

    Returns list of (x, residual_error) tuples sorted by error descending.
    """
    errors = []
    for x in xs:
        try:
            predicted = _eval_tree(tree, vocab, x, ctx)
            target = target_fn(x)
            errors.append((x, abs(predicted - target)))
        except Exception:
            errors.append((x, 1e6))
    errors.sort(key=lambda e: e[1], reverse=True)
    return errors


def symbolic_regression_fitness(tree: ExprNode, vocab: VocabularyLayer,
                                ctx: EvalContext = None) -> float:
    """Target: f(x) = x^2 + 2x + 1  over [-5, 5]."""
    if ctx is None:
        ctx = EvalContext(self_fingerprint=tree.fingerprint(), env_tag="symreg")
    xs = np.linspace(-5, 5, 20)
    error = sum(abs(_eval_tree(tree, vocab, x, ctx) - (x**2 + 2*x + 1)) for x in xs)
    raw = 1.0 / (1.0 + min(error / len(xs), 1e6))
    return max(0.0, raw - _parsimony_penalty(tree, vocab, raw))


def sine_approximation_fitness(tree: ExprNode, vocab: VocabularyLayer,
                               ctx: EvalContext = None) -> float:
    """Target: f(x) = sin(x)  over [-pi, pi]."""
    if ctx is None:
        ctx = EvalContext(self_fingerprint=tree.fingerprint(), env_tag="sine")
    xs = np.linspace(-math.pi, math.pi, 30)
    error = 0.0
    for x in xs:
        try:
            error += abs(_eval_tree(tree, vocab, x, ctx) - math.sin(x))
        except Exception:
            error += 1e6
    raw = 1.0 / (1.0 + min(error / len(xs), 1e6))
    return max(0.0, raw - _parsimony_penalty(tree, vocab, raw))


def absolute_value_fitness(tree: ExprNode, vocab: VocabularyLayer,
                           ctx: EvalContext = None) -> float:
    """Target: f(x) = |x|  over [-5, 5]."""
    if ctx is None:
        ctx = EvalContext(self_fingerprint=tree.fingerprint(), env_tag="absval")
    xs = np.linspace(-5, 5, 30)
    error = 0.0
    for x in xs:
        try:
            error += abs(_eval_tree(tree, vocab, x, ctx) - abs(x))
        except Exception:
            error += 1e6
    raw = 1.0 / (1.0 + min(error / len(xs), 1e6))
    return max(0.0, raw - _parsimony_penalty(tree, vocab, raw))


def cubic_fitness(tree: ExprNode, vocab: VocabularyLayer,
                  ctx: EvalContext = None) -> float:
    """Target: f(x) = x^3 - x  over [-3, 3]."""
    if ctx is None:
        ctx = EvalContext(self_fingerprint=tree.fingerprint(), env_tag="cubic")
    xs = np.linspace(-3, 3, 30)
    error = 0.0
    for x in xs:
        try:
            error += abs(_eval_tree(tree, vocab, x, ctx) - (x**3 - x))
        except Exception:
            error += 1e6
    raw = 1.0 / (1.0 + min(error / len(xs), 1e6))
    return max(0.0, raw - _parsimony_penalty(tree, vocab, raw))


FITNESS_REGISTRY: Dict[str, Callable] = {
    "symbolic_regression": symbolic_regression_fitness,
    "sine_approximation": sine_approximation_fitness,
    "absolute_value": absolute_value_fitness,
    "cubic": cubic_fitness,
}


# Lazy getter for VM fitness registry (if omega_backend is available)
def _get_vm_fitness_registry() -> Dict[str, Callable]:
    """Lazily import and return VM fitness registry."""
    try:
        from omega_backend import VM_FITNESS_REGISTRY
        return VM_FITNESS_REGISTRY
    except ImportError:
        return {}


# ---------------------------------------------------------------------------
# 8. FACTORY
# ---------------------------------------------------------------------------

def build_rsi_system(
    fitness_fn: Callable = None,
    fitness_name: str = "symbolic_regression",
    max_depth: int = 5,
    archive_dims: List[int] = None,
    budget_ops: int = 100_000,
    budget_seconds: float = 60.0,
    expansion_interval: int = 10,
    use_enhanced_archive: bool = False,
    use_library_learning: bool = False,
    library_min_depth: int = 2,
    library_min_freq: int = 2,
    library_max_additions: int = 3,
    similarity_threshold: float = 0.85,
    use_vm_backend: bool = False,
    vm_fitness_name: str = None,
    pruning_window: int = 20,
    pruning_threshold: float = 0.05,
) -> SelfImprovementEngine:
    """
    Factory function to construct a complete RSI system.

    Args:
        fitness_fn: evaluation function (tree, vocab) -> float in [0, 1]
        fitness_name: name of fitness function from FITNESS_REGISTRY
        max_depth: maximum expression tree depth
        archive_dims: MAP-Elites grid dimensions [depth_bins, size_bins]
        budget_ops: max compute operations per evaluation
        budget_seconds: max wall-clock seconds per evaluation
        expansion_interval: generations between meta-grammar expansions
        use_enhanced_archive: if True, use EnhancedMAPElitesArchive with
                              novelty injection to mitigate coverage ceiling
        use_library_learning: if True, enable DreamCoder-inspired library
                              learning that extracts recurring subtrees from
                              elites and promotes them to new primitives
        library_min_depth: minimum subtree depth for library extraction
        library_min_freq: minimum frequency for a subtree to be extracted
        library_max_additions: maximum new primitives per library learning step
        similarity_threshold: Jaccard similarity threshold for novelty rejection
                              sampling (from b-albar/evolve-anything). Candidates
                              above this threshold are rejected to force exploration.
        use_vm_backend: if True, use Omega VM backend fitness functions
        vm_fitness_name: name of VM fitness function (if use_vm_backend=True)
        pruning_window: generations between operator pruning checks
        pruning_threshold: fraction of elites that must use an op to keep it
    """
    if fitness_fn is None:
        if use_vm_backend:
            vm_registry = _get_vm_fitness_registry()
            vm_fitness_name = vm_fitness_name or "vm_symbolic_regression"
            fitness_fn = vm_registry.get(vm_fitness_name)
            if fitness_fn is None:
                logger.warning(f"VM fitness '{vm_fitness_name}' not found, falling back to default")
                fitness_fn = FITNESS_REGISTRY.get(fitness_name, symbolic_regression_fitness)
        else:
            fitness_fn = FITNESS_REGISTRY.get(fitness_name, symbolic_regression_fitness)

    # Determine target function and sample points for error-guided meta-mutation
    target_fn = None
    target_xs = None
    resolved_name = fitness_name
    if fitness_fn == sine_approximation_fitness or fitness_name == "sine_approximation":
        target_fn = math.sin
        target_xs = np.linspace(-math.pi, math.pi, 30)
    elif fitness_fn == symbolic_regression_fitness or fitness_name == "symbolic_regression":
        target_fn = lambda x: x**2 + 2*x + 1
        target_xs = np.linspace(-5, 5, 20)
    elif fitness_fn == absolute_value_fitness or fitness_name == "absolute_value":
        target_fn = abs
        target_xs = np.linspace(-5, 5, 30)
    elif fitness_fn == cubic_fitness or fitness_name == "cubic":
        target_fn = lambda x: x**3 - x
        target_xs = np.linspace(-3, 3, 30)

    vocab = VocabularyLayer()
    grammar = GrammarLayer(vocab, max_depth=max_depth)

    lib_learner = None
    if use_library_learning:
        lib_learner = LibraryLearner(
            vocab=vocab,
            min_subtree_depth=library_min_depth,
            min_frequency=library_min_freq,
            max_library_additions=library_max_additions,
        )

    meta_grammar = MetaGrammarLayer(vocab, grammar, library_learner=lib_learner)
    budget = ResourceBudget(max_compute_ops=budget_ops, max_wall_seconds=budget_seconds)
    cost_loop = CostGroundingLoop(budget)

    if use_enhanced_archive:
        archive = EnhancedMAPElitesArchive(dims=archive_dims or [8, 12], novelty_rate=0.15, similarity_threshold=similarity_threshold)
    else:
        archive = MAPElitesArchive(dims=archive_dims or [6, 10])

    return SelfImprovementEngine(
        vocab=vocab, grammar=grammar, meta_grammar=meta_grammar,
        archive=archive, cost_loop=cost_loop, fitness_fn=fitness_fn,
        expansion_interval=expansion_interval,
        pruning_window=pruning_window,
        pruning_threshold=pruning_threshold,
        target_fn=target_fn,
        target_xs=target_xs,
    )


# ---------------------------------------------------------------------------
# 9. CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    """Run multi-domain RSI experiment across all fitness functions."""
    print("=" * 70)
    print("RSI-Exploration: Recursive Self-Improvement Architecture")
    print("Multi-domain experiment with EnhancedMAPElitesArchive + Library Learning")
    print("=" * 70)

    results = {}
    for domain, fn in FITNESS_REGISTRY.items():
        print(f"\n--- Domain: {domain} ---")
        engine = build_rsi_system(
            fitness_fn=fn,
            max_depth=5,
            budget_ops=100_000,
            budget_seconds=60.0,
            expansion_interval=10,
            use_enhanced_archive=True,
            use_library_learning=True,
        )
        engine.run(generations=50, population_size=20)
        s = engine.archive.summary()
        results[domain] = s
        print(f"  coverage={s['coverage']:.4f} | best={s['best_fitness']:.4f} | "
              f"novelty_inserts={s.get('novelty_inserts', 0)} | vocab={engine.vocab.size}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Domain':<25} {'Coverage':>10} {'Best Fitness':>14} {'Novelty':>10}")
    print("-" * 65)
    for domain, s in results.items():
        print(f"{domain:<25} {s['coverage']:>10.4f} {s['best_fitness']:>14.4f} {s.get('novelty_inserts', 0):>10}")

    return results


if __name__ == "__main__":
    main()
