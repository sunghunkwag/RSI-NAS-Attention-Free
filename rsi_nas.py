"""
RSI-NAS: Recursive Self-Improvement for Attention-Free Neural Architecture Search
==================================================================================

Integrates two independent research tracks into a single system:

Track 1 (RSI-Exploration/main.py):
  Three-layer self-improvement engine — Vocabulary, Grammar, Meta-Grammar —
  with MAP-Elites quality-diversity search and cost grounding.

Track 2 (afn3.py, fractal_gnn.py, nca_lm.py, pfn.py, fen.py):
  Attention-free neural architecture modules — NCA dynamics, GatedShiftMixer,
  FractalGNN, CoarseNCA — zero attention, zero softmax, O(L) complexity.

The Integration:
  Programs = neural architectures (not symbolic ExprNode trees)
  Vocabulary = primitive neural modules with typed I/O signatures
  Grammar = rules for composing modules into full networks
  Meta-Grammar = rules for creating new modules from recurring patterns
  EIE/AFIRSI = rules for evolving the pruning/probing instruments when
    generated-module evidence is being destroyed too early
  Fitness = actual SGD training on character-level language modeling
  Archive = MAP-Elites over (param_count, layer_count) behavior space

Why this might succeed where symbolic RSI gave neutral results:
  - Neural architecture composition has a richer combinatorial space
  - Library learning here means "extract a layer pattern that works well
    and freeze it as a single callable module" — analogous to DreamCoder
    but operating on neural graph topology rather than program syntax
  - Fitness is a smooth, differentiable signal (training loss) rather
    than discrete symbolic regression error
  - F_eff has more room to grow because the design space is larger

Source files:
  afn3.py         -> GatedShiftMixer, NCAStep, CoarseNCA, SqueezeExcite, GatedFFN
  fractal_gnn.py  -> FractalGNNBlock, MultiScaleFractalLayer, SimpleGraphConv
  nca_lm.py       -> PerceptionFilter, ReactionGate, DiffusionOperator
  main.py         -> RSI framework pattern (3-layer, MAP-Elites, cost grounding)
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from afirsi_core import (
    Evaluator,
    FailureResidue,
    FailureResidueLedger,
    InstrumentMutationContract,
    ObservationChannel,
    OperatorGenerator,
    ProblemSpaceVersionGraph,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. ATTENTION-FREE MODULE PRIMITIVES
#    Simplified, self-contained versions from the architecture track.
#    Each module has a uniform interface: (B, L, D) -> (B, L, D)
# ===========================================================================

class PerceptionFilter(nn.Module):
    """Local neighbourhood sensing via depthwise separable convolution.
    Source: nca_lm.py / afn3.py"""
    def __init__(self, d: int, k: int = 7, nf: int = 3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d, d, k, padding=k // 2, groups=d, bias=False)
            for _ in range(nf)
        ])
        self.pw = nn.Conv1d(d * nf, d, 1, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        sensed = []
        for conv in self.convs:
            y = conv(h)
            if y.size(-1) < h.size(-1):
                y = F.pad(y, (0, h.size(-1) - y.size(-1)))
            sensed.append(y[..., :h.size(-1)])
        return self.norm(
            self.pw(torch.cat(sensed, 1)).transpose(1, 2)
        )


class ReactionGate(nn.Module):
    """Gated nonlinear state update.
    Source: nca_lm.py / afn3.py"""
    def __init__(self, d: int, exp: int = 2, drop: float = 0.0):
        super().__init__()
        di = d * exp
        self.wu = nn.Linear(2 * d, di, bias=False)
        self.wg = nn.Linear(2 * d, di, bias=False)
        self.wd = nn.Linear(di, d, bias=False)
        self.drop = nn.Dropout(drop)
        self.alpha = nn.Parameter(torch.full((d,), 0.1))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([x, p], -1)
        return x + self.alpha * self.wd(
            self.drop(torch.sigmoid(self.wg(cat)) * F.silu(self.wu(cat)))
        )


class MultiRateDiffusion(nn.Module):
    """Multi-rate dilated diffusion for long-range transport.
    Source: nca_lm.py / afn3.py"""
    def __init__(self, d: int, dilations: Tuple[int, ...] = (1, 4, 16), ks: int = 3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d, d, ks, padding=dil * (ks // 2), dilation=dil,
                      groups=d, bias=False)
            for dil in dilations
        ])
        self.sel = nn.Linear(d, len(dilations), bias=False)
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = x.transpose(1, 2)
        diffs = torch.stack([c(h) for c in self.convs], dim=-1)
        w = torch.softmax(self.sel(x), dim=-1).unsqueeze(1)
        combined = (diffs * w).sum(-1).transpose(1, 2)
        return x + self.strength * (combined - x)


class NCAStep(nn.Module):
    """One cellular automaton timestep: perceive -> react -> diffuse.
    Source: nca_lm.py / afn3.py"""
    def __init__(self, d: int, k: int = 7, dilations: Tuple[int, ...] = (1, 4, 16),
                 exp: int = 2, drop: float = 0.0):
        super().__init__()
        self.perceive = PerceptionFilter(d, k)
        self.react = ReactionGate(d, exp, drop)
        self.diffuse = MultiRateDiffusion(d, dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.diffuse(self.react(x, self.perceive(x)))


class GatedShiftMixer(nn.Module):
    """Content-preserving long-range transfer via gated shift register.
    No attention, no softmax, O(L * n_shifts * D).
    Source: afn3.py"""
    def __init__(self, d_model: int,
                 shifts: Sequence[int] = (-32, -16, -4, -1, 1, 4, 16, 32)):
        super().__init__()
        self.shifts = list(shifts)
        n = len(shifts) + 1
        self.gate_proj = nn.Linear(d_model, n * d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        shifted = [x]
        for s in self.shifts:
            shifted.append(torch.roll(x, shifts=-s, dims=1))
        stacked = torch.stack(shifted, dim=2)
        stacked = self.value_proj(stacked)
        gates = torch.sigmoid(self.gate_proj(x)).reshape(B, L, self.n, D)
        out = (stacked * gates).sum(dim=2)
        return self.norm(out)


class SqueezeExcite(nn.Module):
    """Global channel recalibration.
    Source: afn3.py"""
    def __init__(self, d: int, r: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, d // r, bias=False)
        self.f2 = nn.Linear(d // r, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        g = h.mean(1, keepdim=True)
        return x * torch.sigmoid(self.f2(F.silu(self.f1(g))))


class CoarseNCA(nn.Module):
    """Multi-scale: downsample -> NCA at coarse resolution -> upsample.
    Source: afn3.py"""
    def __init__(self, d: int, stride: int = 4, n_steps: int = 2,
                 k: int = 5, dilations: Tuple[int, ...] = (1, 4),
                 drop: float = 0.0):
        super().__init__()
        self.stride = stride
        self.down = nn.Conv1d(d, d, stride * 2 - 1, stride=stride,
                              padding=stride - 1, groups=d, bias=False)
        self.dp = nn.Conv1d(d, d, 1, bias=False)
        self.dn = nn.LayerNorm(d)
        self.step = NCAStep(d, k, dilations, 2, drop)
        self.n_steps = n_steps
        self.up = nn.ConvTranspose1d(d, d, stride * 2, stride=stride,
                                     padding=stride // 2, groups=d, bias=False)
        self.up_proj = nn.Conv1d(d, d, 1, bias=False)
        self.gate = nn.Linear(2 * d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.dn(self.dp(self.down(x.transpose(1, 2))).transpose(1, 2))
        for _ in range(self.n_steps):
            h = self.step(h)
        hu = self.up_proj(self.up(h.transpose(1, 2))).transpose(1, 2)
        if hu.size(1) < L:
            hu = F.pad(hu, (0, 0, 0, L - hu.size(1)))
        hu = hu[:, :L]
        return torch.sigmoid(self.gate(torch.cat([x, hu], -1))) * hu


class GatedFFN(nn.Module):
    """SwiGLU-style feed-forward.
    Source: afn3.py"""
    def __init__(self, d: int, exp: int = 2, drop: float = 0.0):
        super().__init__()
        di = d * exp
        self.norm = nn.LayerNorm(d)
        self.wg = nn.Linear(d, di, bias=False)
        self.wu = nn.Linear(d, di, bias=False)
        self.wd = nn.Linear(di, d, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm(x)
        return r + self.drop(self.wd(F.silu(self.wg(x)) * self.wu(x)))


class SimpleGraphConv(nn.Module):
    """One-hop message passing on a linear chain.
    Source: fractal_gnn.py"""
    def __init__(self, d_model: int):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_left = F.pad(h[:, :-1], (0, 0, 1, 0))
        h_right = F.pad(h[:, 1:], (0, 0, 0, 1))
        h_agg = (h + h_left + h_right) / 3.0
        return F.relu(self.lin(h_agg))


class FractalGNNBlock(nn.Module):
    """Chunk-pool -> GNN -> gated broadcast. Simplified version.
    Source: fractal_gnn.py"""
    def __init__(self, d_model: int, chunk_size: int = 16, gnn_depth: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.up_proj = nn.Linear(d_model, d_model)
        self.gnns = nn.ModuleList(
            [SimpleGraphConv(d_model) for _ in range(gnn_depth)]
        )
        self.down_mlp = nn.Sequential(
            nn.Linear(2 * d_model, 4 * d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
        self.gate = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        c = self.chunk_size
        N = (L + c - 1) // c
        pad_len = N * c - L

        if pad_len > 0:
            x_pad = torch.cat([x, x.new_zeros(B, pad_len, D)], dim=1)
        else:
            x_pad = x

        x_chunks = x_pad.reshape(B, N, c, D)
        h_chunk = x_chunks.mean(dim=2)
        h_chunk = self.up_proj(h_chunk)

        for gnn in self.gnns:
            h_chunk = h_chunk + gnn(h_chunk)

        h_broadcast = (
            h_chunk.unsqueeze(2).expand(B, N, c, D)
            .reshape(B, N * c, D)[:, :L]
        )

        concat = torch.cat([x, h_broadcast], dim=-1)
        gate = torch.sigmoid(self.gate(concat))
        delta = self.down_mlp(concat)
        return x + self.drop(gate * delta)


# ===========================================================================
# 2. MODULE REGISTRY (RSI Layer 1 — Vocabulary)
#    Each entry describes a constructable neural module with typed metadata.
# ===========================================================================

@dataclass
class ModuleSpec:
    """Specification for a primitive neural module.

    Analogous to PrimitiveOp in main.py, but for neural components.
    """
    name: str
    builder: Callable  # (d_model, **kwargs) -> nn.Module
    default_kwargs: Dict  # default hyperparameters
    param_cost: float  # approximate params per d_model^2 unit
    description: str = ""
    is_generated: bool = False  # True if created by meta-grammar

    def build(self, d_model: int, **override) -> nn.Module:
        kwargs = {**self.default_kwargs, **override}
        return self.builder(d_model, **kwargs)

    def fingerprint(self) -> str:
        data = json.dumps({"name": self.name, "kwargs": self.default_kwargs},
                          sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()[:10]


@dataclass
class GeneratedModuleRecord:
    """Lifecycle evidence for a module created by the meta layer."""
    name: str
    birth_generation: int
    source_action: str
    parent_modules: Tuple[str, ...] = field(default_factory=tuple)
    evaluations: int = 0
    archive_insertions: int = 0
    elite_uses: int = 0
    best_fitness: float = 0.0
    last_evaluated_generation: int = -1
    prune_attempts: int = 0
    protected_from_prune: int = 0

    def source_kind(self) -> str:
        return self.source_action.split(":", 1)[0]


class ModuleRegistry:
    """Layer 1: Manages available neural module primitives.

    Analogous to VocabularyLayer in main.py.
    """
    def __init__(self):
        self._modules: Dict[str, ModuleSpec] = {}
        self._default_names: set = set()
        self._generated_records: Dict[str, GeneratedModuleRecord] = {}
        self._retired_generated_records: List[GeneratedModuleRecord] = []
        self._register_defaults()

    def _register_defaults(self):
        defaults = [
            ModuleSpec(
                name="nca_step",
                builder=lambda d, **kw: NCAStep(d, **kw),
                default_kwargs={"k": 5, "dilations": (1, 4, 16), "exp": 2, "drop": 0.0},
                param_cost=6.0,
                description="NCA perceive-react-diffuse cycle",
            ),
            ModuleSpec(
                name="gated_shift_mixer",
                builder=lambda d, **kw: GatedShiftMixer(d, **kw),
                default_kwargs={"shifts": (-16, -4, -1, 1, 4, 16)},
                param_cost=8.0,
                description="Content-preserving long-range via fixed-offset gating",
            ),
            ModuleSpec(
                name="fractal_gnn",
                builder=lambda d, **kw: FractalGNNBlock(d, **kw),
                default_kwargs={"chunk_size": 16, "gnn_depth": 1, "dropout": 0.0},
                param_cost=10.0,
                description="Chunk-pool -> graph conv -> gated broadcast",
            ),
            ModuleSpec(
                name="coarse_nca",
                builder=lambda d, **kw: CoarseNCA(d, **kw),
                default_kwargs={"stride": 4, "n_steps": 2, "k": 5,
                                "dilations": (1, 4), "drop": 0.0},
                param_cost=5.0,
                description="Multi-scale NCA: downsample-process-upsample",
            ),
            ModuleSpec(
                name="squeeze_excite",
                builder=lambda d, **kw: SqueezeExcite(d, **kw),
                default_kwargs={"r": 4},
                param_cost=0.5,
                description="Global channel recalibration",
            ),
            ModuleSpec(
                name="gated_ffn",
                builder=lambda d, **kw: GatedFFN(d, **kw),
                default_kwargs={"exp": 2, "drop": 0.0},
                param_cost=4.0,
                description="SwiGLU feed-forward block",
            ),
        ]
        for spec in defaults:
            self._modules[spec.name] = spec
            self._default_names.add(spec.name)

    def register(
        self,
        spec: ModuleSpec,
        birth_generation: int = 0,
        source_action: str = "external",
        parent_modules: Sequence[str] = (),
    ):
        self._modules[spec.name] = spec
        if spec.is_generated and spec.name not in self._generated_records:
            self._generated_records[spec.name] = GeneratedModuleRecord(
                name=spec.name,
                birth_generation=birth_generation,
                source_action=source_action,
                parent_modules=tuple(parent_modules),
            )
        logger.info(f"ModuleRegistry: +{spec.name} (cost={spec.param_cost:.1f})")

    def get(self, name: str) -> Optional[ModuleSpec]:
        return self._modules.get(name)

    def unregister(self, name: str) -> bool:
        if name in self._default_names:
            return False
        if name not in self._modules:
            return False
        del self._modules[name]
        record = self._generated_records.pop(name, None)
        if record is not None:
            self._retired_generated_records.append(copy.deepcopy(record))
        logger.info(f"ModuleRegistry: -{name}")
        return True

    def all_specs(self) -> List[ModuleSpec]:
        return list(self._modules.values())

    def random_spec(self) -> ModuleSpec:
        return random.choice(self.all_specs())

    def generated_names(self) -> List[str]:
        return [
            n for n, spec in self._modules.items()
            if n not in self._default_names and spec.is_generated
        ]

    def generated_record(self, name: str) -> Optional[GeneratedModuleRecord]:
        return self._generated_records.get(name)

    def generated_records(self) -> List[GeneratedModuleRecord]:
        return list(self._generated_records.values())

    def all_generated_records(self) -> List[GeneratedModuleRecord]:
        return self._retired_generated_records + self.generated_records()

    def record_genome_evaluation(
        self,
        genome: "ArchitectureGenome",
        generation: int,
        fitness: float,
        inserted: bool,
    ) -> None:
        """Attach real evaluation evidence to generated modules used by a genome."""
        seen = set()
        for layer in genome.layers:
            name = layer.module_name
            if name in seen:
                continue
            seen.add(name)
            record = self._generated_records.get(name)
            if record is None:
                continue
            record.evaluations += 1
            record.best_fitness = max(record.best_fitness, fitness)
            record.last_evaluated_generation = generation
            if inserted:
                record.archive_insertions += 1

    def record_elite_usage(
        self,
        elite_genomes: List["ArchitectureGenome"],
        generation: int,
    ) -> None:
        """Refresh archive evidence for generated modules."""
        counts = Counter()
        for genome in elite_genomes:
            for layer in genome.layers:
                counts[layer.module_name] += 1
        for name, record in self._generated_records.items():
            record.elite_uses = counts.get(name, 0)

    @property
    def size(self) -> int:
        return len(self._modules)


# ===========================================================================
# 3. ARCHITECTURE GENOME
#    A genome is a list of LayerGene, each specifying a module + hyperparams.
# ===========================================================================

@dataclass
class LayerGene:
    """One layer in an architecture genome."""
    module_name: str
    kwargs_override: Dict = field(default_factory=dict)
    repeat: int = 1  # how many times to apply this module

    def to_dict(self) -> dict:
        return {"module": self.module_name, "kwargs": self.kwargs_override,
                "repeat": self.repeat}


@dataclass
class ArchitectureGenome:
    """Complete architecture specification.

    Analogous to ExprNode in main.py, but for neural networks.
    """
    layers: List[LayerGene] = field(default_factory=list)
    d_model: int = 64
    vocab_size: int = 256  # byte-level
    max_len: int = 512

    def depth(self) -> int:
        return sum(g.repeat for g in self.layers)

    def size(self) -> int:
        return len(self.layers)

    def to_dict(self) -> dict:
        return {
            "d_model": self.d_model, "layers": [g.to_dict() for g in self.layers],
            "vocab_size": self.vocab_size, "max_len": self.max_len,
        }

    def fingerprint(self) -> str:
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def clone(self) -> "ArchitectureGenome":
        return copy.deepcopy(self)

    def estimated_params(self, registry: ModuleRegistry) -> int:
        """Rough parameter estimate without building the network."""
        d2 = self.d_model ** 2
        total = self.vocab_size * self.d_model + self.max_len * self.d_model  # embeddings
        for gene in self.layers:
            spec = registry.get(gene.module_name)
            if spec:
                total += int(spec.param_cost * d2) * gene.repeat
        total += self.vocab_size * self.d_model  # output head (tied)
        return total


# ===========================================================================
# 4. NETWORK BUILDER
#    Constructs a runnable nn.Module from an ArchitectureGenome.
# ===========================================================================

class BuiltNetwork(nn.Module):
    """A complete language model built from an ArchitectureGenome.

    Architecture: token_embed + pos_embed -> [layers] -> head
    Head weights are tied to token embeddings.
    """
    def __init__(self, genome: ArchitectureGenome, registry: ModuleRegistry):
        super().__init__()
        d = genome.d_model
        self.tok_emb = nn.Embedding(genome.vocab_size, d)
        self.pos_emb = nn.Embedding(genome.max_len, d)
        self.emb_norm = nn.LayerNorm(d)
        self.emb_drop = nn.Dropout(0.1)

        # Build layer stack from genome
        layers = []
        for gene in genome.layers:
            spec = registry.get(gene.module_name)
            if spec is None:
                logger.warning(f"Module '{gene.module_name}' not found, skipping")
                continue
            for _ in range(gene.repeat):
                try:
                    module = spec.build(d, **gene.kwargs_override)
                    layers.append(module)
                except Exception as e:
                    logger.warning(f"Failed to build {gene.module_name}: {e}")

        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, genome.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.emb_drop(self.emb_norm(self.tok_emb(x) + self.pos_emb(pos)))
        for layer in self.layers:
            h = layer(h)
        return self.head(self.final_norm(h))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_network(genome: ArchitectureGenome,
                  registry: ModuleRegistry) -> Optional[BuiltNetwork]:
    """Safe network construction with error handling."""
    try:
        net = BuiltNetwork(genome, registry)
        # Smoke test
        dummy = torch.zeros(1, 16, dtype=torch.long)
        with torch.no_grad():
            out = net(dummy)
        assert out.shape == (1, 16, genome.vocab_size)
        return net
    except Exception as e:
        logger.warning(f"Network build failed: {e}")
        return None


# ===========================================================================
# 5. FITNESS EVALUATION
#    Train a network for N steps and measure bits-per-character.
# ===========================================================================

def generate_training_data(text: str, seq_len: int, batch_size: int,
                           device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of (input, target) pairs from raw text."""
    data = torch.tensor([b for b in text.encode("utf-8", errors="replace")],
                        dtype=torch.long, device=device)
    if len(data) < seq_len + 1:
        data = data.repeat((seq_len + 1) // len(data) + 2)

    inputs, targets = [], []
    for _ in range(batch_size):
        start = random.randint(0, len(data) - seq_len - 1)
        inputs.append(data[start:start + seq_len])
        targets.append(data[start + 1:start + seq_len + 1])
    return torch.stack(inputs), torch.stack(targets)


# Default training corpus: a small fragment for fast evaluation
_DEFAULT_CORPUS = (
    "To be, or not to be, that is the question: "
    "Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, "
    "Or to take arms against a sea of troubles, "
    "And by opposing end them. To die, to sleep; "
    "No more; and by a sleep to say we end "
    "The heart-ache and the thousand natural shocks "
    "That flesh is heir to: 'tis a consummation "
    "Devoutly to be wish'd. To die, to sleep; "
    "To sleep, perchance to dream—ay, there's the rub: "
    "For in that sleep of death what dreams may come, "
    "When we have shuffled off this mortal coil, "
    "Must give us pause. The quick brown fox jumps over the lazy dog. "
    "All that glitters is not gold. Actions speak louder than words. "
    "Knowledge is power. Time heals all wounds. "
) * 20  # repeat for sufficient training data


@dataclass
class FitnessResult:
    """Result of evaluating an architecture."""
    bpc: float          # bits per character (lower = better)
    fitness: float      # 1 / (1 + bpc) — higher = better, in [0, 1]
    param_count: int
    train_loss: float
    wall_seconds: float
    converged: bool     # did training loss decrease?


def evaluate_architecture(
    genome: ArchitectureGenome,
    registry: ModuleRegistry,
    corpus: str = None,
    train_steps: int = 200,
    seq_len: int = 128,
    batch_size: int = 8,
    lr: float = 3e-3,
    device: torch.device = None,
    max_params: int = 2_000_000,
) -> FitnessResult:
    """Build, train, and evaluate an architecture genome.

    This is the core fitness function — analogous to symbolic_regression_fitness
    in main.py, but operating on neural networks.

    Returns FitnessResult with fitness in [0, 1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if corpus is None:
        corpus = _DEFAULT_CORPUS

    t0 = time.time()

    # Build network
    net = build_network(genome, registry)
    if net is None:
        return FitnessResult(bpc=99.0, fitness=0.0, param_count=0,
                             train_loss=99.0, wall_seconds=0.0, converged=False)

    param_count = net.count_parameters()
    if param_count > max_params:
        return FitnessResult(bpc=99.0, fitness=0.0, param_count=param_count,
                             train_loss=99.0, wall_seconds=0.0, converged=False)
    if param_count == 0:
        return FitnessResult(bpc=99.0, fitness=0.0, param_count=0,
                             train_loss=99.0, wall_seconds=0.0, converged=False)

    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_steps)

    # Training loop
    net.train()
    losses = []
    for step in range(train_steps):
        inputs, targets = generate_training_data(corpus, seq_len, batch_size, device)
        logits = net(inputs)
        loss = F.cross_entropy(logits.view(-1, genome.vocab_size), targets.view(-1))

        if not torch.isfinite(loss):
            return FitnessResult(bpc=99.0, fitness=0.0, param_count=param_count,
                                 train_loss=99.0, wall_seconds=time.time() - t0,
                                 converged=False)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    # Evaluation: compute BPC on held-out data
    net.eval()
    eval_losses = []
    with torch.no_grad():
        for _ in range(5):
            inputs, targets = generate_training_data(corpus, seq_len, batch_size, device)
            logits = net(inputs)
            loss = F.cross_entropy(logits.view(-1, genome.vocab_size), targets.view(-1))
            if torch.isfinite(loss):
                eval_losses.append(loss.item())

    if not eval_losses:
        return FitnessResult(bpc=99.0, fitness=0.0, param_count=param_count,
                             train_loss=losses[-1] if losses else 99.0,
                             wall_seconds=time.time() - t0, converged=False)

    avg_loss = np.mean(eval_losses)
    bpc = avg_loss / math.log(2)  # nats -> bits
    fitness = 1.0 / (1.0 + bpc)

    # Parsimony pressure: slight penalty for large networks
    parsimony = 0.01 * (param_count / max_params)
    fitness = max(0.0, fitness - parsimony)

    converged = len(losses) >= 10 and np.mean(losses[-10:]) < np.mean(losses[:10])

    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return FitnessResult(
        bpc=round(bpc, 4), fitness=round(fitness, 6), param_count=param_count,
        train_loss=round(losses[-1], 4), wall_seconds=round(time.time() - t0, 2),
        converged=converged,
    )


# ===========================================================================
# 6. ARCHITECTURE GRAMMAR (RSI Layer 2)
#    Rules for constructing and mutating architecture genomes.
# ===========================================================================

class ArchitectureGrammar:
    """Layer 2: Rules for composing module primitives into architectures.

    Analogous to GrammarLayer in main.py.
    """
    def __init__(self, registry: ModuleRegistry, max_layers: int = 8,
                 max_repeat: int = 4):
        self.registry = registry
        self.max_layers = max_layers
        self.max_repeat = max_repeat
        self._rules: List[Callable] = [
            self._mutate_swap_module,
            self._mutate_add_layer,
            self._mutate_remove_layer,
            self._mutate_adjust_repeat,
            self._mutate_tweak_hyperparams,
            self._crossover,
        ]

    def random_genome(self, min_layers: int = 2, max_layers: int = 5,
                      d_model: int = 64) -> ArchitectureGenome:
        """Generate a random architecture genome."""
        n_layers = random.randint(min_layers, max_layers)
        layers = []
        for _ in range(n_layers):
            spec = self.registry.random_spec()
            repeat = random.randint(1, 2)
            layers.append(LayerGene(module_name=spec.name, repeat=repeat))
        return ArchitectureGenome(layers=layers, d_model=d_model)

    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Apply a random mutation to a genome."""
        g = genome.clone()
        rule = random.choice(self._rules[:-1])  # exclude crossover
        return rule(g)

    def crossover(self, g1: ArchitectureGenome,
                  g2: ArchitectureGenome) -> ArchitectureGenome:
        """Crossover two genomes."""
        return self._crossover(g1, g2)

    def _mutate_swap_module(self, g: ArchitectureGenome) -> ArchitectureGenome:
        """Replace one layer's module with a different one."""
        if not g.layers:
            return g
        idx = random.randrange(len(g.layers))
        spec = self.registry.random_spec()
        g.layers[idx] = LayerGene(module_name=spec.name,
                                   repeat=g.layers[idx].repeat)
        return g

    def _mutate_add_layer(self, g: ArchitectureGenome) -> ArchitectureGenome:
        """Insert a random layer at a random position."""
        if len(g.layers) >= self.max_layers:
            return g
        spec = self.registry.random_spec()
        pos = random.randint(0, len(g.layers))
        g.layers.insert(pos, LayerGene(module_name=spec.name, repeat=1))
        return g

    def _mutate_remove_layer(self, g: ArchitectureGenome) -> ArchitectureGenome:
        """Remove a random layer."""
        if len(g.layers) <= 1:
            return g
        idx = random.randrange(len(g.layers))
        g.layers.pop(idx)
        return g

    def _mutate_adjust_repeat(self, g: ArchitectureGenome) -> ArchitectureGenome:
        """Increase or decrease a layer's repeat count."""
        if not g.layers:
            return g
        idx = random.randrange(len(g.layers))
        delta = random.choice([-1, 1])
        g.layers[idx].repeat = max(1, min(self.max_repeat,
                                           g.layers[idx].repeat + delta))
        return g

    def _mutate_tweak_hyperparams(self, g: ArchitectureGenome) -> ArchitectureGenome:
        """Modify a hyperparameter of a random layer."""
        if not g.layers:
            return g
        idx = random.randrange(len(g.layers))
        gene = g.layers[idx]
        spec = self.registry.get(gene.module_name)
        if spec is None or not spec.default_kwargs:
            return g

        # Pick a random hyperparam to tweak
        key = random.choice(list(spec.default_kwargs.keys()))
        val = gene.kwargs_override.get(key, spec.default_kwargs[key])

        if isinstance(val, int):
            delta = random.choice([-1, 0, 1])
            gene.kwargs_override[key] = max(1, val + delta)
        elif isinstance(val, float):
            factor = random.uniform(0.7, 1.4)
            gene.kwargs_override[key] = round(val * factor, 3)
        elif isinstance(val, (tuple, list)):
            # For tuples like dilations or shifts: randomly extend or shrink
            val = list(val)
            if random.random() < 0.5 and len(val) > 1:
                val.pop(random.randrange(len(val)))
            else:
                val.append(random.choice([1, 2, 4, 8, 16, 32]))
            gene.kwargs_override[key] = tuple(sorted(set(val)))

        return g

    def _crossover(self, g1: ArchitectureGenome,
                   g2: ArchitectureGenome = None) -> ArchitectureGenome:
        """Single-point crossover between two genomes."""
        if g2 is None:
            g2 = self.random_genome(d_model=g1.d_model)
        g1, g2 = g1.clone(), g2.clone()

        if not g1.layers or not g2.layers:
            return g1

        cut1 = random.randrange(len(g1.layers))
        cut2 = random.randrange(len(g2.layers))
        new_layers = g1.layers[:cut1] + g2.layers[cut2:]

        # Enforce max_layers
        new_layers = new_layers[:self.max_layers]
        if not new_layers:
            new_layers = [g1.layers[0]]

        return ArchitectureGenome(layers=new_layers, d_model=g1.d_model,
                                  vocab_size=g1.vocab_size, max_len=g1.max_len)

    def add_rule(self, rule_fn: Callable):
        self._rules.append(rule_fn)
        logger.info(f"Grammar: +rule {getattr(rule_fn, '__name__', str(rule_fn))}")

    @property
    def num_rules(self) -> int:
        return len(self._rules)


# ===========================================================================
# 7. META-GRAMMAR (RSI Layer 3)
#    Creates new modules by composing existing ones.
#    This is where library learning happens.
# ===========================================================================

@dataclass
class PruningPolicy:
    """EIE-controlled pruning instrument for generated modules."""
    generated_grace_generations: int = 0
    generated_min_evaluations: int = 0
    generated_probe_rate: float = 0.0
    mutation_count: int = 0
    last_mutation_generation: int = -1


@dataclass
class EIEConfig:
    """Mutable RSI instrument configuration used by the outer loop."""
    grace_multiplier: float = 3.0
    probe_rate: float = 1.0
    clean_probe_first_eval: bool = True
    meta_eval_gain: float = 0.25
    meta_archive_gain: float = 2.0
    meta_fitness_gain: float = 4.0
    meta_exploration_floor: float = 0.25


@dataclass
class MetaOperatorStats:
    """Evidence used to evolve the meta-layer's own generation operators."""
    attempts: int = 0
    successes: int = 0
    generated_modules: int = 0
    evaluations: int = 0
    archive_insertions: int = 0
    best_fitness: float = 0.0
    weight: float = 1.0


class EpistemicInstrumentEvolver:
    """Mutates the search instruments when their failure mode becomes visible.

    The ChatGPT brainstorming thread framed RSI as Epistemic Instrument
    Evolution: not merely creating candidate modules, but rewriting the
    instruments that observe, judge, and preserve evidence. In this repo the
    concrete failure residue is the pruning-propagation race: generated modules
    can be removed before mutation and archive insertion have a realistic
    chance to test them.
    """
    def __init__(
        self,
        expansion_interval: int,
        min_evaluations: int = 2,
        config: Optional[EIEConfig] = None,
        ledger: Optional[FailureResidueLedger] = None,
        observation_channel: Optional[ObservationChannel] = None,
        evaluator: Optional[Evaluator] = None,
        operator_generator: Optional[OperatorGenerator] = None,
        mutation_contract: Optional[InstrumentMutationContract] = None,
        problem_space_graph: Optional[ProblemSpaceVersionGraph] = None,
    ):
        self.config = config or EIEConfig()
        self.expansion_interval = max(1, expansion_interval)
        self.min_evaluations = max(0, min_evaluations)
        self.ledger = ledger if ledger is not None else FailureResidueLedger()
        self.observation_channel = (
            observation_channel
            if observation_channel is not None
            else ObservationChannel()
        )
        self.evaluator = evaluator if evaluator is not None else Evaluator(
            expansion_interval=self.expansion_interval,
            min_evaluations=self.min_evaluations,
            config=self.config,
        )
        self.operator_generator = (
            operator_generator
            if operator_generator is not None
            else OperatorGenerator()
        )
        self.mutation_contract = (
            mutation_contract
            if mutation_contract is not None
            else InstrumentMutationContract()
        )
        self.problem_space_graph = (
            problem_space_graph
            if problem_space_graph is not None
            else ProblemSpaceVersionGraph()
        )
        self.residues: List[FailureResidue] = []

    def evaluate_pruning_attempt(
        self,
        record: GeneratedModuleRecord,
        generation: int,
        policy: PruningPolicy,
    ) -> Optional[FailureResidue]:
        parent_version_id = self.problem_space_graph.active_version_id
        policy_before = self._policy_snapshot(policy)
        observation = self.observation_channel.observe_generated_module(
            record=record,
            generation=generation,
            triggering_event="prune_attempt",
            policy=policy,
            problem_space_version=parent_version_id,
        )
        residues = self.evaluator.evaluate([observation])
        if not residues:
            return None

        residue = self.ledger.record(residues[0])
        patches = self.operator_generator.generate(
            [residue],
            instrument_policy=policy_before,
            config=self.config,
            parent_version_id=parent_version_id,
        )
        if patches:
            patch = patches[0]
            self.mutation_contract.validate_patch(patch)
            self._apply_patch(policy, patch.target_updates, generation)
            residue.policy_before = policy_before
            residue.policy_after = self._policy_snapshot(policy)
            residue.proposed_mutation_targets = sorted(
                set(residue.proposed_mutation_targets)
                | set(patch.target_updates)
            )
            self.ledger.mark_addressed(residue.residue_id, patch.patch_id)
            self.problem_space_graph.create_child(
                parent_version_id=parent_version_id,
                patch_id=patch.patch_id,
                residue_ids=[residue.residue_id],
                observation_channel_version=self.observation_channel.version,
                evaluator_version=self.evaluator.version,
                instrument_policy=self._instrument_policy_snapshot(policy),
            )
        else:
            residue.policy_before = policy_before
            residue.policy_after = policy_before

        self.residues.append(residue)
        return residue

    def _policy_snapshot(self, policy: PruningPolicy) -> Dict[str, float]:
        return {
            "generated_grace_generations": policy.generated_grace_generations,
            "generated_min_evaluations": policy.generated_min_evaluations,
            "generated_probe_rate": policy.generated_probe_rate,
            "mutation_count": policy.mutation_count,
            "last_mutation_generation": policy.last_mutation_generation,
        }

    def _instrument_policy_snapshot(self, policy: PruningPolicy) -> Dict[str, object]:
        return {
            "pruning_policy": self._policy_snapshot(policy),
            "eie_config": {
                "grace_multiplier": self.config.grace_multiplier,
                "probe_rate": self.config.probe_rate,
                "clean_probe_first_eval": self.config.clean_probe_first_eval,
                "meta_eval_gain": self.config.meta_eval_gain,
                "meta_archive_gain": self.config.meta_archive_gain,
                "meta_fitness_gain": self.config.meta_fitness_gain,
                "meta_exploration_floor": self.config.meta_exploration_floor,
            },
        }

    def _apply_patch(
        self,
        policy: PruningPolicy,
        updates: Dict[str, object],
        generation: int,
    ) -> None:
        changed = False
        if "generated_grace_generations" in updates:
            target = int(updates["generated_grace_generations"])
            if policy.generated_grace_generations < target:
                policy.generated_grace_generations = target
                changed = True
        if "generated_min_evaluations" in updates:
            target = int(updates["generated_min_evaluations"])
            if policy.generated_min_evaluations < target:
                policy.generated_min_evaluations = target
                changed = True
        if "generated_probe_rate" in updates:
            target = float(updates["generated_probe_rate"])
            if policy.generated_probe_rate < target:
                policy.generated_probe_rate = target
                changed = True
        if "probe_rate" in updates:
            target = float(updates["probe_rate"])
            if policy.generated_probe_rate < target:
                policy.generated_probe_rate = target
                changed = True
        if "clean_probe_first_eval" in updates:
            target = bool(updates["clean_probe_first_eval"])
            if self.config.clean_probe_first_eval != target:
                self.config.clean_probe_first_eval = target
                changed = True
        if changed:
            policy.mutation_count += 1
            policy.last_mutation_generation = generation


class ArchitectureMeta:
    """Layer 3: Rules for generating new modules and grammar rules.

    Analogous to MetaGrammarLayer in main.py.

    Key mechanisms:
    1. Sequential composition: combine two modules into one fused module
    2. Library extraction: find recurring layer patterns in elites
       and promote them to single callable modules
    3. Hyperparameter specialization: create variants of existing modules
       with proven-good hyperparameter settings from elites
    """
    def __init__(
        self,
        registry: ModuleRegistry,
        grammar: ArchitectureGrammar,
        enable_eie: bool = False,
        enable_meta_operator_evolution: Optional[bool] = None,
        eie_config: Optional[EIEConfig] = None,
        expansion_interval: int = 5,
        generated_min_evaluations: int = 2,
    ):
        self.registry = registry
        self.grammar = grammar
        self.expansion_count = 0
        self._expansion_history: List[str] = []
        self.current_generation = 0
        self.pruning_policy = PruningPolicy()
        self.eie_config = eie_config or EIEConfig()
        self.observation_channel = ObservationChannel()
        self.failure_residue_ledger = FailureResidueLedger()
        self.afirsi_evaluator = Evaluator(
            expansion_interval=expansion_interval,
            min_evaluations=generated_min_evaluations,
            config=self.eie_config,
        )
        self.operator_generator = OperatorGenerator()
        self.mutation_contract = InstrumentMutationContract()
        self.problem_space_graph = ProblemSpaceVersionGraph(
            root_policy=self._instrument_policy_snapshot(),
            observation_channel_version=self.observation_channel.version,
            evaluator_version=self.afirsi_evaluator.version,
        )
        self.instrument_evolver = (
            EpistemicInstrumentEvolver(
                expansion_interval=expansion_interval,
                min_evaluations=generated_min_evaluations,
                config=self.eie_config,
                ledger=self.failure_residue_ledger,
                observation_channel=self.observation_channel,
                evaluator=self.afirsi_evaluator,
                operator_generator=self.operator_generator,
                mutation_contract=self.mutation_contract,
                problem_space_graph=self.problem_space_graph,
            )
            if enable_eie else None
        )
        self.pruning_residues: List[FailureResidue] = []
        self.protected_pruning_attempts = 0
        self.instrumented_candidates = 0
        self.enable_meta_operator_evolution = (
            enable_eie
            if enable_meta_operator_evolution is None
            else enable_meta_operator_evolution
        )
        self.meta_operator_stats: Dict[str, MetaOperatorStats] = {
            "library": MetaOperatorStats(),
            "compose": MetaOperatorStats(),
            "specialize": MetaOperatorStats(),
        }
        self.meta_operator_policy_updates = 0

    def _instrument_policy_snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            "pruning_policy": {
                "generated_grace_generations": (
                    self.pruning_policy.generated_grace_generations
                ),
                "generated_min_evaluations": (
                    self.pruning_policy.generated_min_evaluations
                ),
                "generated_probe_rate": self.pruning_policy.generated_probe_rate,
                "mutation_count": self.pruning_policy.mutation_count,
                "last_mutation_generation": (
                    self.pruning_policy.last_mutation_generation
                ),
            },
            "eie_config": {
                "grace_multiplier": self.eie_config.grace_multiplier,
                "probe_rate": self.eie_config.probe_rate,
                "clean_probe_first_eval": self.eie_config.clean_probe_first_eval,
                "meta_eval_gain": self.eie_config.meta_eval_gain,
                "meta_archive_gain": self.eie_config.meta_archive_gain,
                "meta_fitness_gain": self.eie_config.meta_fitness_gain,
                "meta_exploration_floor": self.eie_config.meta_exploration_floor,
            },
        }

    def set_generation(self, generation: int) -> None:
        self.current_generation = generation

    def _register_generated_module(
        self,
        spec: ModuleSpec,
        source_action: str,
        parent_modules: Sequence[str],
    ) -> None:
        self.registry.register(
            spec,
            birth_generation=self.current_generation,
            source_action=source_action,
            parent_modules=parent_modules,
        )

    def instrument_candidate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Inject under-tested generated modules after EIE changes the policy."""
        policy = self.pruning_policy
        if policy.generated_probe_rate <= 0.0:
            return genome
        if random.random() >= policy.generated_probe_rate:
            return genome

        candidates = []
        for record in self.registry.generated_records():
            if self.registry.get(record.name) is None:
                continue
            if record.evaluations < policy.generated_min_evaluations:
                candidates.append(record)

        if not candidates:
            return genome

        candidates.sort(key=lambda r: (r.evaluations, r.birth_generation))
        selected = candidates[0]
        probe = LayerGene(module_name=selected.name, repeat=1)
        if selected.evaluations == 0 and self.eie_config.clean_probe_first_eval:
            self.instrumented_candidates += 1
            return ArchitectureGenome(
                layers=[probe],
                d_model=genome.d_model,
                vocab_size=genome.vocab_size,
                max_len=genome.max_len,
            )

        g = genome.clone()
        if len(g.layers) < self.grammar.max_layers:
            g.layers.insert(random.randint(0, len(g.layers)), probe)
        elif g.layers:
            g.layers[random.randrange(len(g.layers))] = probe
        else:
            g.layers.append(probe)
        self.instrumented_candidates += 1
        return g

    def refresh_meta_operator_policy(self) -> Dict[str, float]:
        """Update meta-operator weights from generated-module archive evidence."""
        for stats in self.meta_operator_stats.values():
            stats.generated_modules = 0
            stats.evaluations = 0
            stats.archive_insertions = 0
            stats.best_fitness = 0.0

        for record in self.registry.all_generated_records():
            stats = self.meta_operator_stats.get(record.source_kind())
            if stats is None:
                continue
            stats.generated_modules += 1
            stats.evaluations += record.evaluations
            stats.archive_insertions += record.archive_insertions
            stats.best_fitness = max(stats.best_fitness, record.best_fitness)

        changed = False
        for stats in self.meta_operator_stats.values():
            old_weight = stats.weight
            evidence_score = (
                1.0
                + self.eie_config.meta_eval_gain * stats.evaluations
                + self.eie_config.meta_archive_gain * stats.archive_insertions
                + self.eie_config.meta_fitness_gain * stats.best_fitness
            )
            exploration_floor = self.eie_config.meta_exploration_floor
            stats.weight = max(exploration_floor, round(evidence_score, 4))
            if abs(stats.weight - old_weight) > 1e-9:
                changed = True

        if changed:
            self.meta_operator_policy_updates += 1

        return {
            name: stats.weight
            for name, stats in self.meta_operator_stats.items()
        }

    def _meta_operator_order(self) -> List[str]:
        operators = ["library", "compose", "specialize"]
        if not self.enable_meta_operator_evolution:
            return operators

        self.refresh_meta_operator_policy()
        remaining = operators[:]
        ordered = []
        while remaining:
            weights = [self.meta_operator_stats[name].weight for name in remaining]
            choice = random.choices(remaining, weights=weights, k=1)[0]
            ordered.append(choice)
            remaining.remove(choice)
        return ordered

    def meta_operator_weights(self) -> Dict[str, float]:
        return {
            name: stats.weight
            for name, stats in self.meta_operator_stats.items()
        }

    def expand_design_space(
        self, elite_genomes: List[ArchitectureGenome],
        elite_fitnesses: List[float],
    ) -> Optional[str]:
        """Attempt to expand the design space.

        Tries mechanisms in priority order:
        1. Library extraction from elite layer patterns
        2. Sequential module composition
        3. Hyperparameter specialization from best elites
        """
        self.expansion_count += 1
        action = None
        for operator in self._meta_operator_order():
            self.meta_operator_stats[operator].attempts += 1
            if operator == "library" and len(elite_genomes) >= 3:
                action = self._extract_library(elite_genomes, elite_fitnesses)
            elif operator == "compose" and self.registry.size >= 3:
                action = self._compose_sequential()
            elif operator == "specialize" and elite_genomes:
                action = self._specialize_hyperparams(
                    elite_genomes,
                    elite_fitnesses,
                )
            if action is not None:
                self.meta_operator_stats[action.split(":", 1)[0]].successes += 1
                break

        if action:
            self._expansion_history.append(action)
        else:
            self._expansion_history.append("no-op")

        return action

    def _extract_library(
        self, genomes: List[ArchitectureGenome], fitnesses: List[float],
    ) -> Optional[str]:
        """DreamCoder-style library learning for neural architecture patterns.

        Scans elite architectures for recurring 2-3 layer sequences.
        If a pattern appears in multiple high-fitness elites, extract it
        as a new single module.
        """
        # Collect all 2-gram and 3-gram layer patterns
        pattern_counts: Dict[str, Tuple[int, List[LayerGene], float]] = {}

        for genome, fit in zip(genomes, fitnesses):
            layers = genome.layers
            for width in [2, 3]:
                for i in range(len(layers) - width + 1):
                    segment = layers[i:i + width]
                    # Create pattern key from module names
                    key = "|".join(g.module_name for g in segment)
                    if key in pattern_counts:
                        count, exemplar, total_fit = pattern_counts[key]
                        pattern_counts[key] = (count + 1, exemplar, total_fit + fit)
                    else:
                        pattern_counts[key] = (1, copy.deepcopy(segment),  fit)

        # Filter: must appear in at least 2 elites
        candidates = [
            (key, count, exemplar, total_fit)
            for key, (count, exemplar, total_fit) in pattern_counts.items()
            if count >= 2
        ]

        if not candidates:
            return None

        # Sort by fitness-weighted frequency
        candidates.sort(key=lambda c: c[1] * c[3], reverse=True)
        key, count, exemplar, total_fit = candidates[0]

        # Create a fused module
        fused_name = f"fused_{'_'.join(g.module_name for g in exemplar)}"
        if self.registry.get(fused_name) is not None:
            return None

        # Build the fused module as a Sequential wrapper
        captured_exemplar = copy.deepcopy(exemplar)
        captured_registry = self.registry

        def fused_builder(d_model: int, **kwargs) -> nn.Module:
            layers = []
            for gene in captured_exemplar:
                spec = captured_registry.get(gene.module_name)
                if spec:
                    for _ in range(gene.repeat):
                        layers.append(spec.build(d_model, **gene.kwargs_override))
            if not layers:
                return nn.Identity()
            return nn.Sequential(*layers)

        total_cost = sum(
            (self.registry.get(g.module_name).param_cost
             if self.registry.get(g.module_name) else 1.0) * g.repeat
            for g in exemplar
        )

        spec = ModuleSpec(
            name=fused_name,
            builder=fused_builder,
            default_kwargs={},
            param_cost=total_cost * 0.9,  # slight discount for fusion
            description=f"Library-learned: {key} (freq={count}, fit_sum={total_fit:.3f})",
            is_generated=True,
        )
        action = f"library:{fused_name}"
        self._register_generated_module(
            spec,
            source_action=action,
            parent_modules=[g.module_name for g in exemplar],
        )
        logger.info(f"Meta: Library extraction -> '{fused_name}' "
                    f"(freq={count}, fit_sum={total_fit:.3f})")
        return action

    def _compose_sequential(self) -> Optional[str]:
        """Create a new module by sequentially composing two existing ones.

        Analogous to HyperRule[unary_chain] in main.py.
        """
        specs = self.registry.all_specs()
        if len(specs) < 2:
            return None

        a, b = random.sample(specs, 2)
        new_name = f"seq_{a.name}_{b.name}"
        if self.registry.get(new_name) is not None:
            return None

        # Capture references for the closure
        spec_a, spec_b = a, b

        def seq_builder(d_model: int, **kwargs) -> nn.Module:
            return nn.Sequential(
                spec_a.build(d_model),
                spec_b.build(d_model),
            )

        spec = ModuleSpec(
            name=new_name,
            builder=seq_builder,
            default_kwargs={},
            param_cost=a.param_cost + b.param_cost,
            description=f"Sequential: {a.name} -> {b.name}",
            is_generated=True,
        )
        action = f"compose:{new_name}"
        self._register_generated_module(
            spec,
            source_action=action,
            parent_modules=[a.name, b.name],
        )
        return action

    def _specialize_hyperparams(
        self, genomes: List[ArchitectureGenome], fitnesses: List[float],
    ) -> Optional[str]:
        """Extract hyperparameter settings from the best elite
        and create a specialized module variant."""
        # Find the best elite
        best_idx = int(np.argmax(fitnesses))
        best_genome = genomes[best_idx]

        if not best_genome.layers:
            return None

        # Pick a random layer from the best elite
        gene = random.choice(best_genome.layers)
        if not gene.kwargs_override:
            return None

        base_spec = self.registry.get(gene.module_name)
        if base_spec is None:
            return None

        # Create a specialized variant with the elite's hyperparams baked in
        merged_kwargs = {**base_spec.default_kwargs, **gene.kwargs_override}
        new_name = f"spec_{gene.module_name}_{hashlib.md5(json.dumps(merged_kwargs, sort_keys=True).encode()).hexdigest()[:6]}"

        if self.registry.get(new_name) is not None:
            return None

        captured_spec = base_spec
        captured_kwargs = merged_kwargs

        def specialized_builder(d_model: int, **kwargs) -> nn.Module:
            final_kwargs = {**captured_kwargs, **kwargs}
            return captured_spec.build(d_model, **final_kwargs)

        spec = ModuleSpec(
            name=new_name,
            builder=specialized_builder,
            default_kwargs=merged_kwargs,
            param_cost=base_spec.param_cost,
            description=f"Specialized {gene.module_name}: {merged_kwargs}",
            is_generated=True,
        )
        action = f"specialize:{new_name}"
        self._register_generated_module(
            spec,
            source_action=action,
            parent_modules=[gene.module_name],
        )
        return action

    def prune_unused(
        self,
        elite_genomes: List[ArchitectureGenome],
        min_usage: int = 1,
        generation: Optional[int] = None,
    ) -> List[str]:
        """Remove generated modules, unless EIE finds insufficient evidence."""
        generation = self.current_generation if generation is None else generation
        used_modules = Counter()
        for g in elite_genomes:
            for layer in g.layers:
                used_modules[layer.module_name] += 1

        self.registry.record_elite_usage(elite_genomes, generation)

        pruned = []
        for name in list(self.registry.generated_names()):
            if used_modules.get(name, 0) >= min_usage:
                continue

            record = self.registry.generated_record(name)
            if record is not None:
                record.prune_attempts += 1

            if self.instrument_evolver is not None and record is not None:
                residue = self.instrument_evolver.evaluate_pruning_attempt(
                    record=record,
                    generation=generation,
                    policy=self.pruning_policy,
                )
                if residue is not None:
                    self.pruning_residues.append(residue)
                    record.protected_from_prune += 1
                    self.protected_pruning_attempts += 1
                    logger.info(
                        "EIE: protected %s from premature pruning "
                        "(age=%s, evals=%s, grace=%s, min_evals=%s, probe=%.2f)",
                        name,
                        residue.age,
                        residue.evaluations,
                        self.pruning_policy.generated_grace_generations,
                        self.pruning_policy.generated_min_evaluations,
                        self.pruning_policy.generated_probe_rate,
                    )
                    continue

            if self.registry.unregister(name):
                pruned.append(name)

        return pruned


# ===========================================================================
# 8. MAP-ELITES ARCHIVE
#    Quality-diversity search over architecture space.
# ===========================================================================

@dataclass
class ArchiveEntry:
    """One elite in the archive."""
    genome: ArchitectureGenome
    fitness: float
    bpc: float
    param_count: int
    behavior: Tuple[int, ...]
    generation: int


class ArchitectureArchive:
    """MAP-Elites archive for neural architectures.

    Behavior dimensions:
      dim 0: parameter count bucket (log scale)
      dim 1: effective depth (total layer repeats)

    Analogous to MAPElitesArchive in main.py.
    """
    def __init__(self, param_bins: int = 8, depth_bins: int = 6):
        self.param_bins = param_bins
        self.depth_bins = depth_bins
        self._grid: Dict[Tuple[int, int], ArchiveEntry] = {}
        self._total_tried = 0
        self._total_inserted = 0

    def behavior_descriptor(self, genome: ArchitectureGenome,
                           param_count: int) -> Tuple[int, int]:
        """Map genome to behavior space."""
        # Param bucket: log-scale from 1K to 2M
        if param_count <= 0:
            p_bin = 0
        else:
            log_p = math.log10(max(param_count, 1000))
            p_bin = min(self.param_bins - 1,
                        int((log_p - 3) / (6.3 - 3) * self.param_bins))
            p_bin = max(0, p_bin)

        # Depth bucket
        d = genome.depth()
        d_bin = min(self.depth_bins - 1, d - 1)
        d_bin = max(0, d_bin)

        return (p_bin, d_bin)

    def try_insert(self, entry: ArchiveEntry) -> bool:
        self._total_tried += 1
        cell = entry.behavior
        if cell not in self._grid or entry.fitness > self._grid[cell].fitness:
            self._grid[cell] = entry
            self._total_inserted += 1
            return True
        return False

    def sample_parent(self) -> Optional[ArchiveEntry]:
        if not self._grid:
            return None
        return random.choice(list(self._grid.values()))

    def all_entries(self) -> List[ArchiveEntry]:
        return list(self._grid.values())

    @property
    def coverage(self) -> float:
        return len(self._grid) / (self.param_bins * self.depth_bins)

    @property
    def best_fitness(self) -> float:
        return max((e.fitness for e in self._grid.values()), default=0.0)

    @property
    def best_bpc(self) -> float:
        return min((e.bpc for e in self._grid.values()), default=99.0)

    def summary(self) -> dict:
        entries = list(self._grid.values())
        return {
            "filled_cells": len(entries),
            "total_cells": self.param_bins * self.depth_bins,
            "coverage": round(self.coverage, 4),
            "best_fitness": round(self.best_fitness, 6),
            "best_bpc": round(self.best_bpc, 4),
            "mean_bpc": round(np.mean([e.bpc for e in entries]), 4) if entries else 99.0,
            "total_tried": self._total_tried,
            "total_inserted": self._total_inserted,
        }


# ===========================================================================
# 9. RSI ENGINE
#    The main evolutionary loop with recursive self-improvement.
# ===========================================================================

class RSINASEngine:
    """Recursive Self-Improvement engine for Neural Architecture Search.

    Integrates all three layers:
      Layer 1 (ModuleRegistry) — primitive modules
      Layer 2 (ArchitectureGrammar) — composition rules
      Layer 3 (ArchitectureMeta) — meta-rules for design space expansion

    The loop:
      1. Generate candidate architectures via mutation/crossover
      2. Evaluate by actual training (SGD on character-level LM)
      3. Insert into MAP-Elites archive
      4. Periodically expand design space (library learning, composition)
      5. Periodically prune unused modules

    Analogous to SelfImprovementEngine in main.py, but operating on
    neural architectures instead of symbolic expression trees.
    """
    def __init__(
        self,
        registry: ModuleRegistry,
        grammar: ArchitectureGrammar,
        meta: ArchitectureMeta,
        archive: ArchitectureArchive,
        corpus: str = None,
        d_model: int = 64,
        train_steps: int = 200,
        expansion_interval: int = 5,
        pruning_interval: int = 10,
        device: torch.device = None,
    ):
        self.registry = registry
        self.grammar = grammar
        self.meta = meta
        self.archive = archive
        self.corpus = corpus or _DEFAULT_CORPUS
        self.d_model = d_model
        self.train_steps = train_steps
        self.expansion_interval = expansion_interval
        self.pruning_interval = pruning_interval
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.generation = 0
        self.history: List[dict] = []

    def step(self, population_size: int = 6) -> dict:
        """One generation of the RSI loop."""
        self.generation += 1
        self.meta.set_generation(self.generation)
        inserted = 0
        best_gen_fitness = 0.0
        best_gen_bpc = 99.0
        pruned = []

        for _ in range(population_size):
            # Generate candidate
            parent = self.archive.sample_parent()
            if parent is not None:
                if random.random() < 0.2:
                    # Crossover with another elite
                    other = self.archive.sample_parent()
                    if other is not None:
                        genome = self.grammar.crossover(parent.genome, other.genome)
                    else:
                        genome = self.grammar.mutate(parent.genome)
                else:
                    genome = self.grammar.mutate(parent.genome)
            else:
                genome = self.grammar.random_genome(d_model=self.d_model)

            genome = self.meta.instrument_candidate(genome)

            # Evaluate
            result = evaluate_architecture(
                genome, self.registry,
                corpus=self.corpus,
                train_steps=self.train_steps,
                device=self.device,
            )

            # Archive insertion
            behavior = self.archive.behavior_descriptor(genome, result.param_count)
            entry = ArchiveEntry(
                genome=genome, fitness=result.fitness, bpc=result.bpc,
                param_count=result.param_count, behavior=behavior,
                generation=self.generation,
            )
            was_inserted = self.archive.try_insert(entry)
            self.registry.record_genome_evaluation(
                genome=genome,
                generation=self.generation,
                fitness=result.fitness,
                inserted=was_inserted,
            )
            if was_inserted:
                inserted += 1

            best_gen_fitness = max(best_gen_fitness, result.fitness)
            best_gen_bpc = min(best_gen_bpc, result.bpc)

        # Meta-grammar expansion
        expansion_action = None
        if self.generation % self.expansion_interval == 0:
            entries = self.archive.all_entries()
            if entries:
                elite_genomes = [e.genome for e in entries]
                elite_fitnesses = [e.fitness for e in entries]
                expansion_action = self.meta.expand_design_space(
                    elite_genomes, elite_fitnesses
                )

        # Pruning
        if self.generation % self.pruning_interval == 0:
            entries = self.archive.all_entries()
            if entries:
                pruned = self.meta.prune_unused(
                    [e.genome for e in entries],
                    generation=self.generation,
                )
                if pruned:
                    logger.info(f"Pruned {len(pruned)} unused modules: "
                               f"{', '.join(pruned[:3])}")

        record = {
            "generation": self.generation,
            "inserted": inserted,
            "best_gen_fitness": round(best_gen_fitness, 6),
            "best_gen_bpc": round(best_gen_bpc, 4),
            "archive_coverage": round(self.archive.coverage, 4),
            "archive_best_fitness": round(self.archive.best_fitness, 6),
            "archive_best_bpc": round(self.archive.best_bpc, 4),
            "vocab_size": self.registry.size,
            "generated_modules": len(self.registry.generated_names()),
            "expansion_action": expansion_action,
            "pruned_modules": len(pruned),
            "eie_residues": len(self.meta.pruning_residues),
            "instrument_mutations": self.meta.pruning_policy.mutation_count,
            "protected_pruning_attempts": self.meta.protected_pruning_attempts,
            "instrumented_candidates": self.meta.instrumented_candidates,
            "generated_probe_rate": round(
                self.meta.pruning_policy.generated_probe_rate, 4
            ),
            "problem_space_version": self.meta.problem_space_graph.active_version_id,
            "failure_residue_ledger_entries": len(
                self.meta.failure_residue_ledger
            ),
            "meta_operator_policy_updates": self.meta.meta_operator_policy_updates,
            "meta_operator_weights": self.meta.meta_operator_weights(),
        }
        self.history.append(record)
        return record

    def run(self, generations: int = 20, population_size: int = 6) -> List[dict]:
        """Run the full RSI-NAS loop."""
        logger.info(f"RSI-NAS: {generations} gen x {population_size} pop, "
                    f"d_model={self.d_model}, train_steps={self.train_steps}, "
                    f"device={self.device}")

        for g in range(generations):
            record = self.step(population_size)
            if g % 5 == 0 or g == generations - 1:
                logger.info(
                    f"Gen {record['generation']:3d} | "
                    f"best_bpc={record['archive_best_bpc']:.3f} | "
                    f"coverage={record['archive_coverage']:.3f} | "
                    f"vocab={record['vocab_size']} | "
                    f"gen_modules={record['generated_modules']} | "
                    f"action={record['expansion_action']}"
                )

        return self.history


# ===========================================================================
# 10. FACTORY + ENTRY POINT
# ===========================================================================

def build_rsi_nas(
    d_model: int = 48,
    train_steps: int = 150,
    expansion_interval: int = 5,
    pruning_interval: int = 10,
    enable_eie: bool = True,
    enable_meta_operator_evolution: Optional[bool] = None,
    eie_config: Optional[EIEConfig] = None,
    generated_min_evaluations: int = 2,
    corpus: str = None,
    device: torch.device = None,
) -> RSINASEngine:
    """Factory function to construct a complete RSI-NAS system."""
    registry = ModuleRegistry()
    grammar = ArchitectureGrammar(registry, max_layers=6, max_repeat=3)
    meta = ArchitectureMeta(
        registry,
        grammar,
        enable_eie=enable_eie,
        enable_meta_operator_evolution=enable_meta_operator_evolution,
        eie_config=eie_config,
        expansion_interval=expansion_interval,
        generated_min_evaluations=generated_min_evaluations,
    )
    archive = ArchitectureArchive(param_bins=6, depth_bins=5)

    return RSINASEngine(
        registry=registry, grammar=grammar, meta=meta, archive=archive,
        corpus=corpus, d_model=d_model, train_steps=train_steps,
        expansion_interval=expansion_interval,
        pruning_interval=pruning_interval,
        device=device,
    )


def run_ablation(
    seeds: List[int] = None,
    generations: int = 20,
    population_size: int = 4,
    d_model: int = 48,
    train_steps: int = 100,
):
    """Controlled ablation: FROZEN vs baseline RSI vs EIE-enabled RSI.

    FROZEN: expansion_interval=999999 (no RSI)
    SELF-MODIFY: expansion_interval=5 (library learning active)
    AFIRSI-EIE: expansion_interval=5 plus instrument evolution for pruning races
    """
    if seeds is None:
        seeds = [42, 123, 456]

    print("=" * 72)
    print("  RSI-NAS ABLATION STUDY")
    print(f"  {len(seeds)} seeds x {generations} gen x {population_size} pop")
    print(f"  d_model={d_model}, train_steps={train_steps}")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    conditions = [
        ("FROZEN", 999999, False),
        ("SELF-MODIFY", 5, False),
        ("AFIRSI-EIE", 5, True),
    ]

    all_results = {}
    for label, interval, enable_eie in conditions:
        print(f"  Condition: {label} "
              f"(expansion_interval={interval}, eie={enable_eie})")
        results = []
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            engine = build_rsi_nas(
                d_model=d_model,
                train_steps=train_steps,
                expansion_interval=interval,
                enable_eie=enable_eie,
                device=device,
            )
            history = engine.run(generations=generations,
                                population_size=population_size)

            final = history[-1]
            results.append(final)
            print(f"    Seed {seed:4d}: bpc={final['archive_best_bpc']:.3f} "
                  f"fitness={final['archive_best_fitness']:.4f} "
                  f"coverage={final['archive_coverage']:.3f} "
                  f"vocab={final['vocab_size']} "
                  f"gen_modules={final['generated_modules']} "
                  f"residues={final['eie_residues']} "
                  f"instrument_mutations={final['instrument_mutations']}")

        all_results[label] = results

    # Analysis
    print(f"\n{'=' * 72}")
    print("  ANALYSIS")
    print(f"{'=' * 72}")

    for label, results in all_results.items():
        bpcs = [r["archive_best_bpc"] for r in results]
        fits = [r["archive_best_fitness"] for r in results]
        print(f"\n  {label}:")
        print(f"    BPC:     {np.mean(bpcs):.4f} +/- {np.std(bpcs):.4f}  {bpcs}")
        print(f"    Fitness: {np.mean(fits):.4f} +/- {np.std(fits):.4f}")

    frozen_bpc = np.mean([r["archive_best_bpc"] for r in all_results["FROZEN"]])
    modify_bpc = np.mean([r["archive_best_bpc"] for r in all_results["SELF-MODIFY"]])
    eie_bpc = np.mean([r["archive_best_bpc"] for r in all_results["AFIRSI-EIE"]])
    delta = frozen_bpc - modify_bpc  # positive = MODIFY is better
    eie_delta = modify_bpc - eie_bpc  # positive = EIE is better than baseline RSI

    print(f"\n  Delta BPC (FROZEN - MODIFY) = {delta:+.4f}")
    print(f"  Delta BPC (MODIFY - AFIRSI-EIE) = {eie_delta:+.4f}")
    if delta > 0.05:
        print("  VERDICT: RSI_IMPROVES_ARCHITECTURE — Self-modification reduces BPC")
    elif delta > 0.01:
        print("  VERDICT: MARGINAL_IMPROVEMENT")
    elif delta > -0.01:
        print("  VERDICT: NEUTRAL — No significant difference")
    else:
        print("  VERDICT: RSI_HURTS — Self-modification increases BPC")

    gen_modules = [r["generated_modules"] for r in all_results["SELF-MODIFY"]]
    print(f"\n  Generated modules (SELF-MODIFY): {gen_modules}")
    eie_modules = [r["generated_modules"] for r in all_results["AFIRSI-EIE"]]
    print(f"  Generated modules (AFIRSI-EIE): {eie_modules}")
    residues = [r["eie_residues"] for r in all_results["AFIRSI-EIE"]]
    mutations = [r["instrument_mutations"] for r in all_results["AFIRSI-EIE"]]
    print(f"  EIE residues detected: {residues}")
    print(f"  Instrument mutations: {mutations}")
    print(f"  F_theo expansion: {sum(gen_modules)} new module types created")
    print(f"  F_eff question: Did they lower BPC?  Delta = {delta:+.4f}")

    return all_results


def main():
    """Quick demo: run a single RSI-NAS experiment."""
    print("=" * 72)
    print("  RSI-NAS: Recursive Self-Improvement for")
    print("  Attention-Free Neural Architecture Search")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    engine = build_rsi_nas(d_model=48, train_steps=100, device=device)
    history = engine.run(generations=15, population_size=4)

    print(f"\n{'=' * 72}")
    print("  FINAL RESULTS")
    print(f"{'=' * 72}")
    summary = engine.archive.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Show best architecture
    entries = engine.archive.all_entries()
    if entries:
        best = max(entries, key=lambda e: e.fitness)
        print(f"\n  Best architecture (BPC={best.bpc:.3f}):")
        for i, gene in enumerate(best.genome.layers):
            print(f"    Layer {i}: {gene.module_name} x{gene.repeat} "
                  f"{gene.kwargs_override if gene.kwargs_override else ''}")
        print(f"  Parameters: {best.param_count:,}")

    # Show generated modules
    gen_names = engine.registry.generated_names()
    if gen_names:
        print(f"\n  Generated modules ({len(gen_names)}):")
        for name in gen_names:
            spec = engine.registry.get(name)
            if spec:
                print(f"    {name}: {spec.description}")

    return history


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ablation":
        run_ablation()
    else:
        main()
