from pathlib import Path

import torch

from local_agi_architect import (
    LocalInventoryArchitect,
    LocalMotifFusionBlock,
    candidate_source_paths,
    extract_signal,
    primitive_sequence_from_signal,
)
from rsi_nas import build_network


def test_extract_signal_reads_ast_motifs(tmp_path: Path):
    source = tmp_path / "organic_rsi_agent.py"
    source.write_text(
        """
import ast
import torch

class RecursiveSelfImprovementAgent:
    def evolve_population(self):
        assert True
        return ast.parse("x = 1")
""",
        encoding="utf-8",
    )

    signal = extract_signal(source)

    assert signal.syntax_valid
    assert signal.keyword_counts["rsi"] > 0
    assert signal.keyword_counts["neural"] > 0
    assert signal.keyword_counts["evolution"] > 0
    assert "RecursiveSelfImprovementAgent" in signal.class_names


def test_candidate_source_paths_filters_installs():
    paths = [
        Path(r"C:\Python314\Lib\site-packages\torch\nn.py"),
        Path(r"C:\Users\starg\Downloads\True-RSI\unified_rsi.py"),
    ]

    selected = candidate_source_paths(paths, max_files=10)

    assert selected == [Path(r"C:\Users\starg\Downloads\True-RSI\unified_rsi.py")]


def test_motif_fusion_block_preserves_shape():
    block = LocalMotifFusionBlock(
        d_model=8,
        motif_sequence=("gated_shift_mixer", "gated_ffn", "squeeze_excite"),
    )
    x = torch.randn(2, 16, 8)

    y = block(x)

    assert y.shape == x.shape


def test_architect_registers_buildable_candidate(tmp_path: Path):
    source = tmp_path / "cognitive_evolution_rsi.py"
    source.write_text(
        """
import torch

class CognitiveRSIEngine:
    def mutate_and_evaluate(self, tensor):
        fitness = tensor.sum()
        return fitness
""",
        encoding="utf-8",
    )
    architect = LocalInventoryArchitect()
    signals = [extract_signal(source)]
    candidates = architect.synthesize_candidates(signals, count=1, d_model=8)

    assert len(candidates) == 1
    assert candidates[0].buildable
    assert candidates[0].generated_module_name in architect.registry.generated_names()
    assert build_network(candidates[0].genome, architect.registry) is not None


def test_primitive_sequence_is_deterministic(tmp_path: Path):
    source = tmp_path / "symbolic_neural_rsi.py"
    source.write_text(
        """
import ast
import torch

def synthesize_program():
    return ast.parse("x = 1")
""",
        encoding="utf-8",
    )
    signal = extract_signal(source)

    assert primitive_sequence_from_signal(signal) == primitive_sequence_from_signal(signal)
