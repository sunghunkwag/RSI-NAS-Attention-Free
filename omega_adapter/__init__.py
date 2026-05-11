"""AFIRSI-OMEGA integration layer for RSI-NAS."""

from .instrument_generator import OMEGAInstrumentGenerator
from .policy_import import OMEGAPolicyCandidate, apply_instrument_patch
from .residue_export import export_residues
from .schemas import (
    InstrumentMutationContract,
    InstrumentPatch,
    StructuredAFIRSIResidue,
)

__all__ = [
    "InstrumentMutationContract",
    "InstrumentPatch",
    "OMEGAInstrumentGenerator",
    "OMEGAPolicyCandidate",
    "StructuredAFIRSIResidue",
    "apply_instrument_patch",
    "export_residues",
]
