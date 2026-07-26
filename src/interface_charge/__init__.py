"""Interface-resolved charge analysis for charge-controlled ProteinMPNN designs.

This package answers two questions about an inference-time net-charge
controller applied to protein-protein complexes drawn from the RCSB:

1. When the charge dial turns, where does the charge actually land? Net charge
   is partitioned into interface, non-interface surface and buried core, under
   both of the charge definitions the source paper uses, and each partition's
   response to the charge shift is measured separately. The hypothesis is that
   the interface is buffered relative to the bulk surface.

2. Does the interface survive? Per-chain folding is replaced with
   complex-aware AlphaFold2-Multimer refolding, and interface RMSD, interface
   PAE and interface pTM are reported against the charge shift.

Module map
----------
``config``
    Every tunable parameter, in one place, recorded in every manifest.
``contracts``
    The input data contract and its strict enforcement.
``structures``
    RCSB fetching, parsing, and chain extraction.
``interface``
    Two interface definitions, their agreement, and the three-way partition.
``charge``
    The two charge definitions, kept apart, and charge partitioning.
``complementarity``
    Contact-level electrostatic complementarity across the interface.
``provenance``
    Run manifests tying every output to one reproducible run.
``plotting``
    Figure house style, with the manifest hash embedded in the file metadata.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "charge",
    "complementarity",
    "config",
    "contracts",
    "interface",
    "plotting",
    "provenance",
    "structures",
]
