"""Fetching, parsing and chain handling for native RCSB structures.

Everything downstream indexes residues by :class:`ResidueId`, a hashable
``(chain, seqid, icode)`` triple, rather than by position in a list. Author
numbering in a deposited structure has gaps, insertion codes and occasionally
negative numbers, so a positional index computed in one module and used in
another is a silent-error generator. The mapping from :class:`ResidueId` to a
zero-based position in the chain sequence is built once, in
:func:`extract_chain`, and passed around explicitly.
"""

from __future__ import annotations

import copy
import gzip
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

from .config import STANDARD_AA, StructureParams

__all__ = [
    "ChainExtract",
    "ResidueId",
    "StructureError",
    "chains_present",
    "extract_chain",
    "fetch_native",
    "heavy_atoms",
    "load_model",
    "submodel",
]


class StructureError(RuntimeError):
    """Raised on any structural parsing problem that must not be papered over."""


class ResidueId(NamedTuple):
    """Stable identity of a residue: chain, author sequence number, insertion code."""

    chain: str
    seqid: int
    icode: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.chain}{self.seqid}{self.icode.strip()}"


# ---------------------------------------------------------------------------
# Residue name handling
# ---------------------------------------------------------------------------

_THREE_TO_ONE: Final[dict[str, str]] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

#: Modified residues that are mapped onto their parent amino acid. The formal
#: charge of the parent is unchanged by these modifications, so the mapping is
#: safe for a charge calculation. Every substitution actually applied is
#: recorded on :class:`ChainExtract` and written into the run manifest, so the
#: mapping is documented in the output rather than being invisible.
MODIFIED_RESIDUE_MAP: Final[dict[str, str]] = {
    "MSE": "M",  # selenomethionine, ubiquitous in experimentally phased structures
    "FME": "M",  # N-formylmethionine
    "MHO": "M",  # S-oxymethionine
    "HYP": "P",  # 4-hydroxyproline
    "CSO": "C",  # S-hydroxycysteine
    "CME": "C",  # S,S-(2-hydroxyethyl)thiocysteine
    "CSD": "C",  # S-cysteinesulfinic acid
    "OCS": "C",  # cysteinesulfonic acid
    "HIC": "H",  # 4-methylhistidine
    "MLY": "K",  # N-dimethyl-lysine, still cationic
    "M3L": "K",  # N-trimethyl-lysine, permanently cationic
    "ALY": "K",  # N-acetyl-lysine loses the positive charge, see note below
    "KCX": "K",  # lysine carbamylation
    "LLP": "K",  # lysine-pyridoxal phosphate
    "SEP": "S",  # phosphoserine
    "TPO": "T",  # phosphothreonine
    "PTR": "Y",  # phosphotyrosine
    "PCA": "Q",  # pyroglutamate
}

#: Modified residues from the map above whose formal charge at pH 7.4 differs
#: substantially from the parent. Mapping these onto the parent would change a
#: reported net charge, so hitting one raises unless the caller opts in
#: explicitly. The phosphorylated trio each carry close to minus two, and
#: acetyl-lysine and carbamyl-lysine lose the lysine's plus one.
CHARGE_ALTERING_MODIFICATIONS: Final[frozenset[str]] = frozenset(
    {"SEP", "TPO", "PTR", "ALY", "KCX", "PCA"}
)

_STANDARD_SET: Final[frozenset[str]] = frozenset(STANDARD_AA)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def fetch_native(
    pdb_id: str,
    dest_dir: Path,
    params: StructureParams,
    overwrite: bool = False,
) -> Path:
    """Download one native structure from the RCSB into ``dest_dir``.

    Returns the path to the local file. Existing files are reused unless
    ``overwrite`` is set, which is what makes ``scripts/00_fetch_natives.py``
    cheap to re-run.

    Raises :class:`StructureError` with the identifier and the underlying HTTP
    error if the download cannot be completed. It never writes a partial or
    empty file: the download goes to a temporary name and is renamed only on
    success, so an interrupted run cannot leave behind a truncated structure
    that later parses into a subtly wrong answer.
    """
    pdb_id = _normalise_pdb_id(pdb_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{pdb_id}.{params.native_format}"

    if target.is_file() and target.stat().st_size > 0 and not overwrite:
        return target

    url = params.rcsb_url_template.format(pdb_id=pdb_id)
    tmp = target.with_suffix(target.suffix + ".partial")
    last_error: Exception | None = None
    backoff = params.download_backoff_s

    for attempt in range(1, params.download_attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "interface-charge-rcsb/0.1 (research use)"},
            )
            with urllib.request.urlopen(request, timeout=params.download_timeout_s) as response:
                payload = response.read()
            if not payload:
                raise StructureError(f"empty response body for {pdb_id} from {url}")
            tmp.write_bytes(payload)
            tmp.replace(target)
            return target
        except (urllib.error.URLError, TimeoutError, OSError, StructureError) as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < params.download_attempts:
                time.sleep(backoff)
                backoff *= 2.0

    raise StructureError(
        f"failed to download {pdb_id} from {url} after {params.download_attempts} "
        f"attempts. Last error: {last_error!r}. If this is a network policy "
        "denial rather than a transient failure, the host files.rcsb.org has to "
        "be reachable from wherever this is running."
    )


def _normalise_pdb_id(pdb_id: str) -> str:
    """Upper-case and validate a four-character PDB identifier.

    Case normalisation is the only input transformation this project performs
    silently, and it is documented here and in ``data/README.md``. Anything
    else that is not already a valid identifier raises.
    """
    if not isinstance(pdb_id, str):
        raise StructureError(f"pdb_id must be a string, got {type(pdb_id).__name__}")
    cleaned = pdb_id.strip().upper()
    if len(cleaned) != 4 or not cleaned[0].isdigit() or not cleaned.isalnum():
        raise StructureError(
            f"{pdb_id!r} is not a four-character PDB identifier of the form "
            "digit followed by three alphanumerics, for example 1BRS"
        )
    return cleaned


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def load_model(path: Path, params: StructureParams) -> Model:
    """Parse a structure file and return the requested model.

    Both mmCIF and legacy PDB are supported, chosen by file suffix, and both
    may be gzipped. mmCIF is the default for freshly fetched natives because
    the PDB format cannot represent large complexes or multi-character chain
    identifiers.
    """
    if not path.is_file():
        raise StructureError(
            f"structure file not found: {path}. Run scripts/00_fetch_natives.py first."
        )

    suffixes = [s.lower() for s in path.suffixes]
    is_gz = suffixes and suffixes[-1] == ".gz"
    fmt_suffix = (
        suffixes[-2] if is_gz and len(suffixes) >= 2 else (suffixes[-1] if suffixes else "")
    )

    if fmt_suffix in (".cif", ".mmcif"):
        parser: MMCIFParser | PDBParser = MMCIFParser(QUIET=True)
    elif fmt_suffix in (".pdb", ".ent"):
        parser = PDBParser(QUIET=True)
    else:
        raise StructureError(
            f"cannot infer a structure format from {path.name}. "
            "Expected one of .cif, .mmcif, .pdb or .ent, optionally gzipped."
        )

    try:
        if is_gz:
            with gzip.open(path, "rt") as handle:
                structure: Structure = parser.get_structure(path.stem, handle)
        else:
            structure = parser.get_structure(path.stem, path)
    except Exception as exc:
        raise StructureError(f"failed to parse {path}: {exc!r}") from exc

    models = list(structure)
    if not models:
        raise StructureError(f"{path} contains no models")
    if params.model_index >= len(models):
        raise StructureError(
            f"{path} contains {len(models)} model(s) but model_index="
            f"{params.model_index} was requested"
        )
    return models[params.model_index]


def chains_present(model: Model) -> list[str]:
    """Chain identifiers in ``model``, in file order."""
    return [chain.id for chain in model]


# ---------------------------------------------------------------------------
# Chain extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChainExtract:
    """One protein chain, reduced to what the rest of the pipeline needs.

    Attributes
    ----------
    chain_id
        The author chain identifier.
    residues
        Polymer residues in file order, standard and mapped-modified only.
    sequence
        One-letter sequence, same length and order as ``residues``.
    index_of
        Mapping from :class:`ResidueId` to zero-based position in ``sequence``.
    substitutions
        Every modified residue that was mapped onto a parent amino acid, as
        ``(ResidueId, original_three_letter_code, mapped_one_letter)``. Written
        into the run manifest so the mapping is never invisible.
    skipped_heteroatoms
        Count of non-polymer residues (waters, ions, ligands) excluded.
    """

    chain_id: str
    residues: tuple[Residue, ...]
    sequence: str
    index_of: dict[ResidueId, int]
    substitutions: tuple[tuple[ResidueId, str, str], ...]
    skipped_heteroatoms: int

    def residue_ids(self) -> tuple[ResidueId, ...]:
        """Residue identities in sequence order."""
        return tuple(residue_id(r, self.chain_id) for r in self.residues)

    def with_sequence(self, sequence: str) -> ChainExtract:
        """Return a copy carrying a designed sequence on the native backbone.

        This is how a ProteinMPNN design is threaded onto its native structure:
        the coordinates, residue identities and index map are all kept, and only
        the sequence changes. It is valid precisely because ProteinMPNN is
        fixed-backbone, so position *i* of the design is the same structural
        position as residue *i* of the native chain.

        The length check is the whole point of routing this through a method
        rather than assigning the attribute. A design of the wrong length would
        silently shift every residue index by the difference.
        """
        if len(sequence) != len(self.sequence):
            raise StructureError(
                f"cannot thread a {len(sequence)}-residue sequence onto chain "
                f"{self.chain_id}, which has {len(self.sequence)} residues. "
                "ProteinMPNN is fixed-backbone, so a design must match its "
                "native chain length exactly."
            )
        return ChainExtract(
            chain_id=self.chain_id,
            residues=self.residues,
            sequence=sequence,
            index_of=self.index_of,
            substitutions=self.substitutions,
            skipped_heteroatoms=self.skipped_heteroatoms,
        )

    def positions_of(self, ids: Iterable[ResidueId]) -> frozenset[int]:
        """Map an iterable of residue identities to zero-based sequence positions.

        Identities that do not belong to this chain raise, rather than being
        dropped. A dropped interface residue would quietly shrink the interface
        partition and inflate the surface one.
        """
        out: set[int] = set()
        missing: list[ResidueId] = []
        for rid in ids:
            pos = self.index_of.get(rid)
            if pos is None:
                missing.append(rid)
            else:
                out.add(pos)
        if missing:
            raise StructureError(
                f"{len(missing)} residue identit(ies) not present in chain "
                f"{self.chain_id}, for example {[str(m) for m in missing[:5]]}. "
                "This indicates an interface set computed on a different "
                "structure or chain than the one being partitioned."
            )
        return frozenset(out)


def residue_id(residue: Residue, chain_id: str) -> ResidueId:
    """Build a :class:`ResidueId` from a Biopython residue."""
    _het, seqid, icode = residue.id
    return ResidueId(chain=chain_id, seqid=int(seqid), icode=str(icode))


def extract_chain(
    model: Model,
    chain_id: str,
    params: StructureParams,
    allow_charge_altering_modifications: bool = False,
) -> ChainExtract:
    """Pull one protein chain out of ``model``.

    Raises if the chain is absent, if it contains no protein residues, or if it
    contains a residue type this project does not know how to score. The last
    case is deliberate: an unrecognised residue in a charge calculation is a
    wrong answer, not a warning.
    """
    if chain_id not in {c.id for c in model}:
        raise StructureError(
            f"chain {chain_id!r} not found. Chains present: {chains_present(model)}. "
            "Check the designed_chain and fixed_chain columns of designs.csv "
            "against the deposited author chain identifiers."
        )

    chain: Chain = model[chain_id]
    residues: list[Residue] = []
    letters: list[str] = []
    substitutions: list[tuple[ResidueId, str, str]] = []
    unknown: list[tuple[str, str]] = []
    skipped_het = 0

    for residue in chain:
        hetflag, _seqid, _icode = residue.id
        name = residue.get_resname().strip().upper()
        rid = residue_id(residue, chain_id)

        if hetflag != " ":
            # Waters, ions and ligands. Modified polymer residues such as MSE
            # also arrive with a HETATM flag, so they are rescued here rather
            # than being lumped in with the solvent.
            if name in MODIFIED_RESIDUE_MAP:
                pass
            else:
                skipped_het += 1
                continue

        if name in _THREE_TO_ONE:
            one = _THREE_TO_ONE[name]
        elif name in MODIFIED_RESIDUE_MAP:
            one = MODIFIED_RESIDUE_MAP[name]
            if name in CHARGE_ALTERING_MODIFICATIONS and not allow_charge_altering_modifications:
                unknown.append((name, str(rid)))
                continue
            substitutions.append((rid, name, one))
        else:
            unknown.append((name, str(rid)))
            continue

        if one not in _STANDARD_SET:  # pragma: no cover - defensive
            unknown.append((name, str(rid)))
            continue

        residues.append(residue)
        letters.append(one)

    if unknown:
        charge_altering = sorted({n for n, _ in unknown if n in CHARGE_ALTERING_MODIFICATIONS})
        detail = (
            f"\nOf these, {charge_altering} change the formal charge of their parent "
            "residue, so mapping them onto the parent would corrupt the net charge. "
            "Pass allow_charge_altering_modifications=True only if you have decided "
            "how to score them and have said so in the methods."
            if charge_altering
            else ""
        )
        raise StructureError(
            f"chain {chain_id} contains {len(unknown)} residue(s) this project "
            f"cannot score, for example {unknown[:5]} as (residue name, residue id). "
            f"Extend MODIFIED_RESIDUE_MAP in structures.py, or exclude this "
            f"structure from the set, but do not let it through unscored.{detail}"
        )

    if not residues:
        raise StructureError(
            f"chain {chain_id} contains no standard protein residues. "
            "It may be a nucleic acid or a ligand-only chain."
        )

    sequence = "".join(letters)
    index_of = {residue_id(r, chain_id): i for i, r in enumerate(residues)}

    if len(index_of) != len(residues):
        raise StructureError(
            f"chain {chain_id} contains duplicate residue identities. "
            "This normally means alternate conformations were not collapsed."
        )

    return ChainExtract(
        chain_id=chain_id,
        residues=tuple(residues),
        sequence=sequence,
        index_of=index_of,
        substitutions=tuple(substitutions),
        skipped_heteroatoms=skipped_het,
    )


# ---------------------------------------------------------------------------
# Atom selection and sub-structures
# ---------------------------------------------------------------------------


def heavy_atoms(residue: Residue, params: StructureParams) -> list[Atom]:
    """Heavy atoms of a residue, with alternate locations resolved.

    Hydrogens are dropped when ``params`` says so, which is the default,
    because whether a deposited structure carries them is an artefact of the
    experimental method and would otherwise change both the SASA and the
    contact set.
    """
    out: list[Atom] = []
    for atom in residue:
        if params.ignore_hydrogens and atom.element == "H":
            continue
        if atom.get_altloc() not in params.accepted_altlocs:
            continue
        out.append(atom)
    return out


def submodel(model: Model, chain_ids: Iterable[str], params: StructureParams) -> Model:
    """Return a deep copy of ``model`` containing only ``chain_ids``.

    Used to compute the solvent accessible surface of an isolated chain with
    exactly the same atom selection as the complex calculation, which is what
    makes the delta-SASA difference meaningful. Building the monomer by
    deletion from the complex, rather than by re-parsing, guarantees the two
    calculations see identical coordinates.
    """
    wanted = list(chain_ids)
    present = chains_present(model)
    missing = [c for c in wanted if c not in present]
    if missing:
        raise StructureError(f"chain(s) {missing} not in model, present: {present}")

    clone = copy.deepcopy(model)
    for chain_id in [c.id for c in clone]:
        if chain_id not in wanted:
            clone.detach_child(chain_id)

    if not params.keep_heteroatoms:
        for chain in clone:
            for residue in list(chain):
                hetflag, _, _ = residue.id
                if hetflag != " " and residue.get_resname().strip().upper() not in (
                    MODIFIED_RESIDUE_MAP
                ):
                    chain.detach_child(residue.id)

    if params.ignore_hydrogens:
        for chain in clone:
            for residue in chain:
                for atom in list(residue):
                    if atom.element == "H":
                        residue.detach_child(atom.get_id())

    return clone
