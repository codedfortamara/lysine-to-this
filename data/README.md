# Data contract

This directory holds the inputs. **Nothing in it is committed** apart from this
file and the directory markers: the design tables are the collaborator's data,
and the native structures are re-fetchable from their identifiers.

The contract below is enforced in `src/interface_charge/contracts.py` and
checked on every load. If a delivered file does not satisfy it, the loaders
raise and name every problem at once. They never coerce a value, never fill a
blank and never drop a row.

---

## Open question, blocking

**Which chains were redesigned and which were held fixed?**

The whole analysis depends on this. "Interface charge" means the charge carried
by the *designed* chain's interface residues. If the designed and fixed chains
are swapped, the measured trend inverts and the result is not merely noisy, it
is backwards.

What is needed, per row of `designs.csv`:

- `designed_chain`: the author chain identifier(s) ProteinMPNN was free to
  redesign, as deposited in the PDB entry. Author identifiers, not chain
  indices and not label_asym_id.
- `fixed_chain`: the author chain identifier(s) held at the native sequence.

If more than one chain was redesigned per complex, the contract already parses a
comma separated list, but note that `sequence` is a single column, so a row with
several designed chains has no unambiguous reading. Split those into one row per
designed chain, or tell us the convention and the loader will be adjusted.

## Open question, non-blocking

**Which pKa set produced `net_charge_reported`?**

EMBOSS and Bjellqvist disagree by roughly 0.5 to 1.5 charge units on a
hundred-residue chain at pH 7.4. `scripts/02_partition_charge.py` reconciles our
computed charge against that column and reports which combination of definition,
pKa set and terminus handling reproduces it, so this is diagnosable from the
data. Knowing the answer in advance simply saves a round trip. If the column came
from Biopython's `ProtParam`, it is Bjellqvist.

---

## `data/raw/designs.csv`

One row per design: a single ProteinMPNN sample of one chain of one complex at
one value of the charge bias `beta`.

| Column | Type | Meaning |
| --- | --- | --- |
| `pdb_id` | four-character string | RCSB identifier of the native complex the design was made from. Upper-cased on load; see "Permitted normalisations". |
| `designed_chain` | chain id, or comma separated chain ids | Author chain identifier(s) ProteinMPNN was allowed to redesign. |
| `fixed_chain` | chain id, or comma separated chain ids | Author chain identifier(s) held at the native sequence. |
| `beta` | finite float | The signed scalar logit bias. Added to the K and R logits and subtracted from the D and E logits at every position. `beta = 0` recovers vanilla ProteinMPNN. |
| `replicate` | non-negative integer | Distinguishes repeated samples at the same `(pdb_id, designed_chain, beta)`. |
| `sequence` | string over the standard twenty amino acids | The designed sequence of `designed_chain`, in the same residue order as the native chain. |
| `net_charge_reported` | finite float | Net charge as computed by the collaborator's pipeline. Used only for reconciliation, never as an analysis input. |

### Table-level invariants

- `(pdb_id, designed_chain, beta, replicate)` is unique. A duplicate would
  double-weight one design in every average.
- `designed_chain` and `fixed_chain` do not overlap.
- All designs sharing a `(pdb_id, designed_chain)` have the same sequence
  length. ProteinMPNN is fixed-backbone, so a length change means rows have been
  mixed up or a chain was mislabelled.
- Sequence length must equal the native chain length. This is checked in
  `scripts/02_partition_charge.py` once the structures are available, because it
  is the assumption that lets native residue indices address design positions.

### Example

```csv
pdb_id,designed_chain,fixed_chain,beta,replicate,sequence,net_charge_reported
1BRS,A,D,0.0,0,AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR,2.0
1BRS,A,D,1.5,0,AQVINTFKGVAKYLQTYHKLPKNYITKSEAKALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR,8.0
```

The sequences above are illustrative of the *format only*. They are not real
design output and no such file exists in this repository.

---

## `data/raw/test_set.csv`

One row per native complex, with the MMseqs2 clustering used to hold the test
split out at 30 percent sequence identity.

| Column | Type | Meaning |
| --- | --- | --- |
| `pdb_id` | four-character string | RCSB identifier of the native complex. |
| `chains` | comma separated chain ids | The author chain identifiers making up the complex to analyse. Exactly two are required. |
| `mmseqs_cluster_id` | non-empty string | Cluster assigned by MMseqs2 at 30 percent sequence identity. |
| `split` | `train` or `test` | Which side of the leakage-free split this complex falls on. |

### Table-level invariants

- `pdb_id` is unique.
- Every complex names exactly **two** chains. The interface analysis is defined
  between a pair. A complex with three or more has to be split into named pairs,
  and which pair is the biologically relevant one is a judgement call that gets
  recorded rather than defaulted.
- No MMseqs2 cluster appears in both splits. A cluster straddling the split is
  sequence leakage and it invalidates the held-out claim.

### Cross-table invariants

- Every `pdb_id` in `designs.csv` appears in `test_set.csv`.
- Every chain named in `designs.csv` is one of the chains `test_set.csv` lists
  for that complex.

### Example

```csv
pdb_id,chains,mmseqs_cluster_id,split
1BRS,"A,D",cluster_0007,test
```

---

## `data/native/`

Native structures downloaded from the RCSB by `pdb_id`, one file per complex, in
mmCIF by default. Populated by `scripts/00_fetch_natives.py`. Gitignored.

mmCIF rather than legacy PDB because the PDB format cannot represent large
complexes or multi-character chain identifiers. Change
`structure.native_format` in the config to use `pdb` instead.

**This requires network access to `files.rcsb.org`.** That host was blocked by
the network egress policy in the environment this repository was scaffolded in.
Check it before scheduling a run:

```bash
curl -sSI https://files.rcsb.org/download/1BRS.cif | head -1
```

---

## Permitted normalisations

The loaders perform exactly two transformations, both documented here because
"never coerce silently" means the exceptions have to be written down:

1. **`pdb_id` is upper-cased.** The RCSB is case insensitive, and every
   downstream file name would otherwise depend on how the identifier was typed.
2. **Surrounding whitespace is stripped** from every field, and `sequence` is
   upper-cased with internal spaces removed.

Everything else raises. In particular:

- An empty cell raises. It is not filled.
- `NA`, `N/A`, `NaN`, `none`, `null` and `-` raise as placeholders. They are not
  read as missing values, which is what pandas would do by default.
- A non-standard residue code such as `X` raises. It is not treated as neutral,
  because that would shift a reported net charge with nothing visible to show
  for it.
- A `replicate` written as `3.0` raises rather than being rounded to `3`.
- A column not in the contract raises, unless `--allow-extra-columns` is passed.
  If you use that flag, add the column to this document.

## Structure parsing

Applied when reading native structures, not when reading the tables:

- Waters, ions and ligands are excluded. Ions at an interface can matter
  physically, but including them would make a charge partition depend on
  crystallisation conditions.
- Hydrogens are stripped, because whether a deposited structure carries them is
  an artefact of the experimental method.
- Alternate conformations after the first are dropped.
- Modified residues from a documented list (selenomethionine and similar) are
  mapped onto their parent amino acid. Every substitution actually applied is
  recorded in the run manifest and in `results/interfaces.csv`, so the mapping is
  visible in the output rather than invisible in the code.
- Modified residues whose formal charge differs from the parent, such as the
  phosphorylated serine, threonine and tyrosine, raise rather than being mapped.
  Scoring them as their parent would corrupt the net charge. Opting in is
  explicit and has to be justified in the methods.
