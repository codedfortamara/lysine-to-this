# RCSB interface arm: handover

Tamara Dinneen, 29 July 2026.

This covers the RCSB protein-protein complex work: what is finished, what is
not, and exactly where the unfinished part is stuck. Everything computational,
nothing experimental.

## Summary

Three findings are complete on all 55 held-out complexes and need no GPU. They
address Limitations item (i) from a direction the paper does not currently take:
not only whether the interface survives the charge dial, but **why it is cheap
to move charge at an interface in the first place**.

The AlphaFold2-Multimer refold, the thing item (i) names as "the natural fix", is
built, tested, and **not working**. The pipeline runs end to end; the prediction
jobs time out. Details in "Where it is stuck" below.

---

## 1. Where the imposed charge lands

Each designed chain is split into three regions, tested in this order: buried
core (relative solvent accessibility below 0.25), interface, remaining surface.
Net charge is regressed on beta per region, and the slope divided by that
region's share of residues. A value of 1.0 means a region absorbing charge
exactly in proportion to its size.

**Interface 1.19, surface 1.44.** Paired difference **−0.25**, 95% CI
[−0.30, −0.21], negative in **50 of 55** complexes.

The interface takes roughly a quarter less than its proportional share. The
imposed charge is going somewhere else.

## 2. It is not a burial artefact

Interface residues are more buried than average, which could produce the above
on its own. Region labels were permuted 10,000 times per chain, with every
interface residue restricted to swap only with surface residues of matched
relative solvent accessibility.

**Observed minus null: −0.23**, CI [−0.30, −0.17]. Significant in **50 of 51**
chains. Mean rSASA matching error 0.012, so the matching is tight.

## 3. It is caused by partner context

The decisive experiment. The partner chain is deleted from the structure before
ProteinMPNN sees it, and every design is regenerated from scratch. Interface
membership is still defined on the **native complex** in both arms, so the two
are compared over identical positions and the only thing that differs is what
the model can see.

| arm | interface minus surface | 95% CI |
|---|---|---|
| partner present | −0.16 | [−0.23, −0.09] |
| partner removed | **+0.07** | [+0.04, +0.09] |

**Effect of removing the partner: +0.22**, CI [+0.15, +0.29], weakened in **38
of 44** complexes.

The effect does not merely weaken, it **reverses**. ProteinMPNN spares the
contact patch because it can see what is binding there, not because those
positions are intrinsically different.

**This is the part I would put in the paper.** It is a mechanism for the
existing headline claim rather than another check on it: charge control is
essentially free within the usable band partly because the model already
protects the binding site. Nothing in the current draft says this.

## 4. Supporting analyses

- **Salt bridges.** Cross-chain oppositely charged pairs, normalised by
  available charged-pair opportunities. At beta = −3, +0.00071 per opportunity,
  CI [+0.00036, +0.00113], excludes zero. At beta = +3, CI includes zero. The
  response is asymmetric: adding negative charge creates cross-chain bridges,
  adding positive charge does not. **See "Needs your decision" below.**
- **Uversky charge-hydropathy.** Designs cross the 2000 boundary as charge is
  added, because charged residues raise net charge and lower mean hydropathy at
  once. At beta = 0, 14.5% fall on the disordered side; at every non-zero beta,
  100%. Fold rate is 77% on the folded side against 43% on the disordered side,
  so it has real signal, but it classifies 228 of 275 designs as disordered and
  43% of those fold. Reported as a stress indicator, explicitly not a
  classifier, with a note that the boundary was fitted to natural proteins.
- **Electrostatic complementarity**, and **interface-definition agreement**
  between the delta-SASA and contact-distance definitions.

## 5. AlphaFold2-Multimer refold: incomplete

37 jobs across 14 complexes, 9 with the complete beta −1.5 / 0 / +1.5 series.
No beta = ±3 job has ever completed.

Paired change in ipSAE against the same complex at beta = 0:

| beta | change | 95% CI | degraded |
|---|---|---|---|
| −1.5 | +0.059 | [−0.162, +0.265] | 2 of 9 |
| +1.5 | +0.003 | [−0.254, +0.261] | 3 of 10 |

**Both intervals include zero on n = 9.** This is preliminary and does not
support an interface-survival claim. It is consistent with no degradation at
moderate charge shifts, and it cannot exclude a moderate effect.

Two further caveats on these 37:

- They were folded on a **CPU**. A cuDNN version mismatch made JAX fall back
  silently while being billed as a GPU. AlphaFold on CPU and GPU are not
  numerically identical, so these should be recomputed on one backend before
  use. Every result now records `jax_device_kind` so this is checkable.
- `interface_rmsd_a` is null in all 37 rows. The prediction step looked for the
  native structure at a path nothing ever wrote. `scripts/14_interface_rmsd.py`
  computes it locally from `data/native/` and the downloaded predictions, and
  has not yet been run against a complete set.

## Where it is stuck

Prediction jobs hit a 1200 second timeout having produced **no ColabFold output
whatsoever**: two oneDNN lines at import, then twenty minutes of silence, then
the kill. Reproduced identically on both L4 and A100-40GB, on complexes under
200 residues that should fold in one to two minutes.

The only step between JAX importing and ColabFold's first print is CUDA device
initialisation. Modal documents an intermittent CUDA-init hang on some
instances. `require_gpu_backend()` now runs `nvidia-smi` before asking JAX for
devices, so a container with no working driver should now fail in seconds with a
readable message rather than hanging. That change has not yet been observed to
run.

What has been ruled out:

- **Not the cuDNN mismatch.** Pinned to `nvidia-cudnn-cu12>=8.9,<9.0` to match
  `jaxlib 0.4.23+cuda12.cudnn89`, and the build log confirms 8.9.7.29 installs.
- **Not the MSA search.** Moved to a CPU-only `prefetch`. All 55 alignments are
  cached; `msa_seconds` is 0.0 in every recent record.
- **Not memory.** These are the smallest complexes in the set, and
  `TF_FORCE_UNIFIED_MEMORY` has been removed so an OOM would fail fast.
- **Not the alignment format.** Round-tripped through ColabFold's own
  `unserialize_msa`; the paired block carries the query row that
  AlphaFold-Multimer requires.

**Where I would look next**, with GPU access: run one prediction interactively
rather than through the Modal function, and see where it blocks. Everything
needed is in `modal_app/af2_multimer.py::predict`. My guess is CUDA
initialisation, but I have guessed wrong on this three times and would not spend
money on the guess.

## Running it

```
uv pip install -e .
python -m pytest                     # 521 tests

python scripts/01_define_interfaces.py --config config.toml
python scripts/02_partition_charge.py
python scripts/04_join_and_analyse.py
python scripts/07_burial_control.py
python scripts/09_partner_ablation.py
python scripts/05_figures.py
```

The AlphaFold grid runs on Modal, or on a single GPU via
`notebooks/af2_colab.ipynb`. The notebook is **generated** from the Modal source
by `scripts/make_colab_notebook.py`, and a test fails if the two drift apart, so
the two backends cannot compute a metric differently.

`scripts/13_af2_state.py` reports what is actually on disk: which complexes came
back, at which beta, which metrics are populated, and what remains. It reads
local files only.

Every output carries a manifest with input file hashes, git commit, package
versions, seeds and parameters. Figures embed the manifest hash.

## What is not in this directory, and where to get it

Everything needed to reproduce the three findings is here, except the inputs
that are yours already.

**Your own tables.** `designs.csv` and `test_set.csv` are gitignored throughout:
they are your data and not mine to redistribute. Put them in `data/raw/`. The
loaders validate the schema and fail loudly on anything missing or malformed
rather than coercing it.

**Native structures.** Also not included. `python scripts/00_fetch_natives.py`
downloads them from the RCSB by `pdb_id` straight from `test_set.csv`, so
nothing has to be moved by hand.

**The ablation designs are included**, in `data/ablation/paired` and
`data/ablation/isolated`, because regenerating them means a full ProteinMPNN
pass over all 55 complexes with the partner chain deleted. Without them the
causal result cannot be checked without a GPU and several hours.

**The MSA cache is not, and cannot be.** It lives on a Modal volume tied to my
account. `modal run modal_app/af2_multimer.py::prefetch` rebuilds it: 55
searches against the free MMseqs2 server, run on CPU containers four at a time,
which cost 0.86 USD and about 15 minutes when I did it. Do this **before** any
GPU run. The prediction path refuses to search on a GPU by default, because
doing so bills an HTTP queue at GPU rates and was the first expensive mistake of
this project.

**The 37 AlphaFold results** are in `results/af2_metrics.csv` if that file is
present. If it is missing, the grid never completed on the machine that
assembled this and there is nothing to collect.

## Needs your decision

**1. Salt bridges overlap.** The paper reports a proxy count of oppositely
charged C-beta atoms within 6 Å, 11.3 for designs against 9.1 for native,
n ≈ 170, p = 7×10⁻⁵. Mine uses a different distance definition and normalises
per available charged pair. These measure related but not identical things and
should be reconciled or clearly separated, not printed side by side as though
independent.

**2. Cohort.** `test_set.csv` contains 55 complexes, all `split: test`, each in
its own MMseqs cluster, and `designs.csv` has exactly 55 × 5 rows. The paper
describes 114 RCSB complexes for the controller evaluation. I have assumed the
55 are the held-out slice of the 114. Worth confirming before the sample size is
stated.

## What is not claimed

- All evidence is computational.
- ipSAE is from a preprint and is not peer reviewed. actifpTM is a peer-reviewed
  alternative and was not computed.
- The mixed alignment treatment, native partner with an MSA and the redesigned
  chain as a single sequence, means no homologues are paired across the
  interface. Interface confidence therefore derives from the partner's
  evolutionary signal and the design's own sequence alone.
- Predictions use one model and one seed. AlphaFold confidence is not a binding
  assay.
- The fold-quality control uses a **relative** criterion, each design against
  its own beta = 0 reference, not an absolute pLDDT floor. The conventional
  floor of 70 comes from MSA-backed predictions, and the designed chain is
  deliberately folded single-sequence, so 70 rejected 30 of 31 rows on regime
  alone. Both are reported.
