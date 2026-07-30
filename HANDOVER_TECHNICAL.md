# Handover brief for a fresh model

You are picking up an unfinished piece of computational work. The previous
assistant got the analysis done and failed on the compute. Read this whole file
before running anything, because the failures here cost real money and some of
them are non-obvious.

`HANDOVER.md` is the same story written for the co-author, without the
engineering detail. This file is the technical one.

## Who and what

Tamara Dinneen is co-authoring a PSB 2027 paper with Mohammed Sameer Syed:
"ZetaDial: dialing net charge of protein binders at inference time for
therapeutic developability". ZetaDial is a closed-loop per-protein charge
controller with an adaptive-slope (secant) update, which is what distinguishes
it from open-loop logit-bias methods.

Tamara owns the **RCSB protein-protein complex arm**. Mohammed's repo is
`SyedMohammedSameer/ZetaDial`. His instruction: pull from the latest commit and
work on a new branch. Her repo is `codedfortamara/lysine-to-this`, branch
`claude/lysine-to-this-plan-kfmjfn`.

She is not a biologist and neither is Mohammed. Do not explain results in
domain jargon. She sets the compute budget; he has none.

## Her standing constraints, which are not negotiable

- Python 3.11, `uv`, `ruff`, type hints, `pathlib` not string paths.
- All parameters live in `src/interface_charge/config.py` or `config.toml`.
  Scripts stay thin.
- **British English. No em dashes anywhere** in code, comments, docs or
  committed text.
- **No synthetic, placeholder or example data outside `tests/fixtures/`.**
- Loaders validate schema and **raise loudly** on anything missing or
  malformed. Never silently coerce, fill or drop.
- Every script runs end to end on the test fixture and fails cleanly with a
  useful message when real data is absent.
- `complementarity.py` must not use APBS or any Poisson-Boltzmann solver. She
  will not cite a tool she has not run.
- Every output carries a manifest with input file SHA256s, git commit SHA,
  package versions, seeds, parameters and a UTC timestamp. Figures embed the
  manifest hash. She called this non-negotiable.
- **Give her PowerShell commands.** She is on Windows. Not bash, not cmd.
- Results **are** committed, including `results/` and `data/ablation/`, via
  `git add -f`. An earlier instruction said not to; she overrode it.

## State of the work

521 tests pass. `python -m pytest`.

### Complete, on all 55 held-out complexes, no GPU required

The framing is *why* charge control is cheap at an interface, not just whether
the interface survives it. This addresses Limitations item (i) of the paper from
a direction the current draft does not take.

1. **Where the imposed charge lands.** Each designed chain is split into buried
   core (relative SASA below 0.25), interface, then remaining surface, tested in
   that order. Net charge regressed on beta per region, slope divided by that
   region's share of residues, so 1.0 means absorbing charge in proportion to
   size. Interface 1.19, surface 1.44. Paired difference **-0.25**, 95% CI
   [-0.30, -0.21], negative in 50 of 55. Script `02`, `04`.

2. **Not a burial artefact.** Region labels permuted 10,000 times per chain,
   each interface residue restricted to swap only with surface residues of
   matched relative SASA. Observed minus null **-0.23**, CI [-0.30, -0.17],
   significant in 50 of 51 chains, mean rSASA matching error 0.012. Script `07`.

3. **Caused by partner context. This is the headline.** The partner chain is
   deleted before ProteinMPNN sees the structure and every design is
   regenerated. Interface membership stays defined on the **native complex** in
   both arms, so the two are compared over identical positions and the only
   difference is what the model can see. Partner present **-0.16**
   [-0.23, -0.09]. Partner removed **+0.07** [+0.04, +0.09]. Effect of removal
   **+0.22** [+0.15, +0.29], weakened in 38 of 44. The effect **reverses**.
   ProteinMPNN spares the contact patch because it can see what binds there, not
   because those positions are intrinsically different. Scripts `09`, `10`.

   This is a **mechanism** for the paper's existing headline claim rather than
   another check on it. Nothing in the current draft says it.

Supporting: salt bridges (`06`), Uversky charge-hydropathy (`08`),
electrostatic complementarity (`03`), interface-definition agreement.

### Not complete: the AlphaFold2-Multimer refold

This is the thing Limitations item (i) calls "the natural fix". It is built and
tested. It does not run.

37 jobs exist across 14 complexes, 9 with a complete beta -1.5 / 0 / +1.5
series. **Treat all 37 as scrap.** Reasons below. No beta = +/-3 job has ever
completed, and beta +/-3 is part of the grid Mohammed defined.

## The failure, and everything ruled out

Prediction jobs on Modal hit the 1200 second timeout having produced **no
ColabFold output whatsoever**: two oneDNN lines at JAX import, then twenty
minutes of silence, then the kill. Reproduced identically on L4 and A100-40GB,
on complexes under 200 residues that should fold in one to two minutes.

Ruled out:

- **Not the cuDNN mismatch.** That was a separate real bug, see below. Pinned to
  `nvidia-cudnn-cu12>=8.9,<9.0` and the build log confirms 8.9.7.29 installs.
- **Not the MSA search.** Moved to a CPU-only `prefetch` entrypoint. All 55
  alignments are cached and `msa_seconds` is 0.0 in every recent record.
- **Not memory.** These were the smallest complexes in the set, and
  `TF_FORCE_UNIFIED_MEMORY` has been removed so an OOM fails fast.
- **Not the alignment format.** Round-tripped through ColabFold's own
  `unserialize_msa`. The paired block carries the query row that
  AlphaFold-Multimer requires, or it raises.

Remaining hypothesis, unverified: CUDA device initialisation, which is the only
step between JAX import and ColabFold's first print. Modal documents an
intermittent CUDA-init hang. `require_gpu_backend()` now runs `nvidia-smi` with
a 120 s timeout **before** calling `jax.devices()`, so a container with no
working driver should fail in seconds with a readable message. That change has
never been observed to run.

Claude guessed wrong on this three times and stopped spending. Do not spend more
on Modal to test a guess.

## The immediate next step, which is not Modal

Mohammed's advice was Colab from the start, and he was right. Claude told her
the complexes were too big for Colab without checking, which is the single most
expensive error in this project: it pushed the whole run onto Modal for no
reason.

The actual numbers, computed from `results/interface_definitions.json` and
`data/raw/test_set.csv`: 55 complexes, total residues per complex median **395**,
minimum 99, maximum **1257** (2ZXE and 3A3Y, then 1QBK at 1070). Only three
exceed 1000. A 24 GB L4 clears 52 of 55. A 40 GB A100 clears all 55.

`notebooks/af2_colab.ipynb` is ready. It is **generated** from
`modal_app/af2_multimer.py` by `scripts/make_colab_notebook.py`, and a test
fails if the two drift, so both backends cannot compute a metric differently. Do
not hand-edit the generated cells; edit the source and regenerate. It is
resumable, writes each result to Drive as it completes, skips what is already
done, runs smallest first, and records skips rather than dropping them.

To run it, Drive needs `designs.csv`, `test_set.csv` and
`interface_definitions.json` in `MyDrive/interface_charge/`. Cell 6 prints
found or MISSING for each before any GPU time is spent.

`designs.csv` has 275 rows, 55 complexes at each of beta -3.0, -1.5, 0.0, +1.5,
+3.0, so the full grid including +/-3 is picked up with no extra work.

Two pins in the install cell are correctness constraints, not preferences:
`colabfold 1.5.5` requires `biopython<1.83` and `numpy<2`, and `dm-haiku 0.0.10`
imports `jax.linear_util` which was removed in jax 0.4.24, so jax must stay at
or below 0.4.23 or every prediction dies at import.

## Money traps already paid for

Spend to date is roughly 274 USD against a project that should have cost about
10. Each of these was a real bill.

| What happened | Cost shape | Fix in place |
|---|---|---|
| `nvidia-cudnn-cu12` unbounded, pip took 9.x, JAX fell back to CPU silently while Modal billed a GPU | all 37 existing results are CPU, at GPU prices | pin `>=8.9,<9.0` in the same pip layer as jax; `require_gpu_backend()` raises |
| MMseqs2 search run inside the GPU container | billed an HTTP queue at GPU rates, 2 h timeouts | CPU-only `prefetch`; all 55 alignments for 0.86 USD; the predict path refuses to search on GPU by default |
| `TF_FORCE_UNIFIED_MEMORY=1` | turned a 90 s OOM into 2 h of billed thrash | removed, deliberately absent |
| Budget guard counted job wall clock, not container time | undercounted about 2x | `billing_overhead = 2.1`, measured |
| Failures charged at the successful mean | 80x undercharge, 22 cents against 17.64 USD | `record_failure()` always charges full timeout times attempts |
| No run lock | three concurrent drivers paying for the same work | heartbeated `run_lease.json`, stale after `max(4 * timeout_s, 3600)` |
| Guard reset per launch | a 60 USD cap meant 60 USD *per launch* | cumulative `spend_ledger.json`, refuses to run on a corrupt ledger rather than resetting to zero |
| Metrics defect forced a refold | paid twice for the same structure | `folded.json` checkpoint, written only after both scores JSON and structure validate |

Also fixed, not money but correctness: residues were paired by sequence id, and
1FS2 starts at 105 while 1O9S starts at 117, so pairing is now by global
alignment with a per-pair identity check (`pair_residues_by_sequence`).
Interface membership is keyed on `(seqid, icode)` so 52, 52A and 52B do not
collapse.

## Caveats that must survive into any writeup

- All evidence is computational.
- ipSAE is from a preprint (Dunbrack 2025) and is not peer reviewed. actifpTM is
  a peer-reviewed alternative and was not computed. Note that ipSAE's d0 derives
  from interface residue count, not total chain length, and `_d0_scalar` and
  `_d0_array` differ deliberately.
- The alignment treatment is mixed: native partner keeps its MSA, the redesigned
  chain is single-sequence. No homologues are paired across the interface, so
  interface confidence comes from the partner's evolutionary signal and the
  design's own sequence alone.
- One model, one seed. AlphaFold confidence is not a binding assay.
- Fold quality uses a **relative** criterion, each design against its own
  beta = 0 reference, not an absolute pLDDT floor. Single-sequence predictions
  run systematically lower in pLDDT and the conventional floor of 70 comes from
  the MSA-backed regime, so 70 rejected 30 of 31 rows on regime alone. Both are
  reported. `PRIMARY_METRIC = "ipsae_d0res"`, and the analysis in
  `scripts/12_af2_interface_survival.py` is **pre-registered**, with four
  outcomes written before the data existed. Do not rewrite it to fit results.
- The 37 existing AF2 rows have `interface_rmsd_a` null throughout: the
  prediction step looked for the native structure at a path nothing ever wrote.
  `scripts/14_interface_rmsd.py` computes it locally and has not been run on a
  complete set.

## Open questions for Mohammed, not for you to decide

1. **Salt bridges overlap.** The paper reports oppositely charged C-beta atoms
   within 6 A, 11.3 for designs against 9.1 native, n about 170,
   p = 7e-5. Tamara's measure uses a different distance definition and
   normalises per available charged-pair opportunity. Related but not identical.
   They must be reconciled or clearly separated, not printed side by side as
   though independent.
2. **Cohort size.** `test_set.csv` has 55 complexes, all `split: test`, each in
   its own MMseqs cluster, and `designs.csv` is exactly 55 x 5. The paper
   describes 114 RCSB complexes for the controller evaluation. The working
   assumption is that the 55 are the held-out slice of the 114. Confirm before
   any sample size is stated.

## Orientation

- `HANDOVER.md` is the version of this written for Mohammed.
- `scripts/13_af2_state.py` reports what is actually on disk: which complexes
  came back, at which beta, which metrics are populated, what remains. Local
  files only, no Modal, no cost. **Run this first.**
- `scripts/15_package_for_upstream.py` assembles the `rcsb_interface/` directory
  for Mohammed's repo.
- `modal_app/af2_multimer.py` is about 2600 lines and holds the lease, ledger,
  circuit breaker, MSA prefetch, prediction and metrics.
- `src/interface_charge/statistics.py` has the bootstrap, Spearman, permutation
  test and cluster resampling.

## What to do first

1. Run `scripts/13_af2_state.py` and read it. Do not trust this document over
   what is on disk.
2. Run `python -m pytest`. 521 tests should pass.
3. Get the Colab run going. That is the only outstanding blocker.
4. Do not spend on Modal without a specific, testable reason.
