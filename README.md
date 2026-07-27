# interface-charge-rcsb

Interface-resolved charge analysis and complex-aware refolding for the RCSB arm
of *Controllable electrostatic protein design* (PSB 2027).

## What this is

The paper's controller adds a signed scalar bias `beta` to the K and R logits of
ProteinMPNN and subtracts it from the D and E logits, which shifts a design's
net charge. Its current evaluation on RCSB complexes folds **each chain in
isolation** with ESMFold and scores self-consistency RMSD. The paper says so
itself, in Limitations (i):

> The foldability metric folds each chain in isolation, weakening the RCSB
> *complex* numbers; the Cas13 monomer results are cleaner, and a complex-aware
> (AlphaFold-multimer) refold is the natural fix.

Per-chain folding cannot see the protein-protein interface, which is exactly
where a perturbed surface charge does its damage. This repository is that fix,
plus the question the fix makes answerable.

**Question 1: where does the charge land when the dial turns?**
Partition net charge into interface, non-interface surface and buried core, and
measure how each partition responds to the charge shift. The hypothesis is that
the interface is buffered relative to the bulk surface: ProteinMPNN, which sees
the full complex context, preferentially spends the imposed charge on
non-interface surface, and the interface only starts absorbing it once the bulk
surface is saturated. If that holds it converts a limitation into a positive
result, because it says the controller is interface-sparing by default.

**Question 2: does the interface survive?**
Replace per-chain ESMFold with complex-aware AlphaFold2-Multimer refolding and
report interface RMSD, interface PAE and interface pTM against charge shift.
This is the direct answer to Limitation (i) and it is what makes the RCSB
complex numbers load-bearing rather than caveated.

## Status

The input data has not arrived yet. Everything here is built against the
documented contract in [`data/README.md`](data/README.md) and validated on a
committed structural fixture.

| Component | State |
| --- | --- |
| `contracts.py` schema validation | implemented, tested |
| `structures.py` fetch, parse, chain handling | implemented, tested |
| `interface.py` both interface definitions | implemented, tested |
| `charge.py` both charge definitions, partitioning | implemented, tested |
| `complementarity.py` | implemented, tested |
| `provenance.py` run manifests | implemented, tested |
| `scripts/00` to `scripts/03` | implemented, run end to end on the fixture |
| `scripts/04`, `scripts/05` | documented skeletons, see "Deliberately unfinished" |
| `modal_app/af2_multimer.py` | implemented, `--dry-run` works without Modal installed |

No result, figure or table in this repository is derived from real design data,
because there is not any yet. Nothing is committed to `results/` or `figures/`.

## Environment blockers found while building this

Two things in the container this was scaffolded in will bite on any machine
with the same policy, and both need checking wherever the pipeline is actually
run:

1. **`files.rcsb.org` is blocked by the network egress policy.** Every native
   structure comes from there, so `scripts/00_fetch_natives.py` cannot run in
   that environment. The script itself is correct and retries properly; the
   host simply has to be reachable. Verify with
   `curl -sSI https://files.rcsb.org/download/1BRS.cif` before scheduling a run.
2. **The committed structural fixture is not 1BRS.** Barnase-barstar was the
   intended fixture and remains the right one, but it could not be downloaded
   for the reason above. The fixture used instead is
   `tests/fixtures/7ddo_ace2_rbd.pdb`, a genuine two-chain protein-protein
   complex (human ACE2 chain A with SARS-CoV-2 RBD chain C) obtained from the
   Biopython test corpus on PyPI. It happens to be a good substitute: RBD is one
   of the paper's own three BindCraft targets. To switch to 1BRS once RCSB is
   reachable, run `python tests/fixtures/build_fixture.py --pdb-id 1BRS
   --chains A D`; the tests pick it up automatically and will then run against
   both.

## Open questions for Mohammed

These block or change the analysis. The first one blocks it outright.

1. **Which chains were redesigned and which were held fixed?**
   The entire partition analysis rests on this. "Interface charge" means the
   charge of the *designed* chain's interface residues, and if the designed and
   fixed chains are swapped the trend inverts. `designs.csv` needs
   `designed_chain` and `fixed_chain` populated per row, using the deposited
   author chain identifiers, not chain indices. If more than one chain was
   redesigned per complex, say so and the contract will take a comma separated
   list (it already parses one).
2. **Which pKa set produced `net_charge_reported`?**
   EMBOSS and Bjellqvist disagree by roughly 0.5 to 1.5 charge units on a
   hundred-residue chain. `scripts/02_partition_charge.py` reconciles our
   computed charge against that column and reports which combination of
   definition and pKa set reproduces it, so this is diagnosable rather than
   blocking, but knowing the answer saves a round trip. If the column came from
   Biopython `ProtParam`, it is Bjellqvist.
3. **Which `beta` grid and how many replicates per `(pdb_id, beta)`?**
   This sets the GPU bill directly. See "Modal budget" below.
4. **Are the 26 RCSB complexes all two-chain?**
   The interface analysis is defined between a pair of chains. Any complex with
   three or more has to be split into named pairs, and which pair is the
   biologically relevant one is a judgement call that has to be recorded rather
   than defaulted.

## Data contract

Full contract in [`data/README.md`](data/README.md), enforced in
`src/interface_charge/contracts.py`. In brief:

- `data/raw/designs.csv`: `pdb_id`, `designed_chain`, `fixed_chain`, `beta`,
  `replicate`, `sequence`, `net_charge_reported`
- `data/raw/test_set.csv`: `pdb_id`, `chains`, `mmseqs_cluster_id`, `split`
- `data/native/`: structures fetched from the RCSB by `pdb_id`, not committed

Loaders validate the schema and raise on anything missing or malformed. They
never coerce, fill or drop. All errors in a table are reported at once rather
than one exception at a time.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest                     # runs against the committed fixture
uv run ruff check . && uv run ruff format --check .
```

Once the data arrives:

```bash
uv run python scripts/00_fetch_natives.py       --test-set data/raw/test_set.csv
uv run python scripts/01_define_interfaces.py   --test-set data/raw/test_set.csv
uv run python scripts/02_partition_charge.py    --designs data/raw/designs.csv
uv run python scripts/03_complementarity.py     --designs data/raw/designs.csv
```

Every script writes a JSON manifest beside its output. Every script fails with
a message naming the missing file and the expected contract when the real data
is absent.

## Modal budget

`modal_app/af2_multimer.py --dry-run` prints the job count and an estimated
GPU-hour cost without launching anything, and it works without `modal`
installed. The real grid is **55 complexes at 5 betas, one replicate each: 275
jobs**, spanning 99 to 1257 total residues.

Against the ceiling of $122:

| Treatment | Minutes per job | Estimate |
| --- | --- | --- |
| Single sequence throughout | 8 | about $87 |
| Native partner keeps its MSA | 8 + 3 | about $119 |
| Wave 1 only, beta 0 and +/-1.5 | 8 + 3 | about $72 |

Running the native partner with an MSA is the right call scientifically, but it
costs about 38 percent more and leaves only a few dollars of headroom. The dry
run reports that headroom against `--budget-usd` and says so when it is tight.

Both figures are *planning assumptions, not measurements*, and the MSA overhead
is the one most worth replacing with a real number because it applies to every
job in the grid.

### Pilot first

```
modal run modal_app/af2_multimer.py --pilot 4
```

This runs four complexes at two charge settings each, eight jobs and roughly
four dollars, then re-costs the full grid from what it observed and stops. The
results stay on the volume and count towards the grid, so nothing is wasted.

Two things it gets right that a naive pilot does not:

* **It samples across the size distribution**, not just the largest complexes.
  Canary ordering runs the largest first, which is correct for surfacing
  out-of-memory and timeout quickly, but a median taken from the largest
  complexes extrapolates to a cost well above what the grid will pay. The
  largest complex is still included, so the risk check is not lost.
* **It runs each complex at more than one beta.** The MMseqs2 search is paid
  once per complex and cached for the other four charge settings, so a pilot of
  distinct complexes measures only cold jobs. Four jobs in five in the real grid
  are warm. Measuring both is the difference between a projection that is right
  and one that is high by roughly the MSA overhead times four fifths of the
  grid.

If the projection comes in over budget, run wave 1 only with `--wave 1`. Wave 2
(beta +/-3) is the part to drop: the upstream data shows only 10 to 18 percent
of those designs fold at all, so it largely confirms that a broken monomer is
still broken.

The per-call timeout is a single constant, `AF2Params.timeout_s`, set to two
hours, sized for the largest complex rather than the median.

## Five-day plan

Anchored to the actual deadline: submission 3 August, Mohammed wants the work
complete by 28 July with two days for edits and two for review.

**Day 1, 26 July. Unblock and reconcile.**
Get `designs.csv` and `test_set.csv` from Mohammed with the designed and fixed
chains populated. Run `scripts/00` and `scripts/02` immediately. The
reconciliation report tells us within minutes whether our charge calculation
agrees with his, which is the single highest-value early check: if it disagrees,
everything downstream is suspect and we want to know on day 1, not day 4.

**Day 2, 27 July. Question 1 end to end.**
Interfaces defined on all 26 natives, charge partitioned across the beta grid,
complementarity computed. This is the arm that does not need a GPU, and it is
the arm most likely to produce the paper's new figure. Deliverable: the
partition-versus-charge-shift table, with the interface buffering effect either
visible or not. Either outcome is publishable; a null result here is still the
first interface-resolved look at what the controller does.

**Day 3, 28 July. AF2-Multimer pilot and launch.**
Pilot ten jobs, measure the real per-job time, recost, then launch the full
grid. The run is resumable, so a launch late on day 3 is safe. Mohammed's 28
July target is met for question 1; question 2 is in flight.

**Day 4, 29 July. Collect and analyse.**
`modal_app/collect.py` pulls the metrics into `results/`. Join against the
charge partitions and produce interface RMSD, interface PAE and interface pTM
against charge shift. Re-run any failed jobs, which the resumable design makes
cheap.

**Day 5, 30 July. Figures and methods text.**
Final figures through `plotting.save_figure`, so each carries its manifest
hash. Write the methods paragraph and the limitations delta. Verify every
figure and table against one manifest with
`python -c "from interface_charge.provenance import verify_manifest; ..."`.

That leaves 31 July to 2 August for Mohammed's edits and the professor's
review, ahead of the 3 August deadline.

**Scheduling honesty:** the 28 July target is realistic for question 1 and
optimistic for question 2, because question 2 depends on GPU wall-clock we have
not measured yet and on data we do not have yet. If the data slips past 27
July, question 2 is the part to cut. Question 1 stands alone as a contribution,
and Limitation (i) can then be answered in text rather than in results.

## Provenance

Non-negotiable, and implemented in `src/interface_charge/provenance.py`. Every
script writes a JSON manifest next to its output containing:

- SHA256 of every input file
- git commit SHA and whether the working tree was dirty
- versions of every package that can change a number
- random seeds
- the full parameter block from `config.py`
- a UTC timestamp

The **manifest hash** covers the code commit, input digests, parameters and
seeds, and deliberately excludes the outputs. That ordering is what makes it
usable: the hash is known before the first output is written, so a table and a
figure from the same run carry the same hash, and "did Figure 2 and Table 1
come from the same run" becomes a string comparison instead of a matter of
trust. Figures carry it in their file metadata and in a small footer stamp.

`verify_manifest` re-hashes the recorded inputs and outputs and flags drift. A
manifest recorded against a dirty working tree is marked as not
publication-ready, because it cannot be reproduced from its commit.

## Deliberately unfinished

`scripts/04_join_and_analyse.py` and `scripts/05_figures.py` are documented
skeletons. They describe the intended tables and figures in their docstrings and
raise `NotImplementedError` with that description rather than pretending to
work. The reason is that the exact shape of the join depends on what the AF2
output actually contains, and writing the aggregation before seeing the data
would mean rewriting it afterwards. The analysis they will perform is specified;
the code is not written.

## Merging into the upstream repository

This project is laid out to drop into the collaborator's repository as a single
self-contained subdirectory, `rcsb_interface/`, with no file collisions and no
changes to anything already there. Verified: 45 files placed, zero collisions,
and the full test suite passes when run from inside the upstream checkout.

Nesting works without code changes because every path in this project is derived
relative to the package rather than hard-coded. `config.PROJECT_ROOT` resolves to
whatever directory contains `src/`, so it becomes `rcsb_interface/` after the
move, and the run manifests then record the *upstream* commit SHA, which is what
you want once the work lives there.

```bash
git clone https://github.com/SyedMohammedSameer/ZetaDial.git
cd ZetaDial
git checkout -b rcsb-interface-charge
mkdir rcsb_interface
# copy this repository's tracked files, preserving structure
(cd /path/to/this/repo && git ls-files) | while read -r f; do
    mkdir -p "rcsb_interface/$(dirname "$f")"
    cp "/path/to/this/repo/$f" "rcsb_interface/$f"
done
cd rcsb_interface && python -m pytest -q && cd ..
git add rcsb_interface && git commit -m "Add RCSB interface-charge arm"
```

Two things worth knowing before merging:

The upstream `.gitignore` contains a bare `figures/` pattern, which git applies
at any depth, so `rcsb_interface/figures/` is ignored there as well. That matches
this project's own intent, since figures are build products regenerated from a
manifest, but it means an upstream clone will not carry them.

Upstream script numbering has reached the thirties. This project's scripts live
in `rcsb_interface/scripts/` and keep their own `00` to `11` numbering, so they
do not collide, but if any are ever promoted to the upstream root they should be
renumbered from `40` to leave room.

## Conventions

Python 3.11, `uv` for dependencies, `ruff` for lint and format, type hints
throughout, `pathlib` rather than string paths. All parameters live in
`src/interface_charge/config.py` and can be overridden from TOML without a
source edit. Scripts are thin: argument parsing plus calls into `src/`. British
English in prose and comments.
