# Deep research prompt: outstanding questions on the RCSB-complex arm

Paste the section below into a deep research tool. It is written to be
self-contained, so it repeats context that is obvious to us but not to it.

Everything in "What we have already established" is measured, not assumed. The
research is not needed to confirm any of it, and a tool that spends its budget
re-deriving those numbers has wasted the run.

---

## Context

I am co-authoring a short conference paper (PSB 2027, 8 pages) on controllable
electrostatic protein design. The method is a single signed scalar `beta` added
to the ProteinMPNN output logits at inference time: `logit[K] += beta`,
`logit[R] += beta`, `logit[D] -= beta`, `logit[E] -= beta`, sampled at
temperature 0.1. No retraining. This gives a calibrated dial on the net charge
of a redesigned chain. The claimed application is therapeutic binder
developability, where net charge drives solubility, viscosity and
polyspecificity.

My part of the paper uses 55 RCSB two-chain complexes. One chain is redesigned
across `beta` in {-3, -1.5, 0, +1.5, +3}, the partner is held at its native
sequence, and the native backbone is fixed. I answer two questions: where on the
protein the imposed charge actually lands, and whether the interface survives.

## What we have already established, with numbers

Do not re-derive these. Treat them as given, and tell me where they are
vulnerable.

1. **Charge is buffered at the interface.** Partitioning each chain into
   interface / non-interface surface / buried core and regressing each region's
   charge on the whole-chain charge shift gives a buffering ratio (slope divided
   by that region's share of residues; 1.0 means the region absorbs its fair
   share). Interface 1.190, bulk surface 1.444, paired difference
   -0.254 [-0.302, -0.206], negative in 50 of 55 complexes.
2. **This is not just a burial effect.** ProteinMPNN's sequence recovery is
   strongly graded by burial and interface residues are intermediate between
   core and bulk surface, so the effect above is exactly what burial alone would
   predict. A permutation null resampling interface-sized sets from surface
   residues matched on relative SASA (10,000 permutations) gives observed minus
   null of -0.2328 [-0.2953, -0.1714], with 50 of 51 chains individually
   significant. Mean relative SASA is 0.475 at the interface against 0.471 in
   the matched null, so the match is good.
3. **Electrostatic complementarity with the partner is monotonic and
   asymmetric** across beta = -3 to +3: -6.0, -1.0, +4.0. Computed without any
   Poisson-Boltzmann solver.
4. **The salt-bridge gain is compositional, not organisational.** Raw
   cross-interface salt-bridge count rises by +3.96 [+1.82, +6.22] and charged
   residue count by +5.24 [+2.47, +8.13], but normalised per charged-residue
   opportunity the change is +0.0007 [-0.0009, +0.0026], which includes zero.
   Adding charge adds bridges at the rate you would get by chance; it does not
   place them better.
5. **The foldability collapse tracks the Uversky boundary.** Placing all 275
   designs on the charge-hydropathy diagram (Uversky, Gillespie and Fink 2000,
   boundary `mean net charge = 2.785 * mean scaled hydropathy - 1.151`) and
   joining to observed self-consistency RMSD: designs inside the boundary fold
   at 76.6 percent, outside at 43.0 percent. Pooled Spearman of distance from
   boundary against RMSD is +0.628, but that is inflated because distance rises
   with |beta| almost by construction. Within a fixed beta, where only the
   sequences differ, the correlation is +0.402 (p=0.003) at beta=-3, +0.236 at
   -1.5, +0.091 at 0, +0.300 at +1.5, +0.434 (p=0.002) at +3. So the diagram
   carries real sequence-level information exactly where the collapse happens
   and none where there is no charge stress.
6. **There is a sign asymmetry the Uversky diagram cannot explain.** Fold rates
   are 9.1 percent at beta=-3 against 18.2 percent at +3, and 63.6 percent at
   -1.5 against 74.5 percent at +1.5. Negative bias is consistently worse. But
   the diagram uses *absolute* mean net charge, so it places beta=-3 and beta=+3
   at nearly the same distance from the boundary (1.107 against 1.048). It
   cannot account for a roughly two-fold difference in fold rate.

## What I need

Prioritise 6, then 7, then the rest. Depth on one well-sourced answer beats
breadth across all of them.

1. **The sign asymmetry (highest priority).** Why would negative net-charge bias
   be worse for foldability and for structure prediction than positive bias of
   equal magnitude? Separate three candidate explanations and tell me which the
   literature supports: (a) a real biophysical asymmetry, for example the
   different desolvation penalties of Asp/Glu against Lys/Arg, or backbone
   helix-dipole and pKa effects; (b) an amino-acid composition artefact, since
   ProteinMPNN under a negative bias must draw from only two acidic residues
   while a positive bias draws from Lys and Arg which differ in side-chain
   entropy and hydrogen-bonding geometry; (c) a *training-set* artefact, since
   the PDB is enriched in certain charge regimes and ESMFold/AlphaFold inherit
   that. Point me at papers that distinguish these. The supercharging
   literature (Lawrence, Liu, Rees and others) is the obvious place to start but
   I want work that isolates the direction of the asymmetry rather than merely
   noting tolerance.

2. **Are we scooped, and by how much?** The novelty claim is "a signed,
   calibrated dial to an arbitrary net-charge target at inference time, without
   retraining." Find the closest prior art and say plainly how close it gets.
   Specifically assess **ProteinGuide** (arXiv:2505.04823), which does
   principled on-the-fly property guidance for ProteinMPNN among other models
   without retraining, and which I think is the strongest challenge to the
   novelty claim. Also look for: classifier guidance and classifier-free
   guidance applied to protein sequence models; any inference-time logit-bias
   trick for composition control in ProteinMPNN, ESM-IF, LigandMPNN or PiFold;
   and property-driven inverse folding with preference alignment. For each, say
   whether it (i) targets net charge specifically, (ii) is calibrated to a
   numeric target rather than a direction, and (iii) requires training. I would
   rather find this now than have a reviewer find it.

3. **Is interface buffering already known?** Our result 1 and 2 say the
   interface absorbs less imposed charge than bulk surface even after matching
   on burial. Has anyone reported that inverse-folding models are more
   constrained at interfaces than at equally exposed non-interface surface? Look
   at ProteinMPNN follow-ups, interface-conditioned design, and any analysis of
   position-wise entropy or recovery split by interface membership. If it is
   known I need to cite it; if it is not, that is our contribution and I need to
   be confident.

4. **Salt-bridge normalisation.** Our result 4 says the raw count rises but the
   per-opportunity rate does not. Is the normalised statistic (bridges divided
   by cationic times anionic residue pairs) the accepted one, and is there a
   better-established denominator? I want to be sure a reviewer will not say we
   have normalised away a real effect. Cite work on salt-bridge counting
   conventions and on whether interface salt bridges contribute net favourably
   to binding once desolvation is paid.

5. **Does the Uversky boundary transfer to designed sequences?** It was fitted
   to natural proteins in 2000, long before de novo design. Is there published
   work applying it to designed or generated sequences, and is it known to
   over-call disorder for them? Our data shows it does: at beta = +/-1.5 it
   calls 100 percent of designs disorder-prone while 64 to 75 percent of them
   fold. I want a citation for that limitation, or confirmation that nobody has
   checked. Also tell me whether a later refinement of the boundary (Uversky's
   own later work, or the Das and Pappu kappa parameter for charge patterning)
   would be the better instrument, given that our dial changes charge *fraction*
   while leaving *patterning* to the model.

6. **Charge patterning as the missing variable.** Following from 5: our dial
   sets how much charge, not where. Das and Pappu's kappa and the sequence
   charge decoration parameters describe how charge is distributed along a
   sequence, and predict very different conformational behaviour at the same net
   charge. Is there work applying these to designed binders or to inverse
   folding output? If patterning explains variance that net charge alone does
   not, that is a strong follow-up and possibly a limitation we should state.

7. **AlphaFold2-Multimer on charge-perturbed sequences.** We are about to refold
   these complexes with AF2-Multimer, giving the designed chain no MSA (it has
   no evolutionary history) while the native partner keeps its own. Is there
   published evidence on how AF2 behaves on single-sequence input for a designed
   chain paired with an MSA-backed partner, and specifically whether its
   confidence metrics are biased by sequence net charge independently of whether
   the structure is right? If AF2 is systematically less confident about highly
   charged sequences regardless of correctness, our interface pTM trend against
   beta is confounded and I need to know before I report it. Also confirm
   whether ipSAE (Dunbrack 2025) is now the accepted interface confidence metric
   for cross-complex comparison, and whether anyone has characterised its
   behaviour under charge perturbation.

## Output

For each numbered question: a direct answer first, then the evidence, then how
confident you are and what would change your mind. Full citations with DOIs or
arXiv identifiers. Flag anything you could not verify from a primary source
rather than filling the gap, and say explicitly where the literature is silent,
because a genuine gap is a result for us and a fabricated citation is a
retraction.
