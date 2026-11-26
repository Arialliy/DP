M1 candidate-based diagnostic correction, staged only

No main code/ files were changed. Original and fixed copies are in original/code/
and fixed/code/. candidate_m1.patch modifies only code/miss_recovery.py and
code/test_miss_recovery.py. All 11 synthetic tests passed in the existing CPU
Docker image; tests.log and validation.json record that run. No real dataset,
network, training or benchmark run was performed for this fix.

The original M1 classified every annotated centre, regardless of whether its
single-frame candidate generator proposed a point there. It also let -inf scores
pass a -inf threshold and did not handle clutter-score ties conservatively.
The fixed diagnostic measures actual contrast-peak candidates:

- LORO regression trains on actual positive/clutter candidates from other
  recordings. Ambiguous points are excluded from fitting; all real points are
  scored. Held-out labels and sibling-clip features cannot affect that recording's
  fitted score; the regression explicitly tests this exclusion.
- Finite scored candidates with a teacher are matched one-to-one to the nearest
  eligible target, in deterministic score order. No GT-centre feature is used as
  a detection proposal, and a single point cannot recover two adjacent boxes.
- Geometric recovery ceilings use maximum bipartite matching among each group's
  missed targets, before scores/FA limits. These per-group ceilings need not sum
  because each group is a separate optimistic matching problem.
- Clutter thresholds exclude boundary ties as needed. The observed number of
  labelled clutter points and per-frame rate are recorded and checked against the
  requested pooled/per-clip budget. This is clutter beyond r_neg, not official
  pixel Fa or a count of every unmatched candidate.
- Baseline per-box hits retain the official first-row/column pixel eligibility.
  They remain a box metric, not the official merged-component metric.
- Frame denominators are actual clip lengths, including blank frames.

Output/schema changes:
  targets.csv: replaces oracle-location feature columns with candidate_ids,
  candidate_count, matched_candidate and the assigned candidate score. Existing
  target identity, baseline hit flags and group metadata are retained.
  candidates.csv: new audit table with actual coordinates, six features, binary
  positive/clutter label or ambiguous=-1, eligibility, teacher count and LORO score.
  windows.csv: adds owned_first/owned_last to clarify the subset used for miss counts.
  summary.json: existing results keys retained; adds candidate ceilings,
  measured_clutter, candidate_labels, metric/fitting descriptions and decision.

The decision is fixed to pooled 1 clutter candidate/frame, baseline threshold
1e-5, and frames0..59. It requires early recovery>=0.30 and all recovery>=0.20,
nonzero missed-target denominators and measured clutter<=1/frame. No cutoff was
changed. Existing hits are retained through logical OR; appearance hits_covered
is reported separately. M1 still uses evaluated test labels for diagnostic fitting
on other recordings and for clutter-threshold calibration; it is not a held-out
model result. A failed diagnostic only rejects this proposal/feature recipe;
it does not prove that no learned single-frame detector could help.

For applying after the original check/diagnostic finishes, from project root:
  patch -p1 --dry-run < runtime/m1_fix/candidate_m1.patch
  patch -p1 < runtime/m1_fix/candidate_m1.patch
Do not apply while the original full check is still running.
