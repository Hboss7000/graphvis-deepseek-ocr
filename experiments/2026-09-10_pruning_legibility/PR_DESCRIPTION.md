# Make degree pruning optional and add measured legibility evaluation

The local soft degree cap was constraining retrieved graph structure. Disable
it by default (`--max-degree 0`), retain the positive-cap path, raise the default
edge budget to 60, and record actual pruning limits and truncation counts in
metadata. Stable tie-breaks across pruning make the result deterministic under
`PYTHONHASHSEED=0` and `1`; this upstream determinism work was explicitly added
to scope by the user.

Add opt-in font, spacing, orientation and canvas controls without changing
rendering defaults. Add split connectivity statistics, an audit based on a
measured Graphviz reference glyph, and a 12-configuration rendering sweep whose
graph content and Stage 1/2 records remain identical across layouts.

Removing the cap changes graph structure and therefore invalidates reuse of
Stage 1 ground truth: degrees, edge counts, node/neighbour lists, triples and
shortest paths must be regenerated along with the affected images. Full splits
intended for the new regime must be regenerated in distinct directories;
completed zero-shot scores under the old regime are not directly comparable to
new-regime scores. Existing saved splits remain intact, and the generator now
refuses output-file collisions. Stable tie-breaking can also change historical
ties under `--max-degree 5 --max-edges 30`, so old flags alone do not promise
byte-for-byte reproduction of the former nondeterministic implementation.

Validation: 36 tests pass. Default rendering matches the pre-change PNG bytes on
an unchanged graph. The real 20-question controlled comparison found no change
in the correct-option zero-degree fraction from cap removal (30% to 30%). The
20-graph rendering sweep reduced estimated below-10-pixel labels from 17/20 to
5/20 with TB / 30-point nodes / ranksep 0.4; this is a candidate for the Gemma
recall gate, not proof of model readability. Full split regeneration and GPU
preflight have not been performed.
