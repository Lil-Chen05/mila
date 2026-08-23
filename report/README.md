# Final COMP 400 report

This directory contains the source and compiled PDF for **Beyond Endpoint
Confidence: Characterizing Uncertainty Across Reasoning Trajectories**.

## Build

The source is self-contained and uses the bundled ACL style. From this
directory, run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The command writes `main.pdf`. A complete build requires pdfLaTeX, BibTeX,
and latexmk. Generated auxiliary files are ignored by Git.

## Contents

- `main.tex` — authoritative paper source.
- `main.pdf` — reviewed final report.
- `references.bib` — bibliography database.
- `figures/` — publication figures in PDF and PNG form.
- `acl.sty` and `acl_natbib.bst` — local style dependencies required for a
  reproducible build.

The numerical results behind the figures are tracked under
`../analysis/final-r5000/`. The raw production shards remain external and are
not committed to this repository.
