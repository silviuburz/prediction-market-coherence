# Prediction-market coherence

Code and replication materials for **A Payoff-Compiled Audit of Dynamic
Coherence in Linked Prediction-Market Order Books**, by **Silviu Burz** and
**Costin-Gabriel Chiru**, IEEE Access, 2026.

**Project website:** [silviuburz.github.io/prediction-market-coherence](https://silviuburz.github.io/prediction-market-coherence/)

**Read the paper:** [10.1109/ACCESS.2026.3746471](https://doi.org/10.1109/ACCESS.2026.3746471)

Citation details are in [CITATION.cff](./CITATION.cff).

The supplied analysis results let readers recompute the reported numbers and
redraw the figures offline. The **23.70 GB raw Polymarket/ESPN corpus is not
included**. The step from raw observations to these result files cannot be
repeated from this repository.

## Setup and checks

Use Python 3.12 or later in a virtual environment. Install the
[pinned requirements](./requirements.txt), then run from the repository root.
These examples use Windows PowerShell paths:

```powershell
python -m pip install -r requirements.txt
python tools\audit_provenance.py --package .
python tools\check_numeric_claims.py --package .
python code\test_grammar.py
python code\test_pipeline_fixture.py
```

Expected results: 48 of 48 producer files resolved, 354 numeric and result-file
checks passed, 67 payoff-rule assertions passed, and a synthetic-test change
of +1.667 cents. The synthetic test uses generated data with a known answer.
The numeric audit checks fixed targets against the result files; it does not
download or check the paper's wording.

## Figures

Run the appropriate script from the [figures](./figures/) directory.
Each writes a PDF and PNG; generated images are not stored in this repository.

| Paper figure | Script |
|---|---|
| 1 | [fig1_instrument.py](./figures/fig1_instrument.py) |
| 2 | [fig2_event_time.py](./figures/fig2_event_time.py) |
| 3 | [fig3_threshold.py](./figures/fig3_threshold.py) |
| 4 | [fig5_controls.py](./figures/fig5_controls.py) |
| 5 | [fig5_non_goal_comparison.py](./figures/fig5_non_goal_comparison.py) |
| 6 | [fig4_spec_curve.py](./figures/fig4_spec_curve.py) |

## Data and records

- [code](./code/) and [frozen_producers](./frozen_producers/) contain the
  analysis code, its recorded versions, and tests.
- [figures/data](./figures/data/) contains the result files, fixed sample
  split, 4,573 resolution descriptions, 3,252-shape unsupported-contract
  inventory and Wikipedia-derived validation table.
- [PROVENANCE.json](./PROVENANCE.json) records code identities, settings and
  outputs. [PROVENANCE_AUDIT.json](./PROVENANCE_AUDIT.json) accounts for the
  supplied files. SHA-256 hashes identify exact file contents.
- [CLAIM_MAP.json](./CLAIM_MAP.json) maps the numeric checks to their inputs.
  [SOURCE_FILES.sha256](./SOURCE_FILES.sha256) lists file checksums.

[RERUN_VERIFICATION.json](./RERUN_VERIFICATION.json) records a separate
analysis-host rerun. Its `differs` verdict is retained: two cancellation-state
`rand_p` values changed from zero to 1/5001 after the comparison-test correction.
No cancellation-state p-value is reported in the paper.
The recorded host commands are an audit record, not a runnable raw-data workflow.

## Licenses

[LICENSE-code](./LICENSE-code) covers the authors' code.
[LICENSE-data](./LICENSE-data) covers original data within its stated scope and
records the separate Wikipedia attribution and source-text terms.
