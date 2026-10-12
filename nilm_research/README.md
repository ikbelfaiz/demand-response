# Standalone causal NILM research pipeline

This remains a standalone research pipeline with its own configuration, commands, tests, and outputs. The Demand
Response application now imports its checkpoint/model/preprocessing modules for inference, so there is one source of
truth; research training and CLI workflows remain independently runnable and never modify source datasets.

## What it implements

- Native one-minute, aggregate-only windows of shape `[batch, 2, 256]`: robustly normalized aggregate power and a
  causal first difference.
- One shared 64-channel causal TCN encoder and independent primary-AC, water-heater, and washing-machine heads.
- Six two-convolution residual blocks with dilations 1, 2, 4, 8, 16, and 32, kernel size 3, left-only padding,
  weight normalization, GELU, dropout 0.1, and receptive field 253.
- Independent Softplus power and sigmoid activity outputs. Power is never multiplied by activity.
- A small causal CNN baseline using the identical data policy and appliance heads.
- Masked Huber power loss, per-appliance class-weighted BCE, and a noise-tolerant over-aggregate consistency penalty.
- Chronological and household-held-out evaluation, full-minute chunked inference, metrics, CSV exports, and plots.

See `DATASET_NOTES.md` for verified dataset semantics, exclusions, provenance, and limitations.

## Setup

From the repository root, use a dedicated environment; do not install through the repository's shared requirements:

```bash
python -m venv nilm_research/.venv
nilm_research/.venv/bin/python -m pip install -r nilm_research/requirements.txt
```

All commands below are run from the repository root. The scripts bootstrap `nilm_research/src` themselves.

## Inspect and validate preprocessing

```bash
python nilm_research/scripts/inspect_data.py \
  --config nilm_research/configs/default.json \
  --output nilm_research/outputs/preprocessing_report.json
```

This detects the panel homes, verifies timelines and schema, fits transforms on eligible training households only, and
reports split-specific exclusion counts. It does not write to `data/`.

## Train, resume, and evaluate

```bash
python nilm_research/scripts/train.py --config nilm_research/configs/default.json --model tcn
python nilm_research/scripts/train.py --config nilm_research/configs/default.json --model cnn_baseline

python nilm_research/scripts/train.py --config nilm_research/configs/default.json \
  --model tcn --resume nilm_research/outputs/default/tcn/last.pt

python nilm_research/scripts/evaluate.py --checkpoint nilm_research/outputs/default/tcn/best.pt
python nilm_research/scripts/evaluate.py --checkpoint nilm_research/outputs/default/cnn_baseline/best.pt
```

`default.json` is the intended longer research configuration. `executed_cpu.json` is the bounded configuration used
for the checked execution on the available CPU-only host. Training uses weighted replacement sampling; evaluation
always uses the natural, non-oversampled distribution of every quality-valid minute.

The primary split is fixed before windows are built:

- Training: 2025-01-01 through 2025-06-30.
- Validation/model selection: July 2025.
- Test: August through December 2025.

`C023` and `C044` are held out from training, transform fitting, threshold/class-weight fitting, and model selection.
Metrics mark them as `held_out_household`; the other eight are `same_household`. Change the list in a new config for a
different grouped experiment.

## Standalone inference

```bash
python nilm_research/scripts/infer.py \
  --checkpoint nilm_research/outputs/default/tcn/best.pt \
  --household C005 --start "2025-08-04 00:00:00" --end "2025-08-05 00:00:00" \
  --output nilm_research/outputs/default/tcn/example_predictions.csv \
  --plot nilm_research/outputs/default/tcn/example_predictions.png
```

The output contains power in watts, uncalibrated activity probabilities, the diagnostic residual
`aggregate - sum(predicted targets)`, reference labels where valid, and a row-level quality status. Insufficient or
contaminated contexts remain blank. The companion hourly file uses exactly:

`energy_kwh = sum(valid minute-average power_w) / 60000`

It reports valid-minute coverage and never fills missing minutes or extrapolates partial hours.

## Tests

```bash
pytest -q nilm_research/tests
```

Tests use small slices of the existing dataset. They cover model shape/nonnegative power, strict causality, split and
household isolation, training-only fitting, imputation/contamination masks, energy/coverage semantics, checkpoint
reload consistency, gradients and finite loss, and standalone insufficient-history behavior.

## Outputs

Each model directory contains `best.pt`, `last.pt`, full resolved configuration in the checkpoint, transforms,
fit/exclusion provenance, training history, validation/test metrics, evaluation summaries, and plots. Metric CSVs
include per-appliance and per-household power MAE, hourly energy MAE, relative total-energy error, precision, recall,
F1, observation counts, and hourly coverage. Rows are stratified by DR status, season, target activity state, and
same-versus-held-out household population. Zero denominators are emitted as blank/`null`, never coerced to zero.

Activity probabilities are not called confidence: no calibration evaluation is implemented. AC output is always
described as primary-AC consumption.
