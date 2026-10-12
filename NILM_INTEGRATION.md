# NILM application integration

## Architecture

The Streamlit application calls `backend.services.nilm_service` directly; no REST API or second frontend is added.
The service loads only a checkpoint declaring `model_kind=tcn`, validates the 256-minute context, appliance order,
units, and saved architecture, and reuses `nilm_research` preprocessing and inference code. The loaded model is
process-cached, placed in evaluation mode, and executed under `torch.inference_mode()` by the shared predictor.
Result caching is keyed by the checkpoint size/modification time, canonical and cleaning-data signatures, device,
household, interval, and saved quality policy. Requests are bounded and inference is chunked.

The default artifact is:

`nilm_research/outputs/executed_cpu/tcn/best.pt`

Configuration variables:

- `NILM_TCN_CHECKPOINT`: repository-relative path, or absolute path inside the repository.
- `NILM_DEVICE`: `auto` (default), `cpu`, or `cuda`. Explicit unavailable CUDA fails clearly.
- `NILM_MAX_REQUEST_MINUTES`: positive request limit, default 10080.

To replace the model, copy a trusted compatible TCN checkpoint under the repository and set
`NILM_TCN_CHECKPOINT`. Restart the application process. A changed file identity invalidates both model and result
caches. CNN baseline checkpoints, unknown appliance ordering, non-watt outputs, and incompatible contexts or encoder
settings are rejected; random weights are never substituted.

## Run and use

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

In the sidebar select Consumer Dashboard, a demo household, and My Appliances. Choose start and end date/time and
select Analyze consumption. End is exclusive. The source timestamps are timezone-naive Tunis civil time and are
explicitly interpreted as `Africa/Tunis`; JSON timestamps include that offset.

The UI exposes minute predictions, hourly observed energy, coverage, quality reasons, diagnostic residual, optional
measured references for panel homes, and request-specific CSV downloads. Long charts are display-averaged to 15
minutes, while inference and all energy arithmetic remain at one-minute resolution.

## Contract and energy semantics

Minute rows contain household/timestamp, aggregate watts, three predicted appliance powers in watts, three activity
probabilities, quality status/reason, diagnostic residual, DR flag, and nullable reference fields. References are
evaluation-only and never model inputs. The residual is `aggregate - sum(predicted target power)`; it combines
unmodeled load and estimation error and may be negative.

Hourly and period energy use `sum(valid minute-average power_w) / 60000`. Each result reports requested and valid
minutes. Hourly output distinguishes coverage of the requested portion from coverage of all 60 wall-clock minutes.
Unavailable predictions remain null: they are not zero-filled or extrapolated.

Input-context failures are reported as `insufficient_historical_context`, `missing_or_imputed_input`,
`invalid_timestamp_continuity`, `aggregate_spike`, or `frozen_reading`. Null appliance channels do not prevent
inference, so all 50 aggregate-metered households are eligible. Reference target quality is independent of input
quality.

## Limitations

This is a synthetic-data research model, not a billing, control, savings, or DR-recommendation system. AC estimates
refer to the primary AC only. A second AC can remain in background demand. Washing-machine estimates are experimental.
Activity probabilities have not been calibrated as confidence. Saved pooled metrics describe the labeled evaluation
population and do not establish performance for unlabeled/new homes. The UI reads the actual saved test artifact
rather than embedding metric values.

The application has a demo household selector, not authentication or household authorization. No live acquisition
source exists; the service intentionally accepts only historical 2025 intervals.
