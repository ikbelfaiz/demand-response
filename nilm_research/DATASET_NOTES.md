# Dataset verification and quality policy

The pipeline reads, but never modifies, the repository's canonical v3 household files:

- `data/households_1min_C001_C025.parquet`
- `data/households_1min_C026_C050.parquet`

Inspection found 26,280,000 rows: 50 households with 525,600 ordered, unique one-minute readings from
2025-01-01 00:00 through 2025-12-31 23:59. The required columns are `timestamp`, `client_id`,
`aggregate_power_w`, `ac_power_w`, `water_heater_power_w`, and `washing_machine_power_w`.

The ten labeled panel homes are detected independently from non-null appliance channels and
`households_info.has_submeter`; the two sources must agree. The remaining 40 homes have structurally null appliance
channels. Null does not mean off or zero.

Generator interpretation

- Aggregate is gross whole-home demand. Neighborhood PV is separate and is not netted from it.
- The AC channel is the primary AC. A simulated second bedroom AC, where present, remains in aggregate residual.
- Aggregate includes unlabeled base, refrigerator, lights, electronics, small appliances, secondary AC, and noise.
- The generator deliberately inserts missing runs, frozen readings, and aggregate spikes.
- DR-compliant households have simulator-modified appliance behavior. DR response/compliance is never an input.

Provenance and quality policy

Canonical values were completed with two-sided timestamp interpolation. Exact row-level provenance is recovered by
joining `data/cleaned_v3/*_cleaned.parquet` on `(client_id, timestamp)` and reading its `*_was_imputed` flags. The
cleaned values are not used as measurements. If these files or flags are unavailable, preprocessing fails rather than
claiming leakage-safe evaluation.

- A context is excluded if any aggregate point was imputed, is non-finite, exceeds the training-only aggregate
  quantile cap, belongs to an exact-value frozen run of at least 20 minutes, or follows a timestamp gap.
- A target is masked independently if it was imputed, missing, or exceeds its training-derived cap bounded by the
  inspected semantic caps: 2,000 W primary AC, 2,500 W water heater, and 2,500 W washing machine.
- Raw files are unchanged. Exclusion counts are saved with each run.
- Aggregate and difference scaling, target scales, activity prevalence/class weights, caps, and consistency tolerance
  are fitted only on January-June data from non-held-out households.
- Input differences are calculated inside each partition and household only. The first difference of each loaded
  partition is zero; windows never cross that boundary.

Activity is initially a pointwise derived label, not source ground truth: primary AC >=100 W, water heater >=100 W,
and washing machine >=10 W. These thresholds reflect inspected synthetic appliance semantics and are recorded in the
fit artifact. No hysteresis is applied.

Known limitations

- Synthetic, one region, one year, and only ten deliberately selected panel homes, all owning the three targets.
- No voltage, current, reactive power, phase, or high-frequency signatures.
- Labels are noisy synthetic submeters and do not exhaust aggregate demand.
- Exact frozen-value detection can flag real steady readings; counts must be reviewed per run.
- Household-held-out and chronological shifts are reported separately, but with ten panel homes uncertainty is high.

