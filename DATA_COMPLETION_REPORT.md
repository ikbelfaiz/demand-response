# Canonical synthetic v3 completion

The three pre-completion sensor files were copied to `data/original_v3_backup/` and verified with SHA-256 checksums before replacement. `scripts/complete_v3.py` builds and validates temporary outputs before atomic replacement.

| Channel | Corrected values | Boundary fallback |
|---|---:|---:|
| Temperature | 2,280 | 0 |
| Feeder demand | 214 | 0 |
| STEG supply | 0 | 0 |
| PV supply | 315 | 0 |
| Household aggregate | 106,025 | 0 |
| AC submeter | 13,252 | 0 |
| Water-heater submeter | 11,675 | 0 |
| Washing-machine submeter | 16,230 | 0 |
| DR peak labels | 528 reconstructed | n/a |

Internal numerical gaps used timestamp-based linear interpolation. The boundary policy is the closest valid observation, but no eligible series required it. Missing peak labels were reconstructed using the generator definition: centered 15-minute mean feeder demand greater than available STEG plus PV. Existing labels were unchanged.

All 50 aggregate channels and all installed appliance submeters are complete and non-negative. Appliance columns remain null for the 40 homes without submeters. Detailed backup/completion hashes and counts are stored beside the backups.
