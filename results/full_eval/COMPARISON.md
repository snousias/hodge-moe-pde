# Checkpoint comparison — full official test set

Bold = best in column. Each row links to its own SUMMARY.md.

| Checkpoint | modes | precision | epoch | rel-L2 | R² | Volume | Surf | C_L rel-L2 | ρ_L | u_x | u_y | p | ν_t |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [v_hodgemoe_v_best_ep380](v_hodgemoe_v_best_ep380/SUMMARY.md) | v | bf16 | 380 | **0.0300** | **0.9983** | **0.00140** | **0.00164** | **0.0225** | 0.99988 | 0.0056 | 0.0259 | 0.0255 | 0.0629 |
| [vef_hodgemoe_vef_best_ep400](vef_hodgemoe_vef_best_ep400/SUMMARY.md) | vef | fp32 | 400 | 0.0329 | 0.9981 | 0.00170 | 0.00202 | 0.0313 | **0.99991** | 0.0068 | 0.0314 | 0.0364 | 0.0571 |
