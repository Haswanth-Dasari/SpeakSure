# E1 ASR benchmark results

Bootstrap: 2,000 clip resamples; seed=1234. Excluded clips: none.

## Model comparison

| Model | Clips | Ref words | Raw WER (95% CI) | Extended WER (95% CI) | RTF | Mean word probability | Load + first run (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 31 | 371 | 0.205 [0.137, 0.284] | 0.203 [0.134, 0.283] | 0.022 | 0.778 | 1.490 |
| small | 31 | 371 | 0.156 [0.093, 0.240] | 0.149 [0.087, 0.230] | 0.039 | 0.858 | 1.135 |
| medium | 31 | 371 | 0.094 [0.045, 0.164] | 0.095 [0.045, 0.165] | 0.079 | 0.860 | 2.398 |
| large-v3 | 31 | 371 | 0.086 [0.041, 0.148] | 0.076 [0.033, 0.132] | 0.140 | 0.877 | 4.820 |

## Per-category

| Group | Model | Clips | Ref words | Raw WER | Extended WER |
|---|---|---:|---:|---:|---:|
| unspecified | base | 11 | 87 | 0.264 [0.104, 0.425] | 0.264 [0.104, 0.425] |
| plain | base | 8 | 103 | 0.282 [0.170, 0.394] | 0.282 [0.170, 0.394] |
| numbers | base | 5 | 57 | 0.070 [0.021, 0.106] | 0.054 [0.020, 0.082] |
| names | base | 5 | 64 | 0.281 [0.191, 0.462] | 0.281 [0.191, 0.462] |
| natural | base | 2 | 60 | 0.033 [0.000, 0.067] | 0.033 [0.000, 0.067] |
| unspecified | small | 11 | 87 | 0.230 [0.067, 0.433] | 0.230 [0.067, 0.433] |
| plain | small | 8 | 103 | 0.155 [0.088, 0.229] | 0.155 [0.088, 0.229] |
| numbers | small | 5 | 57 | 0.123 [0.000, 0.261] | 0.071 [0.000, 0.182] |
| names | small | 5 | 64 | 0.219 [0.070, 0.491] | 0.219 [0.070, 0.491] |
| natural | small | 2 | 60 | 0.017 [0.000, 0.033] | 0.017 [0.000, 0.033] |
| unspecified | medium | 11 | 87 | 0.218 [0.069, 0.395] | 0.218 [0.069, 0.395] |
| plain | medium | 8 | 103 | 0.078 [0.030, 0.140] | 0.078 [0.030, 0.140] |
| numbers | medium | 5 | 57 | 0.018 [0.000, 0.050] | 0.018 [0.000, 0.050] |
| names | medium | 5 | 64 | 0.109 [0.000, 0.327] | 0.109 [0.000, 0.327] |
| natural | medium | 2 | 60 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| unspecified | large-v3 | 11 | 87 | 0.172 [0.044, 0.333] | 0.172 [0.044, 0.333] |
| plain | large-v3 | 8 | 103 | 0.058 [0.016, 0.121] | 0.058 [0.016, 0.121] |
| numbers | large-v3 | 5 | 57 | 0.070 [0.000, 0.135] | 0.000 [0.000, 0.000] |
| names | large-v3 | 5 | 64 | 0.109 [0.015, 0.269] | 0.109 [0.015, 0.269] |
| natural | large-v3 | 2 | 60 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

## Per-device

| Group | Model | Clips | Ref words | Raw WER | Extended WER |
|---|---|---:|---:|---:|---:|
| Sound Mapper (default) | base | 5 | 40 | 0.175 [0.000, 0.436] | 0.175 [0.000, 0.436] |
| Airdopes hands-free | base | 6 | 47 | 0.340 [0.130, 0.531] | 0.340 [0.130, 0.531] |
| Microphone Array (Realtek(R) Audio) | base | 20 | 284 | 0.187 [0.114, 0.265] | 0.184 [0.110, 0.263] |
| Sound Mapper (default) | small | 5 | 40 | 0.200 [0.024, 0.447] | 0.200 [0.024, 0.447] |
| Airdopes hands-free | small | 6 | 47 | 0.255 [0.040, 0.583] | 0.255 [0.040, 0.583] |
| Microphone Array (Realtek(R) Audio) | small | 20 | 284 | 0.134 [0.070, 0.211] | 0.124 [0.065, 0.197] |
| Sound Mapper (default) | medium | 5 | 40 | 0.225 [0.023, 0.526] | 0.225 [0.023, 0.526] |
| Airdopes hands-free | medium | 6 | 47 | 0.213 [0.040, 0.458] | 0.213 [0.040, 0.458] |
| Microphone Array (Realtek(R) Audio) | medium | 20 | 284 | 0.056 [0.021, 0.108] | 0.057 [0.021, 0.108] |
| Sound Mapper (default) | large-v3 | 5 | 40 | 0.125 [0.000, 0.317] | 0.125 [0.000, 0.317] |
| Airdopes hands-free | large-v3 | 6 | 47 | 0.213 [0.040, 0.458] | 0.213 [0.040, 0.458] |
| Microphone Array (Realtek(R) Audio) | large-v3 | 20 | 284 | 0.060 [0.026, 0.103] | 0.046 [0.015, 0.087] |

## Paired comparisons

| First model vs second | Mean WER difference (first − second) | 95% interval | Fraction first better |
|---|---:|---:|---:|
| medium vs large-v3 | 0.020 | [-0.012, 0.055] | 0.107 |
| small vs medium | 0.053 | [0.017, 0.092] | 0.002 |

## First-word analysis

| Model | First word wrong/missing | Other ref words wrong (substitution/deletion) |
|---|---:|---:|
| base | 7/31 (0.226) | 146/340 (0.429) |
| small | 7/31 (0.226) | 118/340 (0.347) |
| medium | 5/31 (0.161) | 67/340 (0.197) |
| large-v3 | 3/31 (0.097) | 70/340 (0.206) |

## Reading guide

Raw WER counts every token mismatch after lowercase and punctuation cleanup.
Extended WER also equates listed US/UK spellings, clock forms, and digit ordinals.
Intervals resample whole clips, so each clip contributes its complete reference and hypothesis.
A negative paired difference means the first listed model has lower WER.
First-word rates flag a first-position mismatch; other-word rates count remaining reference positions.
