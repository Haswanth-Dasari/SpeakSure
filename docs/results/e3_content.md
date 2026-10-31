# E3: content scoring against an independent rating

**Read this first.** The reference ratings were produced by an AI assistant
(Claude), not by a human. They were made blind: from the answer text only, using the
project's rubric and `make_rating_sheet.py`, before looking at any model score. This
is a useful second opinion but **not** the human baseline the course may expect, and
it is not fully independent of the scoring model (both are language models). Treat
the numbers as provisional until a human spot-checks a sample (see
`docs/EVALUATION.md`, "Spot-check").

Data: 3 saved sessions, 11 answer groups (10 comparable; 1 excluded because the old
strict parser left it unrated, a bug since fixed). One speaker. Raw ratings:
`docs/results/e3_ratings_ai_annotator.csv`.

## Agreement with the scoring model

| Measure | Value |
| --- | ---: |
| Answers compared (n) | 10 |
| Overall Spearman correlation | 0.916 (95% bootstrap CI 0.66 to 0.99) |
| Relevance / Specificity / Clarity Spearman | 0.610 / 0.790 / 0.472 |
| Mean absolute difference (0 to 100 scale) | 11.3 points |
| Mean bias (model minus annotator) | +11.3 points |
| Answers where the model scored higher | 10 of 10 |
| Band agreement (Needs work / Developing / Strong) | 7 of 10 |

## Stability across repeated ratings

Only the most recent session recorded three ratings per question (3 questions).

| Question | Three ratings | Spread |
| --- | --- | ---: |
| intro | 83.3, 90.3, 90.3 | 7.0 |
| motivation | 40.3, 45.8, 47.2 | 6.9 |
| challenge | 55.0, 61.3, 61.3 | 6.3 |

Mean spread 6.7 points (max 7.0, n = 3). This supports reporting the median of
several ratings and the existing note that single scores can vary by roughly this much.

## What it means

- The model **ranks** answers consistently with an independent reading (rho about
  0.92), so scores are useful for comparing answers and tracking progress.
- The model is **generous**: higher on all 10 answers, by about 11 points on
  average, and it moved two answers up a band (intro 67 vs 90; strength 50 vs 74).
  Absolute scores should not be read as a grade.
- Clarity agreed least (rho 0.47). Clarity is judged on speech-recognition text,
  and recognition errors ("full strike development") blur it.
- The sample is small (n = 10, wide interval), from one speaker and from live
  transcripts of the smaller speech models.
