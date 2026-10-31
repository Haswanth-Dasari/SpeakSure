# E4: accuracy of the written review

**Read this first.** The verdicts were given by an AI assistant (Claude), not a
human, by checking each claim against the candidate's transcripts. Every verdict and
reason is in `docs/results/e4_review_claims_verdicts_ai_annotator.csv`. A human should
spot-check a sample (see `docs/EVALUATION.md`).

Data: one interview (6 answers, resume-based, Quick length) with the review written
by `openai/gpt-oss-120b`. 9 claims were checked with `scripts/eval/review_narrative.py`.
"Correct" means the claim is true of what was actually said or how it was said.

## Precision

| Kind of claim | Correct | Incorrect | Precision |
| --- | ---: | ---: | ---: |
| Language notes | 2 | 1 | 0.67 |
| What went well | 2 | 1 | 0.67 |
| What to work on | 3 | 0 | 1.00 |
| **All claims** | **7** | **2** | **0.78** |

Language notes compare directly with the rule-based grammar flags in E2
(precision 0.05, 1 correct of 21): even on this tiny sample the written review's
language feedback was far more often right, and it was written as a plain rewrite
instead of a bare flag.

## The two wrong claims

- **A false language note.** It "corrected" *"Why am I using it in my project?"*,
  which is natural spoken English (a question the speaker asks themselves), and the
  rewrite changed the meaning.
- **An overstated strength.** It praised "a clear reason for choosing Passport.js",
  but the quote given explained why authentication matters, not why that library was
  chosen. The actual reason was weak ("I have learned that one only").

## What the check also found (and what was changed)

- One evidence quote was cut off mid-word ("I really hope and believe th") and did
  not support its point. The quote check matched characters, so a truncated quote
  passed. It now requires whole-word matches (fixed, with a test).
- The larger report-pass speech model heard the candidate's name wrongly
  ("Ashwan") where the live model, using the candidate's own hints, was right
  ("Haswanth"). The report pass now keeps the live transcript for an answer when it
  contains more of the candidate's names and terms (fixed, with tests).

## Limits

One session, one speaker, nine claims, an AI annotator, and a review written by the
same family of model that scores the answers. This shows the review is usually right,
not that its precision is 0.78 in general.
