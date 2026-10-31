# E2: precision of the rule-based grammar flags

**Read this first.** The verdicts were given by an AI assistant (Claude) reading each
flagged sentence, not by a human. A human should spot-check a sample (see
`docs/EVALUATION.md`). Every verdict and reason is in
`docs/results/e2_grammar_verdicts_ai_annotator.csv` so it can be audited.

Data: all 22 flags saved in 3 sessions (one speaker, 11 answer groups). Scored with
`scripts/review_grammar_flags.py --score`. "Correct" means a real error a listener
would notice.

## Precision

| Category | Correct | Incorrect | Unsure | Precision |
| --- | ---: | ---: | ---: | ---: |
| article | 1 | 9 | 1 | 0.10 |
| subject_verb_agreement | 0 | 11 | 0 | 0.00 |
| verb_tense | 0 | 0 | 0 | no flags |
| **all flags** | **1** | **20** | **1** | **0.05 (1/21)** |

## Why the flags were wrong

- **Idioms and fixed phrases** read as missing articles: "from beginning to end",
  "for example", "of course".
- **Mass or compound nouns**: "handle pressure", "system design", "complex architecture".
- **Parser confusion on spoken text**: "I plan things" read as a noun; "will be used"
  and "would be" read as agreement errors; two flags even pointed at the first
  letters of "believe".
- **Recognition errors blamed on the speaker**: "speak show" is "SpeakSure" mis-heard.
- Two false alarms ("pressure", "for example") are already suppressed by the later
  mass-noun fix. Without them precision would still be about 0.05 (1/19).

## Recall (informal)

Reading the same transcripts, at least five clear errors were never flagged, for
example "will be appear", "would of course would be", "the major skill is that how I
handle", "delivered at the best", and "the user can able to practice". Recall is
therefore low (no more than about 1 in 6 on this sample). This is an informal count
from three sessions, not a measured recall.

## Decision

A parser built for written text does not transfer to spoken, run-on, recognition-noisy
transcripts. The report therefore **hides the rule-based grammar section by default**
(`SHOW_RULE_GRAMMAR=1` restores it) and relies on the written review's verified
language notes instead. Their accuracy is measured in E4.
