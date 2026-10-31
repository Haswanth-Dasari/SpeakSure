# Evaluation guide (E1 to E4)

Four experiments give the course report its evidence. E1 is done. The others need
a small amount of human judgement, which only you can supply. Each takes about
30 to 60 minutes. Run every command from the repository root in PowerShell.

| Experiment | Question it answers | Data you create | Time |
| --- | --- | --- | --- |
| E1 Speech recognition | Which Whisper model should run live and for the report? | Done: `docs/results/e1_asr.md` | none |
| E2 Grammar checking | How often are the automatic grammar flags actually right? | Verdicts on flagged sentences | 30 min |
| E3 Content scoring | Do the model's scores agree with a human, and how stable are they? | Your own ratings of answers | 60 min |
| E4 Written review | Are the claims in the friendly review accurate? | Verdicts on review claims | 30 min |

## Before you start: collect sessions

Run 4 to 6 interviews (Standard length is fine). Use different answers: some
strong, some weak, some short. At least two should use your resume. Each finished
interview writes a session file in `data\sessions` and an analyzed file in
`data\analysis`. More answers give more reliable numbers; 25 or more rated
answers is a good target.

## E3: content scoring against a human

1. Create blind rating sheets (the model scores are hidden):

   ```powershell
   .\.venv\Scripts\python.exe scripts\make_rating_sheet.py --help
   .\.venv\Scripts\python.exe scripts\make_rating_sheet.py (Get-ChildItem data\analysis -Filter *.analyzed.json).FullName
   ```

2. Open the generated sheet and fill relevance, specificity and clarity (1 to 5,
   using the rubric printed at the top) in the CSV. Rate **before** looking at the
   report scores. If you remember the model's score, be strict with yourself.
3. Compare:

   ```powershell
   .\.venv\Scripts\python.exe scripts\eval_content.py --help
   ```

   Report: Spearman correlation, mean absolute error, and agreement of the
   Needs work / Developing / Strong bands.
4. Stability: the report already shows the three ratings per question
   ("Median of 3 ratings"). To measure spread over repeated runs, re-run the
   analysis a few times on the same session and summarise:

   ```powershell
   .\.venv\Scripts\python.exe scripts\content_stability.py --help
   ```

## E2: grammar flags

1. Export every flag from the rule-based analyzer:

   ```powershell
   .\.venv\Scripts\python.exe scripts\review_grammar_flags.py (Get-ChildItem data\analysis -Filter *.analyzed.json).FullName
   ```

2. In `data\eval\grammar_flags_review.csv`, read each sentence (the flagged words
   are in `[[double brackets]]`) and write `correct` (a real error a listener would
   notice), `incorrect` (a false alarm) or `unsure` in the `verdict` column.
3. Score it:

   ```powershell
   .\.venv\Scripts\python.exe scripts\review_grammar_flags.py --score
   ```

   Report precision per category. Expect it to be modest: spoken, run-on text is hard
   for a rule-based parser, and this is a finding, not a failure. It is the reason
   the report's main language feedback now comes from the written review (E4).

## E4: the written review

1. Export every claim the review makes (language notes, strengths, improvements):

   ```powershell
   .\.venv\Scripts\python.exe scripts\review_narrative.py (Get-ChildItem data\analysis -Filter *.analyzed.json).FullName
   ```

2. In `data\eval\narrative_review.csv`, mark each claim `correct`, `incorrect` or
   `unsure` against what you actually said and how you actually spoke.
3. Score it:

   ```powershell
   .\.venv\Scripts\python.exe scripts\review_narrative.py --score
   ```

   Compare language-note precision here with the rule-based precision from E2.
   This is the direct comparison the course report needs.

## What to write down for the report

For each experiment record: number of sessions and answers, the table printed by
the script, and two or three concrete examples (one success, one failure). Keep
the raw CSV files (without audio) so the numbers can be reproduced.
