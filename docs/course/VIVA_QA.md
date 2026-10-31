# Viva preparation: questions you should be able to answer

Answer in your own words. Each line gives the point to make and where to look.

## The big picture
1. **What problem does SpeakSure solve?** Spoken interview practice with listening,
   reacting and feedback, without a human partner. (README, report section 1)
2. **Why did you narrow the project?** The original plan (meetings, casual scenes,
   emotion models) was too large and the emotion models transfer poorly from acted
   speech. A focused core could be measured. (Journey document, sections 3 to 4)
3. **Where is NLP in this system?** Speech recognition, language-model interviewing,
   grammar analysis, rubric-based content scoring, information extraction from resumes
   and generation of the written review.

## Speech
4. **Why Whisper, and why local?** Cost, privacy, and the 6 GB GPU. E1 compared four
   sizes. (`../results/e1_asr.md`)
5. **What is word error rate?** (Substitutions + deletions + insertions) divided by
   reference words, after normalising case and punctuation.
6. **Why bootstrap the clips, not the words?** Words in one clip are not independent;
   resampling whole clips gives honest intervals.
7. **Why two transcription passes?** The live loop needs speed; the report does not.
   Medium live, large-v3 for the report. Rejection rules protect against a bad pass.
8. **Why are names hard, and what did you do?** Rare strings are not in the
   language prior. Hotwords from the typed name and the resume help, but only
   distinctive terms are used because unspoken hotwords can be echoed.
9. **How does the system decide an answer has ended?** Voice-activity detection with
   noise calibration, sustained-speech start, and a 2.5 s silence. 1.5 s cut people off.
10. **What did the thinking-time bug teach you?** The first number was a constant
    0.35 s: it measured microphone start-up. Suspiciously uniform data is a warning.

## The interviewer
11. **How does it choose follow-ups?** A language model proposes `follow_up` or
    `next_question` as JSON; code enforces limits and rejects bad output.
12. **What stops it asking "how did you solve those disagreements?" after "I have no
    disagreements"?** Prompt guidance plus a code check that drops a presupposing
    follow-up after a denial.
13. **Why generate resume questions before the interview, not live?** Each needs
    rubric hints so the scoring is fair and reproducible.
14. **What is prompt injection and how do you handle it?** Text in a resume or a
    transcript might contain instructions. It is wrapped as data, markup is stripped,
    and outputs are validated.

## Scoring and the report
15. **How is the content score computed?** The model fills a rubric; code computes
    the score from a documented formula. Evidence quotes must appear in the answer.
16. **Why the median of three ratings?** The same answer once scored 68 and 78.
    The median reduces variance; the spread is shown.
17. **How can the written review be trusted?** Quotes must be exact substrings,
    numbers must come from the facts, and unusable output means no review is shown.
18. **Why is the rule-based grammar checker not the main feedback?** Spoken text breaks
    parser assumptions. E2 measures its precision; E4 measures the review's notes.
19. **What does the report not claim?** It does not detect emotion, and scores are
    approximate guides.

## Evaluation and honesty
20. **What are the main limitations?** One speaker, free-tier models, missing fillers,
    heuristic thresholds. (Report section 5)
21. **What would you do with more time?** More speakers, semantic end-of-answer
    detection, constrained dynamic questions, a technical round.
22. **How did you use AI assistance, and how did you verify the work?** State it
    plainly: assistants designed and wrote code under your direction; you ran it,
    recorded the data, tested, and checked results against real output.
23. **Was the bigger speech model always better?** No. In one interview the larger
    report-pass model heard the name wrongly where the live model, helped by the
    candidate's own hints, was right. So the system keeps the live text for an answer
    when it contains more of the candidate's names and terms.
24. **How accurate was the written review, and how do you know?** One session, nine
    claims checked against the transcript: seven correct (0.78). Language notes were
    right 2 of 3 times, against 1 of 21 for the old rule-based flags. State that the
    verdicts came from an AI annotator and the sample is small. (`../results/e4_review.md`)
25. **Why was the rule-based grammar section removed from the report?** Measured
    precision was about 5% on spoken transcripts. A feature that is mostly wrong
    should not be shown. The verified language notes replace it. (`../results/e2_grammar.md`)
26. **Why do the scores run high, and does it matter?** The model ranked answers like
    an independent reader (Spearman 0.92) but scored about 11 points higher on all 10.
    Use the scores to compare answers and track progress, not as an absolute grade.
