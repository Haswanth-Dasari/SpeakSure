# SpeakSure: a voice-first mock interview coach

*Draft for the NLP course report. E2, E3 and E4 results were produced with an AI
assistant as annotator and are marked provisional until you add a human spot-check
(`docs/EVALUATION.md`). Nothing in this draft is a result unless a number is shown.*

## 1. Introduction

Practising spoken interviews is hard without a partner who listens, reacts and
gives honest feedback. SpeakSure is a local-first system that conducts a spoken
mock interview: it asks a question aloud, listens, decides when the answer has
finished, transcribes it, replies with a follow-up or the next question, and at the
end writes a feedback report. The project began as a broad English-practice
platform and was deliberately reduced to this single goal so that the core could
be built and measured properly (see the project journey document).

**Contributions.** (1) A working end-to-end spoken interview with a browser
interface. (2) A measured choice of speech-recognition models (E1). (3) A resume-aware
interviewer whose questions come from a verified candidate profile. (4) A report
that opens with a plain-language review whose quotes and numbers are verified
against the data, followed by the evidence. (5) An evaluation toolkit.

## 2. System overview

The pipeline has four stages.

1. **Listening.** The browser records the answer. A voice-activity detector
   calibrates to room noise, requires sustained speech to start, and ends the answer
   after a silence of 2.5 seconds. Thinking time is measured from the first audio
   frame to the onset of confirmed speech.
2. **Transcription.** A local Whisper model transcribes each answer for the live
   conversation. After the interview, the saved audio is transcribed again with a
   larger model for the report, because the report has no latency limit.
3. **Interviewing.** A language model decides, under rules enforced by code,
   whether to ask a follow-up or move on. Code limits follow-ups, forbids repeats,
   keeps spoken sentences short, and rejects follow-ups that presuppose something the
   candidate has just denied.
4. **Analysis and report.** Offline analyzers measure delivery (pace, pauses,
   thinking time, hedges) and a rubric-based model rates content. A final step
   writes a short review from the computed facts.

Design principle: the language model proposes, code disposes. Scores are computed
by documented formulas, quotes must appear in the transcript, and every model
output is validated before use.

## 3. Methods

### 3.1 Speech recognition (E1)
Models: Whisper base, small, medium and large-v3 (faster-whisper). Data: 31 clips
(371 reference words) recorded by the author on several microphones, with typed
reference text. Metric: word error rate with a 95% bootstrap interval resampling
whole clips; real-time factor (RTF) for speed.

### 3.2 Resume understanding
Text is extracted locally (PDF, DOCX, TXT). Email addresses, phone numbers and links
are removed before any text is sent to a model. A model returns a structured profile
(skills, projects, education, experience, achievements); each item is then checked
against the resume text and dropped if it is not grounded. The user reviews and edits
the profile. Distinctive terms (project names, tools, institutions) are added to the
speech recogniser's vocabulary.

### 3.3 Interview planning
A planner builds a varied sequence: a warm-up, a shuffled middle mixing resume and
general questions from a pool of 16, and a closing question; negative questions are
not placed first. Resume questions are generated before the interview so each has
rubric hints for scoring; deterministic templates are used if generation fails.

### 3.4 Content scoring (E3)
Each answer is rated against a fixed rubric (relevance, coverage of expected points,
specificity, clarity, and STAR structure for behavioural questions). Because model
ratings vary between runs, each question is rated three times and the median result
is reported with the spread.

### 3.5 Written review (E4)
A model is given only computed facts and the transcripts, as data. It writes a
headline, an overview, strengths, improvements with a concrete tip, language notes,
a practice plan and per-question comments. Validation rejects unsupported numbers and
drops any quote that is not an exact substring of the candidate's own words.

## 4. Experiments and results

### 4.1 E1: speech recognition (completed)

| Model | Raw WER (95% CI) | RTF |
| --- | ---: | ---: |
| base | 0.205 [0.137, 0.284] | 0.022 |
| small | 0.156 [0.093, 0.240] | 0.039 |
| medium | 0.094 [0.045, 0.164] | 0.079 |
| large-v3 | 0.086 [0.041, 0.148] | 0.140 |

Paired over clips, medium beats small by 0.053 WER (95% interval 0.017 to 0.092),
a clear difference; medium versus large-v3 differs by 0.020 (interval -0.012 to
0.055), which is not distinguishable with this sample. Names were the hardest
category for every model. Recognition with a Bluetooth headset in hands-free mode
was consistently worse than with the laptop microphone array (large-v3: 0.213
versus 0.060), though with few clips.

**Decision.** Medium runs live (good accuracy at about 0.08 real-time factor);
large-v3 re-transcribes the audio for the report.

### 4.2 E3: content scoring against an independent rating

*Provisional: the reference ratings come from an AI assistant working blind from
the transcripts, not from a human. A human spot-check is still to be added.*

On 10 comparable answers from 3 sessions, the model's scores rank answers almost
the same way as the independent ratings (Spearman 0.92, 95% bootstrap interval 0.66
to 0.99), but they are systematically more generous: higher on all 10 answers, by
11.3 points on average, with band agreement on 7 of 10. Clarity agreed least
(rho 0.47). Repeated ratings of the same answer varied by 6.3 to 7.0 points, which
supports reporting the median of three. Details: `docs/results/e3_content.md`.

### 4.3 E2: rule-based grammar flags

*Provisional: verdicts are from an AI assistant, with every verdict and reason
auditable in `docs/results/`.*

Of 22 flags in 3 sessions, 1 was a real error, 1 was unsure and 20 were false alarms
(precision 0.05; articles 0.10, subject-verb agreement 0.00). Causes were idioms
("from beginning to end"), mass nouns, parser confusion on spoken text and speech
recognition errors blamed on the speaker. At least five clear errors went unflagged.
A rule-based parser designed for written text therefore does not transfer to spoken
transcripts, and the report hides it by default. Details: `docs/results/e2_grammar.md`.

### 4.4 E4: written review accuracy

*Provisional: one session, nine claims, verdicts from an AI assistant (auditable in
`docs/results/`).*

Seven of nine claims were correct (0.78): language notes 2 of 3, strengths 2 of 3,
improvements 3 of 3. The two errors were a natural sentence wrongly "corrected" and
a strength overstated from an unrelated quote. Language notes were right far more
often than the rule-based flags of E2 (0.67 versus 0.05). The check also exposed two
defects that were then fixed: a quote cut off mid-word was accepted, and the larger
report-pass speech model sometimes lost a name the live model had right, so the live
transcript is now kept for an answer when it contains more of the candidate's names
and terms. Details: `docs/results/e4_review.md`.

## 5. Limitations

- One speaker, one accent, mostly one microphone: results may not generalise.
- Speech recognition mis-hears names and technical terms; the report warns the
  reader, and recognised-word confidence is not proof of correctness.
- Filler words such as "um" are largely removed by Whisper, so filler counts are
  underestimates.
- Content scores come from a language model and are approximate; the median of
  three ratings reduces but does not remove variation.
- The report-pass speech model is not always better: it can lose names that the live
  model gets right with the candidate's hints, so a per-answer guard keeps the better one.
- Pace, pause and thinking-time thresholds are heuristics, not standards.
- Delivery measurements are speech measurements, not emotion detection.
- The interviewer asks questions from a pool; it does not invent new topics mid
  interview, and its follow-up wording depends on a free-tier language model.

## 6. Ethics, privacy and AI assistance

Speech recognition and speech synthesis run locally. A language-model service
(Groq) receives transcripts and the redacted resume profile, never audio. Contact
details are removed before upload and the full resume is not stored. The report
states what its numbers do and do not mean.

**AI assistance (check your course policy and edit this paragraph).** The project
was designed and built with the help of AI assistants and coding agents. The author
ran all code, recorded all audio, wrote or supervised the tests, collected the
evaluation data, made the design decisions and verified results against real output.

## 7. Conclusion and future work

The system conducts a realistic spoken interview and produces useful feedback
entirely from a laptop plus a free language-model key. Future work: semantic
end-of-answer detection, dynamic question generation constrained by rubric hints,
more speakers and microphones, pitch and intonation analysis, and a spoken
technical (data structures) round.
