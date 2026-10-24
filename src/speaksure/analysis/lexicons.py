"""Limited heuristic vocabularies for transcript-based fluency indicators."""

# Limited heuristic list of common spoken hesitation fillers; not a complete vocabulary.
HESITATION_FILLERS = frozenset(
    {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "mm"}
)

# Limited heuristic list of discourse markers; phrases are lowercase token tuples.
DISCOURSE_MARKERS = (
    ("basically",),
    ("actually",),
    ("literally",),
    ("you", "know"),
    ("i", "mean"),
)

# Limited heuristic list of hedges; phrases are lowercase token tuples.
HEDGES = (
    ("maybe",),
    ("perhaps",),
    ("probably",),
    ("possibly",),
    ("somewhat",),
    ("might",),
    ("i", "think"),
    ("i", "guess"),
    ("i", "believe"),
    ("i", "suppose"),
    ("kind", "of"),
    ("sort", "of"),
    ("not", "sure"),
)
