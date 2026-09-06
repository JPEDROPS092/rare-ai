"""HPO term suggestions from free clinical text (curation assistant, no LLM)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from rareai.phenotype.ontology import HPOOntology

_HP_RE = re.compile(r"HP:\d{7}")


@dataclass
class Suggestion:
    id: str
    label: str
    matched_text: str
    source_text: str
    match_type: str  # explicit_code | label | synonym


def suggest_terms(text: str, ontology: HPOOntology) -> list[Suggestion]:
    """Match explicit HP codes, labels and EXACT synonyms against text lines."""
    suggestions: list[Suggestion] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lower_lines = [line.lower() for line in lines]

    # 1) Explicit HPO codes present verbatim in the document.
    for line in lines:
        for code in _HP_RE.findall(line):
            suggestions.append(
                Suggestion(
                    id=ontology.primary(code),
                    label=ontology.label(code),
                    matched_text=code,
                    source_text=line,
                    match_type="explicit_code",
                )
            )

    # 2) Label / synonym containment matches (deterministic string matching).
    for term, label in ontology.labels.items():
        if term in ontology.obsolete or len(label) < 5:
            continue
        needle = label.lower()
        for idx, lower in enumerate(lower_lines):
            if needle in lower:
                suggestions.append(
                    Suggestion(
                        id=term,
                        label=label,
                        matched_text=label,
                        source_text=lines[idx],
                        match_type="label",
                    )
                )
                break  # one occurrence per term is enough for a suggestion

    for term, syns in ontology.synonyms.items():
        if term in ontology.obsolete:
            continue
        for syn in syns:
            if len(syn) < 6:
                continue
            needle = syn.lower()
            for idx, lower in enumerate(lower_lines):
                if needle in lower:
                    suggestions.append(
                        Suggestion(
                            id=term,
                            label=ontology.labels.get(term, term),
                            matched_text=syn,
                            source_text=lines[idx],
                            match_type="synonym",
                        )
                    )
                    break
            break  # one synonym per term is enough

    # De-duplicate by (id, match_type).
    seen: set[tuple[str, str]] = set()
    unique: list[Suggestion] = []
    for sug in suggestions:
        key = (sug.id, sug.match_type)
        if key not in seen:
            seen.add(key)
            unique.append(sug)
    return unique


def suggestions_to_dicts(suggestions: list[Suggestion]) -> list[dict]:
    return [asdict(s) for s in suggestions]
