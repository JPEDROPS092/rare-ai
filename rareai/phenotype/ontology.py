"""HPO ontology parser (hp.obo): terms, synonyms, ancestors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HPOOntology:
    """In-memory HPO ontology graph."""

    labels: dict[str, str]
    parents: dict[str, list[str]]
    alt_to_primary: dict[str, str]
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    obsolete: set[str] = field(default_factory=set)
    _ancestor_cache: dict[str, set[str]] = field(default_factory=dict, repr=False)

    def primary(self, term: str) -> str:
        """Map alt IDs to their primary term."""
        seen: set[str] = set()
        current = term
        while current in self.alt_to_primary and current not in seen:
            seen.add(current)
            current = self.alt_to_primary[current]
        return current

    def ancestors(self, term: str, include_self: bool = True) -> set[str]:
        """All ancestors (including self by default) via BFS over primary IDs."""
        root = self.primary(term)
        if root in self._ancestor_cache:
            cached = self._ancestor_cache[root]
            return cached | {root} if include_self else cached
        result: set[str] = set()
        stack = list(self.parents.get(root, []))
        while stack:
            current = self.primary(stack.pop())
            if current in result:
                continue
            result.add(current)
            stack.extend(self.parents.get(current, []))
        self._ancestor_cache[root] = result
        return result | {root} if include_self else result

    def label(self, term: str) -> str:
        return self.labels.get(term, self.labels.get(self.primary(term), term))


def parse_hp_obo(path: Path | str) -> HPOOntology:
    """Parse ``hp.obo`` into an :class:`HPOOntology`."""
    labels: dict[str, str] = {}
    parents: dict[str, list[str]] = {}
    alt_to_primary: dict[str, str] = {}
    synonyms: dict[str, list[str]] = {}
    obsolete: set[str] = set()

    current_id: str | None = None
    current_name: str | None = None
    current_parents: list[str] = []
    current_alts: list[str] = []
    current_syns: list[str] = []
    is_obsolete = False

    def commit() -> None:
        nonlocal current_id, current_name, current_parents, current_alts, current_syns, is_obsolete
        if current_id:
            if current_name:
                labels[current_id] = current_name
            parents[current_id] = current_parents
            for alt in current_alts:
                alt_to_primary[alt] = current_id
            if current_syns:
                synonyms[current_id] = current_syns
            if is_obsolete:
                obsolete.add(current_id)
        current_id = None
        current_name = None
        current_parents = []
        current_alts = []
        current_syns = []
        is_obsolete = False

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                commit()
            elif line.startswith("id: HP:"):
                current_id = line[4:].strip()
            elif line.startswith("name:"):
                current_name = line[5:].strip()
            elif line.startswith("alt_id:"):
                current_alts.append(line[7:].strip())
            elif line.startswith("is_a:"):
                target = line[5:].split("!")[0].strip()
                current_parents.append(target)
            elif line.startswith("relationship: part_of HP:"):
                target = line[len("relationship: part_of ") :].split("!")[0].strip()
                current_parents.append(target)
            elif line.startswith("synonym:"):
                text = line[len("synonym:") :].strip()
                if "EXACT" in text:
                    quoted = text.split('"')
                    if len(quoted) >= 2:
                        current_syns.append(quoted[1])
            elif line.startswith("is_obsolete: true"):
                is_obsolete = True
    commit()

    return HPOOntology(
        labels=labels,
        parents=parents,
        alt_to_primary=alt_to_primary,
        synonyms=synonyms,
        obsolete=obsolete,
    )
