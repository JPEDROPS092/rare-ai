"""Patient phenotype encoding: curated HPO terms with source traceability."""

from __future__ import annotations

from pathlib import Path

import yaml

from rareai.phenotype.ontology import HPOOntology


def load_patient_terms(yaml_path: Path | str, ontology: HPOOntology) -> set[str]:
    """Load curated patient terms; returns only approved, primary, non-obsolete IDs."""
    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    terms: set[str] = set()
    for entry in data.get("hpo_terms", []):
        if not entry.get("approved", False):
            continue
        term = ontology.primary(entry["id"].strip())
        if term in ontology.obsolete:
            continue
        terms.add(term)
    return terms


def prune_to_specific(terms: set[str], ontology: HPOOntology) -> set[str]:
    """Drop terms that are ancestors of another patient term (redundant specificity)."""
    result = set(terms)
    for term in terms:
        for other in terms:
            if other != term and term in ontology.ancestors(other, include_self=False):
                result.discard(term)
                break
    return result


def write_patient_template(out_path: Path | str, suggestions: list[dict]) -> Path:
    """Write the curation template with suggested terms (human approves/edits)."""
    entries = []
    for sug in suggestions:
        entries.append(
            {
                "id": sug["id"],
                "label": sug["label"],
                "source_text": sug.get("source_text", ""),
                "approved": False,
                "comment": "",
            }
        )
    payload = {
        "description": "Curated patient HPO encoding. Set approved: true for terms"
        " confirmed against the clinical phenotype text. Every term must keep its"
        " source_text for auditability.",
        "hpo_terms": entries,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out
