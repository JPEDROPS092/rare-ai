"""Semantic similarity over HPO: information content, Resnik and Lin.

Gene- and disease-level annotations come from the official
``phenotype_to_genes.txt`` release file; information content is computed from
gene counts propagated to ancestors (standard HPO-similarity practice).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rareai.phenotype.ontology import HPOOntology

ROOT = "HP:0000118"  # Phenotypic abnormality


@dataclass
class GeneDiseaseMatch:
    """Best disease match for a gene against the patient phenotype."""

    disease_id: str
    disease_name: str
    similarity: float


class HPOSimilarity:
    """Resnik/Lin semantic similarity backed by gene-count information content."""

    def __init__(self, ontology: HPOOntology, annotations: list[dict[str, str]]) -> None:
        """``annotations``: rows of phenotype_to_genes.txt
        (hpo_id, gene_symbol, disease_id, disease_name)."""
        self.ont = ontology
        self.gene_to_terms: dict[str, set[str]] = {}
        self.disease_to_terms: dict[str, set[str]] = {}
        self.disease_names: dict[str, str] = {}
        self.gene_to_diseases: dict[str, set[str]] = {}
        self._lin_cache: dict[tuple[str, str], float | None] = {}
        self._build(annotations)

    # ------------------------------------------------------------------ build
    def _build(self, annotations: list[dict[str, str]]) -> None:
        term_genes_direct: dict[str, set[str]] = {}
        for row in annotations:
            term = self.ont.primary(row["hpo_id"].strip())
            gene = row["gene_symbol"].strip()
            disease = row.get("disease_id", "").strip()
            if not term.startswith("HP:") or not gene:
                continue
            term_genes_direct.setdefault(term, set()).add(gene)
            self.gene_to_terms.setdefault(gene, set()).add(term)
            if disease:
                self.disease_to_terms.setdefault(disease, set()).add(term)
                if row.get("disease_name"):
                    self.disease_names[disease] = row["disease_name"]
                self.gene_to_diseases.setdefault(gene, set()).add(disease)

        # Propagate gene sets upward so ancestor terms include descendant genes.
        term_genes: dict[str, set[str]] = {t: set(g) for t, g in term_genes_direct.items()}
        for term in term_genes_direct:
            for ancestor in self.ont.ancestors(term, include_self=False):
                term_genes.setdefault(ancestor, set()).update(term_genes_direct[term])

        total = float(len(term_genes.get(ROOT, set())) or 1)
        self.ic: dict[str, float] = {}
        for term, genes in term_genes.items():
            if len(genes) > 0:
                self.ic[term] = -math.log2(len(genes) / total)

    # -------------------------------------------------------------- pairwise
    def resnik(self, t1: str, t2: str) -> float:
        ancestors_1 = self.ont.ancestors(t1)
        ancestors_2 = self.ont.ancestors(t2)
        common = ancestors_1 & ancestors_2
        best = max((self.ic.get(a, 0.0) for a in common), default=0.0)
        return best

    def lin(self, t1: str, t2: str) -> float | None:
        key = (t1, t2) if t1 <= t2 else (t2, t1)
        if key in self._lin_cache:
            return self._lin_cache[key]
        ic1 = self.ic.get(self.ont.primary(t1))
        ic2 = self.ic.get(self.ont.primary(t2))
        value: float | None = None
        if ic1 and ic2 and (ic1 + ic2) > 0:
            mica = self.resnik(t1, t2)
            value = 2.0 * mica / (ic1 + ic2)
        self._lin_cache[key] = value
        return value

    # ---------------------------------------------------------------- set sim
    def set_similarity(
        self,
        query_terms: set[str],
        target_terms: set[str],
        method: str = "lin",
    ) -> float:
        """Symmetric best-match average similarity between two term sets."""
        if not query_terms or not target_terms:
            return 0.0

        def best_match(term: str) -> float:
            if method == "resnik":
                return max((self.resnik(term, t) for t in target_terms), default=0.0)
            values = [self.lin(term, t) for t in target_terms]
            values = [v for v in values if v is not None]
            return max(values, default=0.0)

        def best_match_reverse(term: str) -> float:
            if method == "resnik":
                return max((self.resnik(term, t) for t in query_terms), default=0.0)
            values = [self.lin(term, t) for t in query_terms]
            values = [v for v in values if v is not None]
            return max(values, default=0.0)

        forward = sum(best_match(t) for t in query_terms) / len(query_terms)
        reverse = sum(best_match_reverse(t) for t in target_terms) / len(target_terms)
        # Normalize Lin (0..1 by construction); Resnik scaled by max IC for comparability.
        if method == "resnik":
            max_ic = max(self.ic.values(), default=1.0) or 1.0
            forward /= max_ic
            reverse /= max_ic
        return (forward + reverse) / 2.0

    # ------------------------------------------------------------- gene level
    def patient_gene_similarity(self, patient_terms: set[str], gene: str) -> float:
        gene_terms = self.gene_to_terms.get(gene, set())
        return self.set_similarity(patient_terms, gene_terms)

    def patient_disease_similarity(self, patient_terms: set[str], disease: str) -> float:
        disease_terms = self.disease_to_terms.get(disease, set())
        return self.set_similarity(patient_terms, disease_terms)

    def best_disease_for_gene(self, patient_terms: set[str], gene: str) -> GeneDiseaseMatch | None:
        best: GeneDiseaseMatch | None = None
        for disease in self.gene_to_diseases.get(gene, ()):
            sim = self.patient_disease_similarity(patient_terms, disease)
            if best is None or sim > best.similarity:
                best = GeneDiseaseMatch(
                    disease_id=disease,
                    disease_name=self.disease_names.get(disease, disease),
                    similarity=sim,
                )
        return best

    def rank_genes(
        self, patient_terms: set[str], min_similarity: float = 0.0
    ) -> list[tuple[str, float]]:
        """All annotated genes ranked by phenotype similarity (descending)."""
        scores = [
            (gene, self.patient_gene_similarity(patient_terms, gene)) for gene in self.gene_to_terms
        ]
        scores = [(g, s) for g, s in scores if s >= min_similarity]
        scores.sort(key=lambda gs: (-gs[1], gs[0]))
        return scores


def load_annotations(path) -> list[dict[str, str]]:
    """Load phenotype_to_genes.txt (HPO official release file)."""
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        header = None
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if header is None:
                header = [p.strip().lower().replace(" ", "_") for p in parts]
                continue
            if len(parts) < 5:
                continue
            row = dict(zip(header, parts, strict=False))
            rows.append(
                {
                    "hpo_id": row.get("hpo_id", ""),
                    "hpo_name": row.get("hpo_name", ""),
                    "gene_symbol": row.get("gene_symbol", ""),
                    "disease_id": row.get("disease_id", ""),
                    "disease_name": row.get("disease_name", ""),
                }
            )
    return rows
