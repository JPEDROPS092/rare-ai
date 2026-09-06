"""MVA context derived from HPO resources.

Mosaic Variegated Aneuploidy (MVA) associated genes and diseases are looked up
at runtime from the official phenotype-to-gene annotations - this module only
provides the lookup, never a fixed answer.
"""

from __future__ import annotations

from rareai.phenotype.similarity import HPOSimilarity

MVA_KEYWORDS = ("mosaic variegated aneuploidy",)


def mva_diseases(similarity: HPOSimilarity) -> list[tuple[str, str]]:
    """Diseases whose name contains MVA keywords (from official annotations)."""
    hits: list[tuple[str, str]] = []
    for disease_id, name in similarity.disease_names.items():
        if any(keyword in name.lower() for keyword in MVA_KEYWORDS):
            hits.append((disease_id, name))
    return sorted(hits)


def mva_genes(similarity: HPOSimilarity) -> list[str]:
    """Genes associated with MVA-like diseases (derived, not hard-coded)."""
    genes: set[str] = set()
    for disease_id, _ in mva_diseases(similarity):
        for gene, diseases in similarity.gene_to_diseases.items():
            if disease_id in diseases:
                genes.add(gene)
    return sorted(genes)
