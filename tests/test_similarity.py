"""HPO ontology/similarity tests."""

from __future__ import annotations

import math

from rareai.phenotype.ontology import parse_hp_obo
from rareai.phenotype.similarity import HPOSimilarity, load_annotations


def test_ontology_ancestors(mini_obo):
    ont = parse_hp_obo(mini_obo)
    assert ont.label("HP:0000001") == "Specific symptom A"
    ancestors = ont.ancestors("HP:0000001")
    assert "HP:0000118" in ancestors
    assert "HP:0000001" in ancestors  # include_self


def test_information_content(mini_obo, mini_p2g):
    ont = parse_hp_obo(mini_obo)
    rows = load_annotations(mini_p2g)
    sim = HPOSimilarity(ont, rows)

    # Root sees all 22 genes; specific term A sees {G1, G2}.
    assert sim.ic["HP:0000118"] == 0.0
    assert math.isclose(sim.ic["HP:0000001"], -math.log2(2 / 22), rel_tol=1e-9)

    # Resnik of a term with itself is its IC; root-relative pairs share only root.
    assert sim.resnik("HP:0000001", "HP:0000001") == sim.ic["HP:0000001"]
    assert sim.resnik("HP:0000001", "HP:0000002") == 0.0


def test_lin_similarity(mini_obo, mini_p2g):
    ont = parse_hp_obo(mini_obo)
    sim = HPOSimilarity(ont, load_annotations(mini_p2g))

    assert sim.lin("HP:0000001", "HP:0000001") == 1.0  # identical terms
    assert sim.lin("HP:0000001", "HP:0000002") == 0.0  # MICA is root (IC 0)
    # A vs broad term C: MICA root -> 0
    assert sim.lin("HP:0000001", "HP:0000003") == 0.0


def test_gene_similarity_and_ranking(mini_obo, mini_p2g):
    ont = parse_hp_obo(mini_obo)
    sim = HPOSimilarity(ont, load_annotations(mini_p2g))

    patient = {"HP:0000001"}  # Specific symptom A
    # G2 carries exactly {A}: symmetric best-match similarity = 1.0
    assert sim.patient_gene_similarity(patient, "G2") == 1.0
    # G1 carries {A, B}: forward 1.0, reverse penalizes the unmatched B -> 0.75
    assert sim.patient_gene_similarity(patient, "G1") == 0.75
    # Broad genes share only the root -> ~0 similarity
    assert sim.patient_gene_similarity(patient, "G10") == 0.0

    ranking = sim.rank_genes(patient, min_similarity=0.0)
    assert ranking[0][0] == "G2"  # exact-set match ranks first
    top_genes = {g for g, s in ranking if s > 0.99}
    assert top_genes == {"G2"}

    best = sim.best_disease_for_gene(patient, "G1")
    assert best is not None
    assert best.similarity == 1.0
    assert best.disease_id in ("OMIM:100001", "OMIM:100002")
