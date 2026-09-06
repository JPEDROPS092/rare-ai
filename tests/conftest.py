"""Shared fixtures: synthetic VCF/GTF/OBO - no network, no real data."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

MINI_VCF = """##fileformat=VCFv4.2
##FILTER=<ID=LowQual,Description="Low quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
##FORMAT=<ID=PGT,Number=1,Type=String,Description="Phased genotype">
##contig=<ID=1,length=200000>
##contig=<ID=2,length=200000>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1
1\t1000\trs1\tA\tG\t50\tPASS\t.\tGT:DP:GQ:AD\t0/1:20:40:10,10
1\t2000\t.\tG\tTA\t60\tPASS\t.\tGT:DP:GQ:AD\t1/1:30:50:2,28
2\t3000\t.\tC\tT,G\t70\tPASS\t.\tGT:DP:GQ:AD\t0/1:25:45:12,10,3
2\t4000\t.\tT\tC\t30\tLowQual\t.\tGT:DP:GQ:AD\t0/1:15:35:8,7
1\t5000\t.\tA\tT\t80\tPASS\t.\tGT:DP:GQ:AD:PGT\t0|1:22:44:11,11:0|1
1\t6000\t.\tA\tT\t80\tPASS\t.\tGT:DP:GQ:AD:PGT\t1|0:22:44:11,11:1|0
"""

MINI_GTF = """#!genome-build GRCh38
1\tensembl\tgene\t1000\t1100\t.\t+\t.\tgene_id "GENE1"; gene_name "GENE1"; gene_type "protein_coding";
1\tensembl\texon\t1020\t1060\t.\t+\t.\tgene_id "GENE1"; gene_name "GENE1";
1\tensembl\tgene\t5000\t7000\t.\t+\t.\tgene_id "GENE2"; gene_name "GENE2"; gene_type "protein_coding";
1\tensembl\texon\t5050\t5100\t.\t+\t.\tgene_id "GENE2"; gene_name "GENE2";
1\tensembl\texon\t6000\t6100\t.\t+\t.\tgene_id "GENE2"; gene_name "GENE2";
2\tensembl\tgene\t3000\t3200\t.\t+\t.\tgene_id "GENE3"; gene_name "GENE3"; gene_type "protein_coding";
2\tensembl\texon\t3050\t3100\t.\t+\t.\tgene_id "GENE3"; gene_name "GENE3";
"""

MINI_OBO = """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0000001
name: Specific symptom A
is_a: HP:0000118

[Term]
id: HP:0000002
name: Specific symptom B
is_a: HP:0000118

[Term]
id: HP:0000003
name: Broad symptom
is_a: HP:0000118
"""

MINI_P2G_LINES = [
    "hpo_id\thpo_name\tentrez_id\tgene_symbol\tdisease_id\tdisease_name",
    "HP:0000001\tA\t1\tG1\tOMIM:100001\tDisease A",
    "HP:0000001\tA\t2\tG2\tOMIM:100001\tDisease A",
    "HP:0000002\tB\t1\tG1\tOMIM:100002\tDisease B",
]
# 20 genes annotated only to the broad term HP:0000003
MINI_P2G_LINES += [
    f"HP:0000003\tC\t{i}\tG{i:02d}\tOMIM:100003\tBroad Disease" for i in range(3, 23)
]


@pytest.fixture()
def mini_vcf(tmp_path: Path) -> Path:
    path = tmp_path / "mini.vcf"
    path.write_text(MINI_VCF, encoding="utf-8")
    return path


@pytest.fixture()
def mini_gtf(tmp_path: Path) -> Path:
    path = tmp_path / "mini.gtf"
    path.write_text(MINI_GTF, encoding="utf-8")
    return path


@pytest.fixture()
def mini_obo(tmp_path: Path) -> Path:
    path = tmp_path / "hp_mini.obo"
    path.write_text(MINI_OBO, encoding="utf-8")
    return path


@pytest.fixture()
def mini_p2g(tmp_path: Path) -> Path:
    path = tmp_path / "phenotype_to_genes_mini.txt"
    path.write_text("\n".join(MINI_P2G_LINES) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def phenotype_scores() -> pl.DataFrame:
    """Phenotype scores frame matching the mini gene models."""
    return pl.DataFrame(
        {
            "gene": ["GENE1", "GENE2", "GENE3"],
            "phenotype_similarity": [0.9, 0.4, 0.2],
            "gene_disease_score": [0.8, 0.3, 0.1],
            "disease_id": ["OMIM:100001", "OMIM:9", "OMIM:9"],
            "disease_name": ["Disease A", "X", "X"],
        }
    )
