# Dataset Catalog - SageBio/mva-hackathon-2026-data

Generated: 2026-09-06T02:55:07.335851+00:00  
Files: 13  
Total size: 79.1 GB

| File | Size | Type | Purpose | Track 1 | Track 2 |
|---|---|---|---|---|---|
| `.gitattributes` | 2.5 KB | Git meta | Repository metadata | - | - |
| `Challenge_Clinical_Phenotype_1.docx` | 16.5 KB | DOCX | Clinical phenotype description (HPO encoding source) | YES | YES |
| `README.md` | 5.0 KB | Markdown | Dataset documentation and download instructions | ref | ref |
| `WGS_EX2312012_HGWCNDSX7.vcf.gz` | 300.6 MB | VCF (bgzip) | Whole-genome variant calls (SNV/indel) for proband | YES | YES |
| `WGS_EX2312012_HGWCNDSX7.vcf.gz.tbi` | 2.2 MB | Tabix/csi index | Random-access index for the VCF | YES | YES |
| `WGS_EX2312012_HGWCNDSX7_S16_L001_R1_001.fastq.gz` | 9.7 GB | FASTQ (gz) | Raw WGS reads (paired-end lanes); only needed for re-calling | optional | optional |
| `WGS_EX2312012_HGWCNDSX7_S16_L001_R2_001.fastq.gz` | 10.3 GB | FASTQ (gz) | Raw WGS reads (paired-end lanes); only needed for re-calling | optional | optional |
| `WGS_EX2312012_HGWCNDSX7_S16_L002_R1_001.fastq.gz` | 9.7 GB | FASTQ (gz) | Raw WGS reads (paired-end lanes); only needed for re-calling | optional | optional |
| `WGS_EX2312012_HGWCNDSX7_S16_L002_R2_001.fastq.gz` | 10.2 GB | FASTQ (gz) | Raw WGS reads (paired-end lanes); only needed for re-calling | optional | optional |
| `WGS_EX2312012_HGWCNDSX7_S16_L003_R1_001.fastq.gz` | 9.5 GB | FASTQ (gz) | Raw WGS reads (paired-end lanes); only needed for re-calling | optional | optional |
| `WGS_EX2312012_HGWCNDSX7_S16_L003_R2_001.fastq.gz` | 9.9 GB | FASTQ (gz) | Raw WGS reads (paired-end lanes); only needed for re-calling | optional | optional |
| `WGS_EX2312012_HGWCNDSX7_S16_L004_R1_001.fastq.gz` | 9.6 GB | FASTQ (gz) | Raw WGS reads (paired-end lanes); only needed for re-calling | optional | optional |
| `WGS_EX2312012_HGWCNDSX7_S16_L004_R2_001.fastq.gz` | 10.0 GB | FASTQ (gz) | Raw WGS reads (paired-end lanes); only needed for re-calling | optional | optional |

## Notes

- Gated dataset: file downloads require an accepted access request plus `HF_TOKEN`.
- Minimum Track 1 working set: VCF + index + clinical phenotype DOCX (~318 MB).
- FASTQ files (~84.7 GB) are only required for re-calling (e.g. mosaicism-aware
  analysis) and are never downloaded automatically.
