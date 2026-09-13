# ALIGNED: Adaptive aLignment for Inconsistent Genetic kNowledgE and Data

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **ALIGNED** from the paper:

> **Adaptive Data-Knowledge Alignment in Genetic Perturbation Prediction**  
> Yuanfang Xiang, Lun Ai  
> *In proceedings of the 14th International Conference on Learning Representations, 2026*  
> <!--[OpenReview](https://openreview.net/forum?id=CxLaZWbUjc)-->Accepted by ICLR 2026 | [arXiv](https://arxiv.org/abs/2510.00512)

## Overview

ALIGNED is a neuro-symbolic framework for predicting genetic perturbation responses that adaptively aligns data-driven learning with biological knowledge. Built on the Abductive Learning (ABL) paradigm, ALIGNED:

- **Handles inconsistencies** between data and knowledge bases with trade-off between simultaneously imperfect sources
- **Performs systematic knowledge refinement** to improve biological networks and enables the evolution of domain knowledge bases
- **Achieves state-of-the-art performance** while substantially improving biological interpretability

## Installation

### Requirements
- Python ≥ 3.8
- CUDA-compatible GPU (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/yfxiang0112/Aligned.git
cd Aligned

# Create conda environment
conda env create -f environment.yml
conda activate aligned

# Install the package
pip install -e .
```

## Reproducibility

### Section 4.1: Perturbation Prediction on Benchmark Datasets

Run experiments for all three human datasets using the Quick Start commands above.

```bash
# Norman dataset (non-random split)
python experiments/ex1_bench/run_benchmark.py \
    --data_name='norman' \
    --model_save_name='norman_gnn' \
    --model_type='GNN' \
    --device='cuda:0' \
    --seed=42 \
    --random_split=False

# Dixit dataset (random split)
python experiments/ex1_bench/run_benchmark.py \
    --data_name='dixit' \
    --model_save_name='dixit_gnn' \
    --model_type='GNN' \
    --device='cuda:0' \
    --seed=42 \
    --random_split=True

# Adamson dataset (random split)
python experiments/ex1_bench/run_benchmark.py \
    --data_name='adamson' \
    --model_save_name='adamson_gnn' \
    --model_type='GNN' \
    --device='cuda:0' \
    --seed=42 \
    --random_split=True
```

**Arguments:**
- `--data_name`: Dataset selection from `['norman', 'dixit', 'adamson']`
- `--model_save_name`: File name to save trained models in `./models`
- `--model_type`: Neural component architecture: `['GNN', 'MLP']`
- `--device`: CUDA device ordinal (e.g., `'cuda:0'`)
- `--seed`: Random seed for reproducibility
- `--random_split`: Use random test split (see paper Section 4.1 for details)

Results, figures and trained models in `results/ex1_aligned/`, baseline comparison results in `results/ex1_baselines/`.

**Baseline comparisons**: For state-of-the-art baseline methods in Section 4.1, we use baseline method implementations and results from:

> Constantin Ahlmann-Eltze, Wolfgang Huber and Simon Anders. "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines" *Nature Methods* (2025). DOI: [10.1038/s41592-025-02772-6](https://doi.org/10.1038/s41592-025-02772-6)
> 
> Code: [const-ae/linear_perturbation_prediction-Paper](https://github.com/const-ae/linear_perturbation_prediction-Paper)

We thank the authors for making their implementations publicly available.


### Section 4.2: Knowledge Refinement of Gene Regulatory Networks

```bash
# Network refinement with incompleteness injection
python experiments/ex2_refinement/run_refinement.py

# Evaluate with gene set recovery
python experiments/ex2_refinement/eval_gene_set_recovery.py
```


Results in `results/ex2_refinement/`.

### Section 4.3: Perturbation Prediction on Bacterial Genome
```bash
python experiments/ex3_ecoli/run_ecoli.py \
    --data_name='ncbi-sra' \
    --model_save_name='ecoli_gnn' \
    --model_type='GNN' \
    --device='cuda:0' \
    --seed=42
```


Results and trained models in `results/ex3_ecoli/`.

### Section 4.4: Ablation Studies

Ablation results are located in `results/ex4_ablations/`.

## Project Structure

```
Aligned/
├── aligned/                  # Core package (renamed from egoal)
│   ├── abl.py                 # Abductive learning main loop
│   ├── learner_adap.py        # Neural learner with adaptor (renamed from refl)
│   ├── reasoner.py            # Symbolic reasoner (knowledge base)
│   └── utils.py               # Utility functions
│
├── experiments/              # Experimental scripts
│   ├── ex1_bench/             # Section 4.1: Benchmark experiments
│   ├── ex2_refinement/        # Section 4.2: Knowledge refinement
│   ├── ex3_ecoli/             # Section 4.3: E. coli experiments
│   └── ex4_ablation/          # Section 4.4: Ablation studies
│
├── dataset/                  # Data files
│   ├── human/                 # Human benchmark datasets
│   ├── ncbi-sra/              # E. coli RNA-seq data
│   └── precise1k/             # E. coli PRECISE-1K data
│
├── rules/                    # Knowledge bases
│   ├── ecoli/                 # E. coli regulatory knowledge base
│   └── human/                 # Human regulatory knowledge base
│
├── results/                  # Experiment results and figure generation scripts
│   ├── ex1_aligned/           # Main benchmark results
│   ├── ex1_baselines/         # Baseline comparisons
│   ├── ex2_refinement/        # Refinement experiment results
│   ├── fig1_incons/           # Inconsistency visualization
│   ├── fig3_radar/            # Radar plots
│   ├── fig4_line/             # Performance curves
│   └── fig5_refine/           # Refinement results
│
├── models/                    # Trained models
├── log/                       # Training logs
└── scripts/                   # Utility and preprocessing scripts
```

## Citation

If you use ALIGNED in your research, please cite:

```bibtex
@inproceedings{xiang2026aligned,
  title={Adaptive Data-Knowledge Alignment in Genetic Perturbation Prediction},
  author={Xiang, Yuanfang and Ai, Lun},
  booktitle={The 14th International Conference on Learning Representations (ICLR)},
  year={2026}
}
```



## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See [LICENSE](LICENSE) file for details.

## Contact

For questions, feedback, or bug reports, please open an issue on GitHub. 
For direct inquiries, please find contact information on the GitHub profile.
