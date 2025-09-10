# EGOAL: Gene Expression Prediction with Abductive Learning

## Project Overview
EGOAL predicts gene expression using Gene Ontology and Abductive Learning (ABL), combining neural networks with symbolic reasoning for regulatory network inference. The system operates on E. coli and human datasets using both regulatory and metabolic knowledge bases.

## Core Architecture

### Main Components
- **`egoal/abl.py`**: Central ABL loop implementing the iterative refinement between neural prediction and symbolic reasoning
- **`egoal/learner_refl.py`**: Neural networks (MLP/GNN) with reinforcement learning heads for reflection confidence
- **`egoal/reasoner.py`**: Knowledge base manager handling regulatory networks with transitive closure operations
- **`egoal/main_eco.py`** / **`egoal/main_hsa.py`**: Entry points for E. coli and human experiments

### Data Flow
1. **Pretrain** base learner on labeled data (`X_label`, `Y_label`)
2. **ABL Loop** (T iterations):
   - Predict pseudo-labels on unlabeled data (`X_unlabel`)
   - Generate confidence scores via reflection head (`R`)
   - Apply knowledge base deduction (`reasoner.deduce()`)
   - Merge predictions where confidence is high (`R_binary >= threshold`)
   - Refine knowledge base via sparse optimization
   - Retrain learner on augmented pseudo-labeled data

## Key Patterns

### GPU/CUDA Usage
- **Multiple device targeting**: Scripts use specific CUDA devices (`cuda:6`, `cuda:7`) - check device availability
- **Mixed precision**: CuPy for sparse operations, PyTorch for neural networks
- **Memory management**: Large matrices moved to GPU with `.to(device)`, use `.cpu()` for CPU operations

### Knowledge Base Handling
```python
# Regulatory KB initialization with positive/negative rules
reasoner = RegualtoryKB(pos_trn_pth='rules/regu_pos.npz', 
                        neg_trn_pth='rules/regu_neg.npz')
reasoner.closure_(T=closure, closure_type='weighted')  # Transitive closure

# Deduction: X @ KB -> {-1, 0, 1} ternary predictions
Y_deduction = reasoner.deduce(X_unlabel)
```

### Neural Architecture
- **Dual-head design**: Classification head (3-class softmax for {-1,0,1}) + reflection head (binary confidence)
- **Base learner types**: `'MLP'` (dense) or `'GNN'` (graph neural network via torch-geometric)
- **Label weighting**: Uses precomputed weights in `rules/label_weight.npy`

## Critical Workflows

### Environment Setup
```bash
conda env create -f environment.yml
conda activate egoal
pip install zoopt  # Required for knowledge refinement optimization
pip install -e .   # Install package in development mode
```

### Running Experiments
- **E. coli**: `python egoal/main_eco.py` (uses Precise1K training, NCBI-SRA testing)
- **Human**: `python egoal/main_hsa.py` (supports norman dataset)
- **Models saved**: `models/` directory with timestamped outputs

### Data Conventions
- **Input matrices**: `.npy` files with specific shapes (genes × conditions/regulators)
- **Knowledge bases**: `.npz` sparse matrices for positive/negative regulations
- **Ternary labels**: `{-1, 0, 1}` for down/no-change/up regulation
- **Dataset splits**: Separate `X_label`/`Y_label` for training, `X_test`/`Y_test` for evaluation

## Development Notes

### Performance Optimization
- **Sparse matrix ops**: Prefer CuPy CSR format for GPU matrix multiplication
- **Boolean operations**: PyTorch uses float tensors with `.clamp().round().bool()` pattern
- **Memory**: Large closure operations cached in `self.KB_P`, `self.KB_N` tensors

### Debugging Patterns
- **Logging**: Use `log_file` parameter for detailed ABL loop tracking
- **Visualization**: `data_anal/` contains plotting scripts for heatmaps, F1 scores
- **Intermediate outputs**: Save `Y_modified.npy`, `KB_before.npy`, `KB_after.npy` for analysis

### Common Gotchas
- **Device mismatch**: Ensure all tensors on same device before operations
- **Closure types**: `'naive'`, `'weighted'`, `'combined'` produce different KB behaviors
- **Index alignment**: `output_idx_list` filters KB columns to match label dimensions
- **Confidence thresholds**: R >= 0.6 threshold determines when to trust deduction over neural prediction

## Testing & Validation
- **Unit tests**: `scripts/test_*.py` files for individual components
- **Benchmarks**: `scripts/kb/test_matrix_*.py` for performance comparisons
- **Cross-validation**: `data_anal/cross_val.tsv` tracks experimental results
