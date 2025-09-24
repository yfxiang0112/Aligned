# EGOAL: predicting gene Expression with GenOme-scale knowledge and Abductive Learning

This repository is
`ALIGNED: Adaptive aLignment of Inconsistent Genetic kNowledgE and Data`,
a variant of EGOAL that focuses on balancing data-knowledge inconsistencies.

## Reproducibility

### Setup

```
cd egoal
conda env create -f environment.yml
conda activate egoal
```

### Run ALIGNED on human genome benchmark dataset

```
python egoal/main_hsa.py --data_name='norman' --model_save_name='...' --model_type='GNN' --device='cuda:...' --seed=... --random_split=False
```

- data\_name: select benchmark dataset among `['norman', 'dixit', 'adamson']`.
- model\_save\_name: file name to save trained models in `./models`.
- model\_type: select model of the neural component in ALIGNED,
    currently support `['GNN', 'MLP']` mentioned in section 4.1.
- device: specify cuda device ordinal to train ALIGNED
- seed: random seed in training
- random\_split: use random splitted test set.
    To reproduce results in section 4.1, use `False` for `norman` dataset and
    `True` for `dixit` and `adamson` dataset.

### Git-ignored datasets

The datasets in `dataset/human/` are git-ignored due to excessive size,
and can be obtained from following link:
```
https://box.nju.edu.cn/f/d141fed10400452197ae/?dl=1
```

### Figures and Experiment Results

All results and trained models of ALIGNED in the benchmark experiment
(section 4.1, 4.3) are at `data_anal/experiment_results`,
and compared methods are at `data_anal/pert_benchmark`.

Results of network refinement experiment (section 4.2)
are at `data_anal/refine`.

Scripts to generate figures in the paper are at `plots`.
