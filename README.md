# Poisoned Attack for Graph Contrastive Learning Model

The implementation of paper

## Requirements

pytorch 1.7.1

torch_geometric 1.6.3

## Usage

1. Produce poisoned graph

```
python PAGCL.py --dataset Cora 
```

2. Run baseline model

```
python baseline_attacks.py --dataset Cora
```

3. Node Classification

```
python train.py --dataset Cora --attack_method PAGCL
```

4. Link Prediction

```
python linkprediction.py --dataset Cora --attack_method PAGCL
```

