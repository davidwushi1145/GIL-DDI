# Multi-modal Graph Invariant Learning for Drug-Drug Interaction Prediction

## Introduction

This project focuses on the prediction of Drug-Drug Interactions (DDI) using a novel model called MGIL-DDI (Multi-modal Graph Invariant Learning for Drug-Drug Interaction Prediction). The model integrates invariant and variant features of drugs to predict DDIs more accurately, addressing the domain-shift problem encountered with new drugs. The approach uses Graph Neural Networks (GNNs) and a self-attention mechanism to embed multi-modal knowledge graphs into comprehensive drug features.

## Prerequisites

Ensure you have the following software and packages installed:

- Python 3.8 or later
- PyTorch 1.12.1
- CUDA (for GPU support)
- Additional Python libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`

You can install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Dataset

Please see dataset.

## Usage

### Preparing the Data

Ensure that your dataset is correctly formatted and placed in the appropriate directory. Each part of the dataset should be structured as follows:

- **Part 1**: Drug-Drug Interaction events
- **Part 2**: Drug SMILES representations
- **Part 3**: DDI matrix
- **Part 4**: Molecular ACCess System (MACCS) keys

### Training the Model

The model can be trained using the following command:

```bash
python train.py --epochs 120 --learning_rate 0.001 --batch_size 1024 --embedding_dim 256 --neighborhood_size 6 --weight_decay 1e-8 --dropout_rate 0.3 --attention_heads 8
```

**Parameters:**

- `--epochs`: Number of training epochs.
- `--learning_rate`: Learning rate for the optimizer.
- `--batch_size`: Size of the batches used in training.
- `--embedding_dim`: Dimension of the drug embeddings.
- `--neighborhood_size`: Size of the neighborhood sample.
- `--weight_decay`: Weight decay (L2 regularization) parameter.
- `--dropout_rate`: Dropout rate for regularization.
- `--attention_heads`: Number of heads in the attention mechanism.

## Experimental Results

The model has been evaluated on three tasks:

1. **Task 1**: Prediction of unobserved interaction events between known drugs.
2. **Task 2**: Prediction of interaction events between known drugs and new drugs.
3. **Task 3**: Prediction of interaction events between new drugs.

The MGIL-DDI model consistently outperforms existing approaches in various metrics, including accuracy, AUPR, AUC, F1 score, precision, and recall.

## Contributions

1. **Addressing the Domain-Shift Problem**: Fusing invariant features of known drugs with new drug features.
2. **Fusion of Invariant and Variant Features**: Combining invariant features learned from existing drug knowledge with variant features specific to new drugs.
3. **Proposes a Model with Strong Generalization**: The MGIL-DDI model shows superior performance in predicting DDI events.

## Contact

For any queries or issues, please start a issue.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.