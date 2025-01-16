# Multi-view Graph Invariant Learning for Drug-Drug Interaction Prediction

## Introduction

Drug-Drug Interaction(DDI) prediction is essential for evaluating a new drug's side effects and adverse interactions before the clinical application. The latest research applies multiview knowledge to enhance the model's generalization ability to predict new drug interactions, i.e., unknown Drug-Drug Interaction (uDDI). However, a new drug's feature inevitably encounters the feature-shift problem because the trained models have not previously learned knowledge of the new drug. This significantly decreases the accuracy of the uDDI prediction. To this end, this work tries to extract the invariant features of known drugs to alleviate the impact of the feature-shift problem on the prediction of uDDI.In detail, first, the GNN models embed multiview knowledge graphs, including drug-chemical entities,drug-substructures, drug-drug interactions, and molecular structures, into drug features. Then, an invariant feature corresponding to the new drug is learned from the knowledge graph of the previous drugs. After that, according to its knowledge, a variant feature corresponding to the new drug is embedded through the GNN models. Finally, the variant and invariant drug features are fused to predict the DDI. Extensive experiments on real-world drug datasets show that the proposed method achieves new state-of-the-art records on new drug DDI prediction tasks. 

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

Please see the dataset.

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
3. **Proposes a Model with Strong Generalization**: The MGIL-DDI model performs superiorly in predicting DDI events.

## Contact

For any queries or issues, please start a issue.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
