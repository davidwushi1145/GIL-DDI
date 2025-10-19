# GIL-DDI: Multi-view Graph Invariant Learning for Drug-Drug Interaction Prediction

## Overview

**GIL-DDI** is a novel deep learning framework designed to predict Drug-Drug Interactions (DDIs), with a particular focus on interactions involving new drugs (unknown DDI or uDDI). By leveraging invariant learning principles and multi-view graph attention networks, GIL-DDI addresses the critical feature-shift problem that occurs when predicting interactions for drugs not seen during training.
![](https://raw.githubusercontent.com/davidwushi1145/photo2/main/202510172209488.jpeg)
## Abstract

Drug-Drug Interaction (DDI) prediction is essential for evaluating the side effects of a new drug and adverse interactions before clinical application. The latest research applies multi-view data to enhance the generalization ability of models to predict new drug interactions, mainly unknown Drug-Drug Interaction (uDDI). However, a new drug's feature inevitably encounters the feature-shift problem; the trained models have not previously learned information about the new drug, significantly decreasing the uDDI prediction's accuracy.

Thus, we proposed the **GIL-DDI model** that tries to extract the invariant features of known drugs, alleviating the impact of the feature-shift problem on the prediction of uDDI. In essence, a graph attention network (GAT) initially embeds multi-view knowledge graphs of known drugs, capturing features from chemical entities, substructures, and interactions. Inspired by invariant learning, we subsequently construct a robust feature space of these stable, known-drug characteristics. In the context of a novel pharmaceutical agent, the GAT is employed to initially embed its unique characteristics. Subsequently, these features are fused with the invariant features borrowed from the most similar known drugs in the feature space. This fusion enables the model to enhance the representation of the new drug, thereby addressing issues such as data scarcity and feature-shift problems.

Extensive experiments on real-world drug datasets indicate that the proposed method achieves new state-of-the-art records on new drug DDI prediction tasks.

## Key Features

- **Invariant Feature Learning**: Extracts stable, transferable features from known drugs to improve predictions for new drugs
- **Multi-view Knowledge Graphs**: Integrates multiple drug representations including:
  - Drug-chemical entities
  - Drug substructures
  - Drug-drug interactions
  - Molecular structures (SMILES)
  - MACCS keys (Molecular ACCess System)
- **Graph Attention Networks (GAT)**: Captures complex relationships in drug knowledge graphs
- **Feature Fusion Strategy**: Combines invariant features from similar known drugs with variant features of new drugs
- **State-of-the-art Performance**: Achieves superior results on unknown DDI prediction tasks

## Prerequisites

Ensure you have the following software and packages installed:

- Python 3.8 or later
- PyTorch 1.12.1
- CUDA (for GPU support, optional but recommended)
- Additional Python libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `rdkit`

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/GIL-DDI.git
cd GIL-DDI
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

## Dataset

### Data Acquisition

**Important Notice:** This project does not include the DrugBank database. To use this tool, you need to obtain a license from DrugBank.

#### Obtaining DrugBank Data

1. Visit the [DrugBank website](https://www.drugbank.com)
2. Register and apply for an academic license
3. Once you have obtained the license, follow the instructions provided by DrugBank to download and import the data
4. Follow the data preparation guide in `dataset/DATA_FORMAT.md`

#### Citation

Please use the following citation for the version of DrugBank data you are using:

> Knox C, Wilson M, Klinger CM, et al. DrugBank 6.0: the DrugBank Knowledgebase for 2024. Nucleic Acids Res. 2024 Jan 5;52(D1): D1265-D1275. doi: 10.1093/nar/gkad976.

### Dataset Structure

Ensure that your dataset is correctly formatted and placed in the appropriate directory. Each part of the dataset should be structured as follows:

Required files:
- `dataset/event.db`: SQLite database with DDI events
- `dataset/dataset1.txt`: Drug-chemical entity relationships
- `dataset/dataset2.txt`: Drug substructure relationships
- `dataset/dataset3.txt`: Drug-drug interactions
- `dataset/dataset4.txt`: Molecular features (SMILES, MACCS keys)

## Usage

### Quick Start

```bash
# Task 1: Known drug interactions
python train.py --task 1 --epochs 120 --learning_rate 0.01 --batch_size 1024 --embedding_dim 128

# Task 2: Known-new drug interactions
python train.py --task 2 --epochs 120 --learning_rate 0.001 --batch_size 1024 --embedding_dim 256

# Task 3: New drug interactions
python train.py --task 3 --epochs 120 --learning_rate 0.001 --batch_size 1024 --embedding_dim 256
```

### Detailed Usage

#### Task 1: Prediction of Unobserved Interactions Between Known Drugs

```bash
python train.py \
    --task 1 \
    --epochs 120 \
    --learning_rate 0.01 \
    --batch_size 1024 \
    --embedding_dim 128 \
    --neighborhood_size 6 \
    --weight_decay 1e-8 \
    --dropout_rate 0.3 \
    --attention_heads 8
```

Alternatively, you can run the task-specific script directly:

```bash
cd code
python GILIP-task1.py --epoches 120 --lr 0.01 --batch_size 1024 --embedding_num 128
```

#### Task 2: Prediction of Interactions Between Known Drugs and New Drugs

```bash
python train.py \
    --task 2 \
    --epochs 120 \
    --learning_rate 0.001 \
    --batch_size 1024 \
    --embedding_dim 256 \
    --neighborhood_size 8
```

Or:

```bash
cd code
python GILIP-task2.py --epoches 120 --lr 0.001 --batch_size 1024 --embedding_num 256
```

#### Task 3: Prediction of Interactions Between New Drugs

```bash
python train.py \
    --task 3 \
    --epochs 120 \
    --learning_rate 0.001 \
    --batch_size 1024 \
    --embedding_dim 256 \
    --neighborhood_size 6
```

Or:

```bash
cd code
python GILIP-task3.py --epoches 120 --lr 0.001 --batch_size 1024 --embedding_num 256
```

#### Available Parameters:

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--task` | Task number | 1 | 1, 2, 3 |
| `--epochs` | Number of training epochs | 120 | 100, 500, 1000, 2000 |
| `--learning_rate` (or `--lr`) | Learning rate for optimizer | Task-dependent | 1e-5 to 0.01 |
| `--batch_size` | Size of training batches | 1024 | 128, 256, 512, 1024, 2048 |
| `--embedding_dim` | Dimension of drug embeddings | Task-dependent | 32, 64, 128, 256 |
| `--neighborhood_size` | Size of neighborhood sample | Task-dependent | 4, 6, 8, 10, 16 |
| `--weight_decay` | Weight decay (L2 regularization) | 1e-8 | 1e-8 to 1e-1 |
| `--dropout_rate` | Dropout rate for regularization | 0.3 | 0.0 to 1.0 |
| `--attention_heads` | Number of heads in attention mechanism | 8 | 1, 2, 4, 8 |
| `--device` | Device to use | auto | auto, cpu, cuda, cuda:0, etc. |
| `--seed` | Random seed for reproducibility | 0 | Any integer |

## Project Structure

```
GIL-DDI/
├── code/                      # Task-specific training scripts
│   ├── GILIP-task1.py        # Known drug DDI prediction
│   ├── GILIP-task2.py        # Known-new drug DDI prediction
│   └── GILIP-task3.py        # New drug DDI prediction
├── model/                     # Model architectures
│   ├── task1/                # Task 1 models
│   │   ├── gnn_model.py     # GNN implementations
│   │   └── layers.py         # Attention and fusion layers
│   ├── task2/                # Task 2 models
│   │   ├── gnn_model.py     # GNN implementations
│   │   └── layers.py         # Attention and fusion layers
│   └── task3/                # Task 3 models
│   │   ├── gnn_model.py     # GNN implementations
│   │   └── layers.py         # Attention and fusion layers
├── util/                      # Utility functions
│   └── utils.py              # General utilities
├── dataset/                   # Dataset directory
│   ├── README.md             # Dataset acquisition guide
│   └── DATA_FORMAT.md        # Data format specification
├── train.py                  # Main training script
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── LICENSE                  # MIT License
└── README.md                # This file
```

## Methodology

### Architecture Overview

1. **Multi-view Graph Embedding**: GAT models embed multi-view knowledge graphs of known drugs
2. **Invariant Feature Space Construction**: Build a robust feature space capturing stable characteristics of known drugs
3. **New Drug Feature Extraction**: GAT embeds unique variant features of new drugs
4. **Feature Fusion**: Variant features of new drugs are fused with invariant features from the most similar known drugs
5. **DDI Prediction**: Enhanced drug representations are used to predict drug-drug interactions

### Addressing the Feature-Shift Problem

The key innovation of GIL-DDI is its approach to handling new drugs:
- **Known drugs**: The model learns invariant features that are stable across different drugs
- **New drugs**: Instead of relying solely on limited information, the model:
  1. Extracts variant features specific to the new drug
  2. Identifies similar known drugs in the invariant feature space
  3. Borrows invariant features from these similar drugs
  4. Fuses variant and invariant features for robust prediction

## Experimental Results

The model has been evaluated on three tasks:

1. **Task 1**: Prediction of unobserved interaction events between known drugs
2. **Task 2**: Prediction of interaction events between known drugs and new drugs
3. **Task 3**: Prediction of interaction events between new drugs

The GIL-DDI model consistently outperforms existing approaches across various metrics, including:
- Accuracy
- AUPR (Area Under Precision-Recall curve)
- AUC (Area Under ROC Curve)
- F1 score (Micro/Macro)
- Precision (Micro/Macro)
- Recall (Micro/Macro)

## Contributing

We welcome contributions! 

## Contributions

1. **Addressing the Feature-Shift Problem**: Novel approach to fusing invariant features of known drugs with new drug features to handle the distribution shift in uDDI prediction
2. **Invariant Learning Framework**: First application of invariant learning principles to DDI prediction, extracting stable drug characteristics transferable to new drugs
3. **Multi-view Graph Integration**: Comprehensive integration of diverse drug knowledge sources through graph attention networks
4. **State-of-the-art Performance**: The GIL-DDI model demonstrates superior generalization ability and achieves new benchmarks in predicting DDI events involving new drugs

## Contact

For any queries or issues, please open an issue in this repository.

## License

The code is distributed under the CC BY-NC-SA 4.0 License. See [LICENSE](LICENSE) for more information.

## Citation

If you use this code or model in your research, please cite:

```bibtex
@article{gil-ddi,
  title={GIL-DDI: Multi-view Graph Invariant Learning for Drug-Drug Interaction Prediction},
}
```

## Acknowledgments

We acknowledge DrugBank for providing the drug interaction database. Please ensure you have the appropriate license before using their data.

**DrugBank Citation:**
```bibtex
@article{drugbank2024,
  title={DrugBank 6.0: the DrugBank Knowledgebase for 2024},
  author={Knox, C and Wilson, M and Klinger, CM and others},
  journal={Nucleic Acids Research},
  volume={52},
  number={D1},
  pages={D1265--D1275},
  year={2024},
  publisher={Oxford University Press}
}
```
