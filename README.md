# Macon

Macon is an innovative approach that utilizes contrastive learning to improve protein mutation representation and predict their effects on protein-protein interactions (PPIs).

## Background

Mutations in protein sequences can significantly alter protein-protein interactions (PPIs), leading to various functional consequences. Understanding and predicting these effects is crucial for biomedical research and therapeutic strategies. However, current tools like MIPPI have limitations in accuracy and capturing nuanced sequence information.

## Study Overview

We introduce Macon, a novel approach utilizing contrastive learning to enhance protein mutation representation and predict their effects on PPIs. Our validation demonstrated the advantages of contrastive learning in these tasks. We evaluated various deep learning architectures, ultimately selecting one-dimensional convolution as the foundational model. Additionally, we assessed the effectiveness of different pre-trained protein language models as base encoders. Macon, trained with ProtTrans, outperformed existing tools, highlighting the potential of contrastive learning in improving mutation effect predictions.

## Code Structure

- `IMEx_preprocess.ipynb`: Data preprocessing script
- `onehot_embedding.py`: Encoding protein sequences using One-Hot encoding
- `pLMs_embedding.py`: Encoding using pre-trained protein language models
- `pretrain_mlp.py`: Pretraining using contrastive learning
- `PPI_train_macon_mlp.py`: Downstream task prediction of mutation effects on PPIs using Macon approach
- `PPI_train_onehot.py`: Downstream task prediction using One-Hot encoding
- `PPI_xgb.py`: Downstream task prediction using XGBoost
- `t-SNE_visualization.py`: Visualization tool for displaying embedding results

## Installation

Please ensure you have the following dependencies installed:

- Python 3.8+
- pandas
- numpy
- scikit-learn
- tensorflow or pytorch
- matplotlib
- seaborn
- jupyter
