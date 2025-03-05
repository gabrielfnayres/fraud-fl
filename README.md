# LEX-GNN with Privacy-Preserving Features

## Overview
This project implements the LEX-GNN (Label-Exploring Graph Neural Network) architecture with privacy-preserving features using Apple's Private Federated Learning (PFL) framework. The implementation focuses on fraud detection in graph-structured data while maintaining privacy guarantees.

## Key Features
- Implementation of LEX-GNN architecture with Graph Attention Networks (GAT)
- Privacy-preserving node label prediction using PFL
- Support for multiple fraud detection datasets (Yelp, Amazon)
- Efficient neighbor sampling for large-scale graphs
- Label masking mechanisms to prevent information leakage


## Technical Details

### Model Architecture
- **LEXGNN**: Main model class implementing the Label-Exploring GNN
- **LEXGAT**: Graph attention mechanism for node feature learning
- **NodeLabelPredictor**: MLP-based label prediction
- **NodeLabelEmbedding**: Embedding layer for node labels

### Data Loading
- Supports Yelp and Amazon fraud detection datasets
- Implements efficient neighbor sampling with customizable sampling rates
- Features row-wise feature normalization
- Stratified data splitting for train/validation/test sets


## Installation
```bash
# Clone the repository
git clone [repository-url]

# Install dependencies with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh  # If you haven't installed uv yet
uv pip sync pyproject.toml
```

## References

### Research Papers
1. LEX-GNN: Label-Exploring Graph Neural Network for Accurate Fraud Detection (CIKM '24)
   - [Paper Link](https://dl.acm.org/doi/10.1145/3627673.3679956)
   - Implements novel label exploration techniques for fraud detection

### Related Projects
- [Apple PFL Research](https://github.com/apple/pfl-research)
  - Provides the privacy-preserving federated learning framework
  - Used for implementing secure model training

## License
This project is licensed under the MIT License - see the LICENSE file for details.
